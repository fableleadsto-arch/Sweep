"""
Location Intelligence — geocoding, coordinate handling, geographic relationships.

Sweep's entity system can understand:
- Cities, countries, addresses
- Coordinates when available
- Places, organizations associated with places
- Events associated with places
- Geographic relationships (distance, containment)

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │           LOCATION INTELLIGENCE                      │
    │                                                     │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Location Extractor                          │  │
    │  │  (regex + NER for location mentions)         │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Geocoding Engine                            │  │
    │  │  (location name → coordinates)               │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Geographic Reasoning                        │  │
    │  │  (distance, containment, proximity)          │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Location Graph                              │  │
    │  │  (places, orgs, events → location map)       │  │
    │  └──────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Coordinates:
    """Geographic coordinates."""
    latitude: float
    longitude: float
    altitude: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"lat": self.latitude, "lon": self.longitude}


@dataclass
class Location:
    """A resolved location."""
    name: str
    coordinates: Coordinates | None = None
    country: str = ""
    region: str = ""
    city: str = ""
    location_type: str = ""  # city, country, address, landmark, etc.
    confidence: float = 0.8
    alternate_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "coordinates": self.coordinates.to_dict() if self.coordinates else None,
            "country": self.country,
            "region": self.region,
            "city": self.city,
            "type": self.location_type,
            "confidence": self.confidence,
        }


@dataclass
class GeographicRelation:
    """A geographic relationship between two locations."""
    location_a: str
    location_b: str
    relation_type: str  # "near", "contains", "bordering", "same_region", "far"
    distance_km: float | None = None
    confidence: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.location_a,
            "b": self.location_b,
            "relation": self.relation_type,
            "distance_km": self.distance_km,
            "confidence": self.confidence,
        }


@dataclass
class LocationEvidence:
    """Evidence associated with a location."""
    location_name: str
    evidence_text: str
    source: str = ""
    timestamp: str = ""
    entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location_name,
            "evidence": self.evidence_text[:200],
            "source": self.source,
            "entities": self.entities,
        }


@dataclass
class LocationAnalysisResult:
    """Result of location intelligence analysis."""
    locations: list[Location]
    relations: list[GeographicRelation]
    evidence_by_location: dict[str, list[LocationEvidence]]
    timeline: list[dict[str, Any]]
    analysis: str
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "location_count": len(self.locations),
            "relation_count": len(self.relations),
            "locations": [l.to_dict() for l in self.locations],
            "relations": [r.to_dict() for r in self.relations],
            "analysis": self.analysis,
        }


class LocationIntelligence:
    """
    Location intelligence engine.

    Extracts, resolves, and reasons about locations in text.
    """

    # ── Major world locations database ────────────────────────
    _KNOWN_LOCATIONS: dict[str, dict[str, Any]] = {
        # Countries
        "india": {"lat": 20.5937, "lon": 78.9629, "type": "country", "region": "south_asia"},
        "united states": {"lat": 37.0902, "lon": -95.7129, "type": "country", "region": "north_america"},
        "usa": {"lat": 37.0902, "lon": -95.7129, "type": "country", "region": "north_america"},
        "china": {"lat": 35.8617, "lon": 104.1954, "type": "country", "region": "east_asia"},
        "japan": {"lat": 36.2048, "lon": 138.2529, "type": "country", "region": "east_asia"},
        "united kingdom": {"lat": 55.3781, "lon": -3.4360, "type": "country", "region": "europe"},
        "uk": {"lat": 55.3781, "lon": -3.4360, "type": "country", "region": "europe"},
        "germany": {"lat": 51.1657, "lon": 10.4515, "type": "country", "region": "europe"},
        "france": {"lat": 46.2276, "lon": 2.2137, "type": "country", "region": "europe"},
        "australia": {"lat": -25.2744, "lon": 133.7751, "type": "country", "region": "oceania"},
        "brazil": {"lat": -14.2350, "lon": -51.9253, "type": "country", "region": "south_america"},
        "russia": {"lat": 61.5240, "lon": 105.3188, "type": "country", "region": "europe_asia"},
        "canada": {"lat": 56.1304, "lon": -106.3468, "type": "country", "region": "north_america"},
        "south korea": {"lat": 35.9078, "lon": 127.7669, "type": "country", "region": "east_asia"},
        "italy": {"lat": 41.8719, "lon": 12.5674, "type": "country", "region": "europe"},
        "spain": {"lat": 40.4637, "lon": -3.7492, "type": "country", "region": "europe"},
        "mexico": {"lat": 23.6345, "lon": -102.5528, "type": "country", "region": "north_america"},
        "nepal": {"lat": 28.3949, "lon": 84.1240, "type": "country", "region": "south_asia"},
        "pakistan": {"lat": 30.3753, "lon": 69.3451, "type": "country", "region": "south_asia"},
        # Cities
        "delhi": {"lat": 28.7041, "lon": 77.1025, "type": "city", "country": "india"},
        "new delhi": {"lat": 28.6139, "lon": 77.2090, "type": "city", "country": "india"},
        "dehradun": {"lat": 30.3165, "lon": 78.0322, "type": "city", "country": "india"},
        "mumbai": {"lat": 19.0760, "lon": 72.8777, "type": "city", "country": "india"},
        "bangalore": {"lat": 12.9716, "lon": 77.5946, "type": "city", "country": "india"},
        "london": {"lat": 51.5074, "lon": -0.1278, "type": "city", "country": "united kingdom"},
        "paris": {"lat": 48.8566, "lon": 2.3522, "type": "city", "country": "france"},
        "tokyo": {"lat": 35.6762, "lon": 139.6503, "type": "city", "country": "japan"},
        "new york": {"lat": 40.7128, "lon": -74.0060, "type": "city", "country": "united states"},
        "beijing": {"lat": 39.9042, "lon": 116.4074, "type": "city", "country": "china"},
        "berlin": {"lat": 52.5200, "lon": 13.4050, "type": "city", "country": "germany"},
        "sydney": {"lat": -33.8688, "lon": 151.2093, "type": "city", "country": "australia"},
        "dubai": {"lat": 25.2048, "lon": 55.2708, "type": "city", "country": "uae"},
        "singapore": {"lat": 1.3521, "lon": 103.8198, "type": "city", "country": "singapore"},
        "moscow": {"lat": 55.7558, "lon": 37.6173, "type": "city", "country": "russia"},
        "rome": {"lat": 41.9028, "lon": 12.4964, "type": "city", "country": "italy"},
        "madrid": {"lat": 40.4168, "lon": -3.7038, "type": "city", "country": "spain"},
        "toronto": {"lat": 43.6532, "lon": -79.3832, "type": "city", "country": "canada"},
        "seoul": {"lat": 37.5665, "lon": 126.9780, "type": "city", "country": "south korea"},
        "cairo": {"lat": 30.0444, "lon": 31.2357, "type": "city", "country": "egypt"},
        "kathmandu": {"lat": 27.7172, "lon": 85.3240, "type": "city", "country": "nepal"},
    }

    _LOCATION_PATTERNS = [
        r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'\bnear\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'\bfrom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'\bto\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'\b([A-Z][a-z]+),\s+([A-Z][a-z]+)\b',
        r'\b([A-Z][a-z]+)\s+(?:city|state|country|region)\b',
    ]

    def __init__(self) -> None:
        self._locations: dict[str, Location] = {}
        self._relations: list[GeographicRelation] = []
        self._evidence: dict[str, list[LocationEvidence]] = {}

    @staticmethod
    def _haversine_distance(c1: Coordinates, c2: Coordinates) -> float:
        """Calculate distance between two coordinate points in km."""
        R = 6371.0  # Earth radius in km
        lat1, lon1 = math.radians(c1.latitude), math.radians(c1.longitude)
        lat2, lon2 = math.radians(c2.latitude), math.radians(c2.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _resolve_location(self, name: str) -> Location:
        """Resolve a location name to coordinates and metadata."""
        name_lower = name.lower().strip()

        if name_lower in self._KNOWN_LOCATIONS:
            data = self._KNOWN_LOCATIONS[name_lower]
            return Location(
                name=name,
                coordinates=Coordinates(latitude=data["lat"], longitude=data["lon"]),
                country=data.get("country", ""),
                region=data.get("region", ""),
                city=name if data.get("type") == "city" else "",
                location_type=data.get("type", "unknown"),
                confidence=0.95,
            )

        return Location(
            name=name,
            location_type="unknown",
            confidence=0.4,
        )

    def extract_locations(self, text: str) -> list[Location]:
        """Extract all location mentions from text."""
        found: list[Location] = []
        seen_names: set[str] = set()

        for pat in self._LOCATION_PATTERNS:
            for m in re.finditer(pat, text):
                name = m.group(1) if m.lastindex else m.group(0)
                name = name.strip()
                if len(name) < 3 or name.lower() in seen_names:
                    continue
                seen_names.add(name.lower())
                loc = self._resolve_location(name)
                found.append(loc)
                self._locations[name.lower()] = loc

        return found

    def add_evidence(
        self,
        location_name: str,
        evidence_text: str,
        source: str = "",
        timestamp: str = "",
        entities: list[str] | None = None,
    ) -> None:
        """Associate evidence with a location."""
        key = location_name.lower()
        ev = LocationEvidence(
            location_name=location_name,
            evidence_text=evidence_text,
            source=source,
            timestamp=timestamp,
            entities=entities or [],
        )
        self._evidence.setdefault(key, []).append(ev)

    def compute_relations(
        self, threshold_km: float = 500.0
    ) -> list[GeographicRelation]:
        """Compute geographic relationships between all known locations."""
        self._relations.clear()
        locs = list(self._locations.values())

        for i, a in enumerate(locs):
            for b in locs[i + 1:]:
                if not a.coordinates or not b.coordinates:
                    continue

                dist = self._haversine_distance(a.coordinates, b.coordinates)

                if dist < 50:
                    rel = "very_near"
                elif dist < threshold_km:
                    rel = "near"
                elif dist < 2000:
                    rel = "same_region"
                else:
                    rel = "far"

                relation = GeographicRelation(
                    location_a=a.name,
                    location_b=b.name,
                    relation_type=rel,
                    distance_km=round(dist, 1),
                    confidence=0.9,
                )
                self._relations.append(relation)

        return self._relations

    def analyze(
        self, text: str, additional_evidence: list[str] | None = None
    ) -> LocationAnalysisResult:
        """Full location intelligence analysis of text."""
        import time as _time
        t0 = _time.perf_counter()

        locations = self.extract_locations(text)

        # Add evidence
        for loc in locations:
            self.add_evidence(loc.name, text)

        if additional_evidence:
            for ev in additional_evidence:
                ev_locations = self.extract_locations(ev)
                for loc in ev_locations:
                    self.add_evidence(loc.name, ev)

        # Compute relations
        self.compute_relations()

        # Build timeline
        timeline: list[dict[str, Any]] = []
        for loc in locations:
            if loc.coordinates:
                timeline.append({
                    "location": loc.name,
                    "coordinates": loc.coordinates.to_dict(),
                    "type": loc.location_type,
                    "country": loc.country,
                })

        # Analysis summary
        location_names = [l.name for l in locations]
        countries = list({l.country for l in locations if l.country})
        analysis = f"Found {len(locations)} locations"
        if countries:
            analysis += f" across {len(countries)} countries ({', '.join(countries[:5])})"
        analysis += f" with {len(self._relations)} geographic relationships"

        latency = (_time.perf_counter() - t0) * 1000

        return LocationAnalysisResult(
            locations=locations,
            relations=self._relations,
            evidence_by_location=dict(self._evidence),
            timeline=timeline,
            analysis=analysis,
            latency_ms=latency,
        )
