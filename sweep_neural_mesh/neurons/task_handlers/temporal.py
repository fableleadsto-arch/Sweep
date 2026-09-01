"""
Temporal Handler — date/time reasoning and temporal logic.

Handles:
  - Date math: add/subtract days, calculate differences
  - Timeline construction: order events chronologically
  - Chronological reasoning: "what happened before/after X?"
  - Duration calculation: "how long between X and Y?"
  - Day-of-week: "what day was X?"
  - Age calculation: "how old is X?"
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass(frozen=True, slots=True)
class TemporalResult:
    """Structured result from temporal reasoning."""
    answer: str
    confidence: float
    method: str
    steps: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Historical events database (expanded) ─────────────────────
EVENTS: list[tuple[str, str, str]] = [
    ("1776-07-04", "US Independence", "Declaration of Independence signed"),
    ("1789-07-14", "French Revolution", "Storming of the Bastille"),
    ("1859-11-24", "Origin of Species", "Darwin published On the Origin of Species"),
    ("1876-03-10", "Telephone", "Alexander Graham Bell patented the telephone"),
    ("1879-10-21", "Light Bulb", "Thomas Edison demonstrated the incandescent light bulb"),
    ("1889-03-31", "Eiffel Tower", "Eiffel Tower completed for World's Fair"),
    ("1903-12-17", "First Flight", "Wright brothers' first powered flight"),
    ("1914-07-28", "World War I", "Austria-Hungary declares war on Serbia"),
    ("1918-11-11", "WWI Ends", "Armistice signed, World War I ends"),
    ("1928-09-28", "Penicillin", "Alexander Fleming discovers penicillin"),
    ("1939-09-01", "World War II", "Germany invades Poland"),
    ("1945-05-08", "V-E Day", "Victory in Europe Day"),
    ("1945-08-06", "Hiroshima", "Atomic bomb dropped on Hiroshima"),
    ("1945-09-02", "WWII Ends", "Japan surrenders, World War II ends"),
    ("1953-04-25", "DNA Structure", "Watson and Crick publish DNA double helix"),
    ("1957-10-04", "Sputnik", "Soviet Union launches Sputnik, first artificial satellite"),
    ("1961-04-12", "First Human in Space", "Yuri Gagarin becomes first human in space"),
    ("1969-07-20", "Moon Landing", "Apollo 11 — first humans on the Moon"),
    ("1971-10-29", "Email", "Ray Tomlinson sends first network email"),
    ("1981-08-12", "IBM PC", "IBM Personal Computer released"),
    ("1989-03-12", "World Wide Web", "Tim Berners-Lee proposes the World Wide Web"),
    ("1989-11-09", "Berlin Wall Falls", "Berlin Wall falls"),
    ("1990-04-24", "Hubble Space Telescope", "Hubble Space Telescope launched"),
    ("1991-08-06", "World Wide Web", "First website goes live"),
    ("1997-05-11", "Deep Blue", "IBM Deep Blue defeats Garry Kasparov"),
    ("2001-09-11", "9/11", "September 11 attacks"),
    ("2003-02-01", "Columbia Disaster", "Space Shuttle Columbia disintegrates on re-entry"),
    ("2004-02-04", "Facebook", "Facebook launched"),
    ("2006-03-21", "Twitter", "Twitter launches"),
    ("2007-06-29", "iPhone", "Apple iPhone released"),
    ("2008-09-15", "Financial Crisis", "Lehman Brothers files for bankruptcy"),
    ("2010-04-03", "iPad", "Apple iPad released"),
    ("2012-06-06", "Higgs Boson", "CERN announces discovery of Higgs boson"),
    ("2015-09-14", "Gravitational Waves", "LIGO detects gravitational waves"),
    ("2019-04-10", "Black Hole Image", "First image of a black hole"),
    ("2020-01-30", "COVID-19 Pandemic", "WHO declares COVID-19 a Public Health Emergency"),
    ("2022-11-30", "ChatGPT", "OpenAI launches ChatGPT"),
]


class TemporalHandler:
    """Handles temporal reasoning tasks."""

    def process(self, query: str, evidence: list[str] | None = None) -> TemporalResult:
        t0 = time.perf_counter()
        q = query.strip()
        ev = evidence or []

        result = self._try_date_math(q, t0)
        if result:
            return result

        result = self._try_duration(q, t0)
        if result:
            return result

        result = self._try_event_lookup(q, t0)
        if result:
            return result

        result = self._try_timeline(q, ev, t0)
        if result:
            return result

        result = self._try_chronological(q, ev, t0)
        if result:
            return result

        result = self._try_age_calculation(q, t0)
        if result:
            return result

        result = self._try_day_of_week(q, t0)
        if result:
            return result

        return TemporalResult(
            answer="", confidence=0.0, method="none",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ── Date Math ────────────────────────────────────────

    def _try_date_math(self, q: str, t0: float) -> TemporalResult | None:
        """Calculate date differences or date arithmetic."""
        q_lower = q.lower()

        # "How many days between DATE1 and DATE2?"
        days_between = re.search(
            r"how\s+(?:many\s+)?days?\s+between\s+(\w+\s+\d{1,2},?\s+\d{4})\s+and\s+(\w+\s+\d{1,2},?\s+\d{4})",
            q_lower,
        )
        if days_between:
            try:
                d1 = datetime.strptime(days_between.group(1).replace(",", ""), "%B %d %Y")
                d2 = datetime.strptime(days_between.group(2).replace(",", ""), "%B %d %Y")
                diff = abs((d2 - d1).days)
                return TemporalResult(
                    answer=f"{diff} days", confidence=0.95,
                    method="date_difference",
                    steps=[f"{d1.date()} to {d2.date()} = {diff} days"],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            except ValueError:
                pass

        # "What is 30 days after January 1, 2024?"
        add_days = re.search(
            r"(?:what\s+is\s+)?(\d+)\s+days?\s+(?:after|from)\s+(\w+\s+\d{1,2},?\s+\d{4})",
            q_lower,
        )
        if add_days:
            try:
                n = int(add_days.group(1))
                d = datetime.strptime(add_days.group(2).replace(",", ""), "%B %d %Y")
                result = d + timedelta(days=n)
                return TemporalResult(
                    answer=result.strftime("%B %d, %Y"), confidence=0.95,
                    method="date_addition",
                    steps=[f"{d.date()} + {n} days = {result.date()}"],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            except ValueError:
                pass

        return None

    # ── Duration ─────────────────────────────────────────

    def _try_duration(self, q: str, t0: float) -> TemporalResult | None:
        """Calculate duration between events."""
        q_lower = q.lower()

        # "How long between EVENT1 and EVENT2?"
        duration_match = re.search(
            r"how\s+long\s+between\s+(.+?)\s+and\s+(.+?)(?:\?)?$",
            q_lower,
        )
        if duration_match:
            event1 = duration_match.group(1).strip()
            event2 = duration_match.group(2).strip()

            # Look up events in database
            d1 = self._find_event_date(event1)
            d2 = self._find_event_date(event2)

            if d1 and d2:
                diff = abs((d2 - d1).days)
                years = diff // 365
                months = (diff % 365) // 30
                days = diff % 30
                parts = []
                if years:
                    parts.append(f"{years} year{'s' if years != 1 else ''}")
                if months:
                    parts.append(f"{months} month{'s' if months != 1 else ''}")
                if days:
                    parts.append(f"{days} day{'s' if days != 1 else ''}")
                duration_str = ", ".join(parts) if parts else "0 days"
                return TemporalResult(
                    answer=f"{duration_str} ({diff} days)", confidence=0.90,
                    method="duration",
                    steps=[f"{event1}: {d1.date()}", f"{event2}: {d2.date()}", f"Duration: {duration_str}"],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        return None

    # ── Event Lookup ─────────────────────────────────────

    def _try_event_lookup(self, q: str, t0: float) -> TemporalResult | None:
        """Look up dates for historical events."""
        q_lower = q.lower()

        # "When did EVENT happen?"
        when_match = re.search(
            r"when\s+(?:did|was|has)\s+(.+?)(?:\s+(?:happen|occur|start|end|take place))?\??$",
            q_lower,
        )
        if when_match:
            event_text = when_match.group(1).strip()
            date_str = self._find_event_date_str(event_text)
            if date_str:
                return TemporalResult(
                    answer=date_str, confidence=0.90,
                    method="event_lookup",
                    steps=[f"Event: {event_text}", f"Date: {date_str}"],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        # "What year was EVENT?"
        year_match = re.search(
            r"what\s+year\s+(?:was|did|has)\s+(.+?)(?:\s+(?:happen|occur|start|end))?\??$",
            q_lower,
        )
        if year_match:
            event_text = year_match.group(1).strip()
            date_str = self._find_event_date_str(event_text)
            if date_str:
                # Extract year
                year = re.search(r"\d{4}", date_str)
                if year:
                    return TemporalResult(
                        answer=year.group(), confidence=0.90,
                        method="event_lookup",
                        steps=[f"Event: {event_text}", f"Year: {year.group()}"],
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )

        return None

    # ── Timeline Construction ────────────────────────────

    def _try_timeline(self, q: str, ev: list[str], t0: float) -> TemporalResult | None:
        """Construct a timeline from evidence."""
        q_lower = q.lower()
        if not any(w in q_lower for w in ("timeline", "chronological", "order", "sequence")):
            return None

        # Extract dates from evidence
        events = []
        for e in ev:
            # Find dates in text
            dates = re.findall(r"(\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{4})", e)
            for d in dates:
                try:
                    if re.match(r"\d{4}-\d{2}-\d{2}", d):
                        dt = datetime.strptime(d, "%Y-%m-%d")
                    elif re.match(r"\d{4}$", d):
                        dt = datetime(int(d), 1, 1)
                    else:
                        dt = datetime.strptime(d.replace(",", ""), "%B %d %Y")
                    events.append((dt, e[:100]))
                except ValueError:
                    pass

        if len(events) >= 2:
            events.sort(key=lambda x: x[0])
            timeline = [
                f"{dt.strftime('%Y-%m-%d')}: {text}"
                for dt, text in events
            ]
            return TemporalResult(
                answer=f"Timeline: {len(events)} events", confidence=0.80,
                method="timeline",
                steps=timeline[:10],
                latency_ms=(time.perf_counter() - t0) * 1000,
                metadata={"event_count": len(events)},
            )

        return None

    # ── Chronological Ordering ───────────────────────────

    def _try_chronological(self, q: str, ev: list[str], t0: float) -> TemporalResult | None:
        """Determine what happened before/after something, or what happened first/last."""
        q_lower = q.lower()

        # "What happened first, X or Y?"
        first_match = re.search(
            r"what\s+(?:happened|occurred|was)\s+first[,:]?\s*(.+?)\s*\?*$",
            q_lower,
        )
        if first_match:
            events_text = first_match.group(1).strip()
            # Split by 'or'
            parts = re.split(r'\s+or\s+', events_text)
            if len(parts) >= 2:
                event_a = parts[0].strip().rstrip(',').strip()
                event_b = parts[1].strip().rstrip('?').strip()
                date_a = self._find_event_date(event_a)
                date_b = self._find_event_date(event_b)
                if date_a and date_b:
                    if date_a < date_b:
                        answer = f"{event_a} happened first ({date_a.strftime('%Y-%m-%d')})"
                    else:
                        answer = f"{event_b} happened first ({date_b.strftime('%Y-%m-%d')})"
                    return TemporalResult(
                        answer=answer, confidence=0.90,
                        method="chronological",
                        steps=[
                            f"{event_a}: {date_a.strftime('%Y-%m-%d')}",
                            f"{event_b}: {date_b.strftime('%Y-%m-%d')}",
                            f"Earlier: {event_a if date_a < date_b else event_b}",
                        ],
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )

        before_match = re.search(r"what\s+(?:happened|occurred)\s+before\s+(.+?)(?:\?)?$", q_lower)
        after_match = re.search(r"what\s+(?:happened|occurred)\s+after\s+(.+?)(?:\?)?$", q_lower)

        if before_match or after_match:
            target = (before_match or after_match).group(1).strip()
            target_date = self._find_event_date(target)
            if not target_date:
                return None

            related_events = []
            for date_str, name, desc in EVENTS:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    continue
                if before_match and dt < target_date:
                    related_events.append((dt, name, desc))
                elif after_match and dt > target_date:
                    related_events.append((dt, name, desc))

            related_events.sort(key=lambda x: x[0], reverse=bool(before_match))

            if related_events:
                direction = "before" if before_match else "after"
                steps = [f"{dt.strftime('%Y-%m-%d')}: {name}" for dt, name, _ in related_events[:5]]
                return TemporalResult(
                    answer=f"{len(related_events)} events {direction} {target}",
                    confidence=0.80,
                    method="chronological",
                    steps=[f"What happened {direction} {target}:"] + steps,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        return None

    # ── Age Calculation ──────────────────────────────────

    def _try_age_calculation(self, q: str, t0: float) -> TemporalResult | None:
        """Calculate ages from birth dates."""
        q_lower = q.lower()

        age_match = re.search(
            r"(?:how\s+old|what\s+is.*age)\s+(?:is|was)\s+(.+?)(?:\s+(?:in|on)\s+(.+?))?\??$",
            q_lower,
        )
        if age_match:
            subject = age_match.group(1).strip()
            # Look for a date in the query
            date_match = re.search(r"(\w+\s+\d{1,2},?\s+\d{4})", q)
            if date_match:
                try:
                    birth = datetime.strptime(date_match.group(1).replace(",", ""), "%B %d %Y")
                    now = datetime.now()
                    age_days = (now - birth).days
                    years = age_days // 365
                    return TemporalResult(
                        answer=f"{years} years old", confidence=0.90,
                        method="age_calculation",
                        steps=[f"Born: {birth.date()}", f"Now: {now.date()}", f"Age: {years}"],
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                except ValueError:
                    pass

        return None

    # ── Day of Week ──────────────────────────────────────

    def _try_day_of_week(self, q: str, t0: float) -> TemporalResult | None:
        """Determine the day of the week for a date."""
        q_lower = q.lower()

        dow_match = re.search(
            r"what\s+day\s+(?:was|is|of the week)\s+(\w+\s+\d{1,2},?\s+\d{4})",
            q_lower,
        )
        if dow_match:
            try:
                dt = datetime.strptime(dow_match.group(1).replace(",", ""), "%B %d %Y")
                day_name = dt.strftime("%A")
                return TemporalResult(
                    answer=f"{day_name}, {dt.strftime('%B %d, %Y')}",
                    confidence=0.99,
                    method="day_of_week",
                    steps=[f"{dt.date()} was a {day_name}"],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            except ValueError:
                pass

        return None

    # ── Helpers ──────────────────────────────────────────

    def _find_event_date(self, event_text: str) -> datetime | None:
        """Find a date for an event from the database."""
        event_lower = event_text.lower()
        # First pass: exact or substring match of full name/desc
        for date_str, name, desc in EVENTS:
            if (name.lower() in event_lower or
                event_lower in name.lower() or
                desc.lower() in event_lower or
                event_lower in desc.lower()):
                try:
                    return datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    pass
        # Second pass: require >= 50% word overlap (avoids "world war ii" matching "world war i")
        for date_str, name, desc in EVENTS:
            name_words = set(name.lower().split())
            desc_words = set(desc.lower().split())
            all_words = name_words | desc_words
            event_words = set(event_lower.split())
            overlap = len(all_words & event_words)
            if overlap >= max(2, len(all_words) * 0.5):
                try:
                    return datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    pass
        return None

    def _find_event_date_str(self, event_text: str) -> str | None:
        """Find a date string for an event."""
        event_lower = event_text.lower()
        for date_str, name, desc in EVENTS:
            if (name.lower() in event_lower or
                event_lower in name.lower() or
                desc.lower() in event_lower or
                event_lower in desc.lower()):
                return date_str
        # Second pass: word overlap
        for date_str, name, desc in EVENTS:
            name_words = set(name.lower().split())
            desc_words = set(desc.lower().split())
            all_words = name_words | desc_words
            event_words = set(event_lower.split())
            overlap = len(all_words & event_words)
            if overlap >= max(2, len(all_words) * 0.5):
                return date_str
        return None
