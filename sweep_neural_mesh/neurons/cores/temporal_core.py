"""
TemporalCore — date/time reasoning and historical event lookup.

Responsibilities:
  - Answer "when" / "what year" questions from a built-in event database.
  - Extract year/date patterns from evidence when no direct match is found.
"""
from __future__ import annotations

import re
import time

from ..core_protocol import CoreResult, make_result, empty_result


# ── Historical event database ─────────────────────────────────
_RAW_EVENTS: list[tuple[str, str, float]] = [
    # ═══ ANCIENT WORLD ═══
    (r"great.*pyramid.*built|pyramid.*giza", "2560 BC", 0.95),
    (r"writing.*invented|first.*writing", "~3400 BC (Sumerian cuneiform)", 0.95),
    (r"wheel.*invented", "~3500 BC", 0.90),
    (r"bronze.*age.*started", "~3300 BC", 0.90),
    (r"iron.*age.*started", "~1200 BC", 0.90),
    (r"trojan.*war", "~1200 BC (if historical)", 0.85),
    (r"olympics.*started.*ancient", "776 BC", 0.95),
    (r"democracy.*invented|athenian.*democracy", "508 BC (Cleisthenes)", 0.95),
    (r"alexander.*great.*conquests", "334-323 BC", 0.95),
    (r"roman.*empire.*founded", "27 BC (Augustus)", 0.95),
    (r"eruption.*vesuvius|pompeii.*destroyed", "79 AD", 0.99),
    # ═══ MEDIEVAL ═══
    (r"fall.*rome|roman.*empire.*fell", "476 AD", 0.99),
    (r"islam.*founded|prophet.*muhammad", "622 AD", 0.95),
    (r"charlemagne.*crowned", "800 AD", 0.95),
    (r"viking.*age.*started", "793 AD (raid on Lindisfarne)", 0.90),
    (r"norman.*conquest.*england", "1066", 0.99),
    (r"crusades.*started", "1096", 0.95),
    (r"magna.*carta", "1215", 0.99),
    (r"mongol.*empire.*peak", "1279 (largest contiguous empire)", 0.95),
    (r"black.*death.*started", "1347-1351", 0.99),
    # ═══ EARLY MODERN ═══
    (r"printing.*press.*invented|gutenberg.*press", "~1440", 0.99),
    (r"constantinople.*fell", "1453", 0.99),
    (r"columbus.*americas|columbus.*sailed", "1492", 0.99),
    (r"reformation.*started|luther.*95.*theses", "1517", 0.99),
    (r"shakespeare.*born", "1564", 0.95),
    (r"galileo.*telescope", "1609", 0.95),
    (r"newton.*principia|principia.*published", "1687", 0.99),
    # ═══ 18TH CENTURY ═══
    (r"independence.*day.*usa|american.*independence", "1776", 0.99),
    (r"french.*revolution.*started", "1789", 0.99),
    (r"louis.*xvi.*executed", "1793", 0.95),
    (r"napoleon.*crowned", "1804", 0.95),
    # ═══ 19TH CENTURY ═══
    (r"steam.*locomotive.*first", "1804 (Richard Trevithick)", 0.95),
    (r"electricity.*generated.*first", "1832 (Michael Faraday)", 0.95),
    (r"telegraph.*invented", "1837 (Cooke and Wheatstone)", 0.95),
    (r"photography.*invented", "1826 (Niépce)", 0.95),
    (r"darwin.*origin.*species", "1859", 0.99),
    (r"american.*civil.*war.*started", "1861", 0.99),
    (r"american.*civil.*war.*ended", "1865", 0.99),
    (r"eiffel.*tower.*built", "1889", 0.99),
    (r"edison.*light.*bulb", "1879", 0.99),
    (r"telephone.*invented|bell.*telephone", "1876", 0.99),
    (r"radio.*invented|marconi.*radio", "1895", 0.95),
    (r"x.*ray.*discovered|rontgen.*x.*ray", "1895", 0.99),
    # ═══ 20TH CENTURY ═══
    (r"wright.*brothers.*flight|first.*powered.*flight", "1903", 0.99),
    (r"world.*war.*1.*started|ww1.*started", "1914", 0.99),
    (r"world.*war.*1.*ended|ww1.*ended", "1918", 0.99),
    (r"russian.*revolution", "1917", 0.99),
    (r"penicillin.*discovered", "1928", 0.99),
    (r"great.*depression.*started", "1929", 0.99),
    (r"world.*war.*2.*started|ww2.*started", "1939", 0.99),
    (r"world.*war.*2.*ended|ww2.*ended", "1945", 0.99),
    (r"atomic.*bomb.*hiroshima", "1945", 0.99),
    (r"un.*founded", "1945", 0.99),
    (r"dna.*structure.*discovered|watson.*crick", "1953", 0.99),
    (r"sputnik.*launched|first.*satellite", "1957", 0.99),
    (r"cuban.*missile.*crisis", "1962", 0.99),
    (r"jfk.*assassinated|kennedy.*assassinated", "1963", 0.99),
    (r"moon.*landing|first.*moon|apollo.*11", "1969", 0.99),
    (r"first.*email", "1971 (Ray Tomlinson)", 0.95),
    (r"watergate.*scandal", "1972-1974", 0.95),
    (r"berlin.*wall.*fell", "1989", 0.99),
    (r"www.*invented|world.*wide.*web.*invented", "1989 (Tim Berners-Lee)", 0.99),
    (r"soviet.*union.*collapsed", "1991", 0.99),
    (r"deep.*blue.*kasparov", "1997", 0.99),
    (r"human.*genome.*project.*completed", "2003", 0.99),
    # ═══ 21ST CENTURY ═══
    (r"facebook.*launched", "2004", 0.99),
    (r"iphone.*released", "2007", 0.99),
    (r"financial.*crisis.*2008|lehman.*brothers", "2008", 0.99),
    (r"higgs.*boson.*discovered", "2012", 0.99),
    (r"gravitational.*waves.*detected", "2015", 0.99),
    (r"black.*hole.*image.*first", "2019", 0.99),
    (r"covid.*19.*pandemic.*started", "2020", 0.99),
    (r"chatgpt.*launched", "2022 (November 30)", 0.99),
]


class TemporalCore:
    """Core D — Temporal reasoning and date/time analysis.

    Matches "when" questions against a pre-compiled event database.
    Falls back to extracting year patterns (1000-2099) from evidence.
    """

    CORE_ID = "temporal"

    def __init__(self) -> None:
        self._compiled: list[tuple[re.Pattern[str], str, float]] = []
        for pattern, answer, confidence in _RAW_EVENTS:
            try:
                self._compiled.append(
                    (re.compile(pattern, re.IGNORECASE), answer, confidence)
                )
            except re.error:
                pass

    @property
    def core_id(self) -> str:
        return self.CORE_ID

    def process(self, query: str, evidence: list[str]) -> CoreResult:
        t0 = time.perf_counter()
        q = query.lower()

        if q.startswith(("when ", "what year ", "what date ")):
            # Check built-in events
            for compiled, answer, confidence in self._compiled:
                if compiled.search(q):
                    return make_result(
                        self.CORE_ID, answer, confidence,
                        f"Temporal fact: {compiled.pattern}", t0,
                    )

            # Extract dates from evidence
            if evidence:
                for ev in evidence:
                    dates = re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", ev)
                    if dates:
                        return make_result(
                            self.CORE_ID, dates[0], 0.7,
                            "Date extracted from evidence", t0, evidence_used=1,
                        )

        return empty_result(self.CORE_ID, t0, "No temporal pattern matched")
