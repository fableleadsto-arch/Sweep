"""Contradiction detection across extracted claims.

When a new claim disagrees with a stored claim about the same
entity + property + subject, the more *authoritative* claim wins:

* ``authority`` = source authority × claim confidence × recency.
* The losing claim is marked ``superseded`` when the winner is clearly
  stronger, or ``contradicted`` when the two are comparable and both stay
  visible for human inspection.
* Every resolution produces a :class:`KnowledgeUpdate` so the system has a
  verifiable change log (spec: knowledge versioning).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

from .models import ClaimStatus, KnowledgeClaim, KnowledgeUpdate, utcnow
from .scoring import score_freshness


@dataclass
class ContradictionResolution:
    claim: KnowledgeClaim
    status: ClaimStatus
    superseded_ids: list[str]
    update: Optional[KnowledgeUpdate]
    reason: str = ""


def _claim_key(claim: KnowledgeClaim) -> tuple[str, str, str]:
    return (
        claim.entity.strip().lower(),
        claim.property.strip().lower(),
        claim.subject.strip().lower(),
    )


def _norm_value(value: str) -> str:
    return " ".join(value.lower().split()).strip(" .;,")


def _authority(claim: KnowledgeClaim, now: datetime) -> float:
    freshness = score_freshness(claim.collected_at, now=now, half_life_days=90.0)
    return claim.authority * (0.6 + 0.4 * claim.confidence) * freshness


def values_conflict(a: str, b: str) -> bool:
    """True when two values mean the same thing yet disagree.

    Version-like values compare as numeric tuples; ``1.2`` vs ``1.2.0`` are
    prefix-equal so they are **not** contradictions. Anything else conflicts
    when the normalized forms differ.
    """
    na, nb = _norm_value(a), _norm_value(b)
    if na == nb:
        return False
    import re

    version_a = re.match(r"^v?\d+(?:\.\d+)*$", na)
    version_b = re.match(r"^v?\d+(?:\.\d+)*$", nb)
    if version_a and version_b:
        digits = lambda s: tuple(int(x) for x in re.findall(r"\d+", s))  # noqa: E731
        da, db = digits(na), digits(nb)
        if da == db:
            return False
        # Prefix-equal versions ("1.2" vs "1.2.0") are the same fact.
        if len(da) < len(db) and db[: len(da)] == da:
            return False
        if len(db) < len(da) and da[: len(db)] == db:
            return False
        return True
    return True


def resolve_contradictions(
    new_claim: KnowledgeClaim,
    existing: Iterable[KnowledgeClaim],
    *,
    now: Optional[datetime] = None,
) -> ContradictionResolution:
    """Resolve a new claim against stored claims; return status + updates.

    Resolution rules:

    * No conflict → the claim stays ``active``, no update is recorded.
    * One conflicting claim found and the new one is clearly more authoritative
      (ratio ≥ 1.5) → old is ``superseded``, new stays ``active``.
    * Comparable authority → both are ``contradicted`` (visible, flagged).
    * New claim loses → new is ``contradicted``, old stays active; the losing
      claim is still stored (nothing is ever deleted without a trace).
    """
    now = now or utcnow()
    conflict: Optional[KnowledgeClaim] = None
    for stored in existing:
        if stored.status == ClaimStatus.SUPERSEDED:
            continue
        if _claim_key(stored) != _claim_key(new_claim):
            continue
        if values_conflict(stored.value, new_claim.value):
            conflict = stored
            break

    if conflict is None:
        return ContradictionResolution(
            claim=new_claim,
            status=ClaimStatus.ACTIVE,
            superseded_ids=[],
            update=None,
            reason="no conflicting claim",
        )

    new_power = _authority(new_claim, now)
    old_power = _authority(conflict, now)
    ratio = new_power / old_power if old_power > 0 else float("inf")

    update = KnowledgeUpdate(
        id=uuid.uuid4().hex,
        entity=conflict.entity,
        property=conflict.property,
        old_value=conflict.value,
        new_value=new_claim.value,
        source_id=new_claim.source_id,
        source_url=new_claim.source_url,
        detected_at=now,
        reason="claim conflict",
        confidence=new_claim.confidence,
    )

    if ratio >= 1.5:
        return ContradictionResolution(
            claim=new_claim,
            status=ClaimStatus.ACTIVE,
            superseded_ids=[conflict.id],
            update=update,
            reason=f"new claim more authoritative (ratio {ratio:.2f})",
        )
    if ratio <= 1 / 1.5:
        return ContradictionResolution(
            claim=new_claim,
            status=ClaimStatus.CONTRADICTED,
            superseded_ids=[],
            update=update,
            reason=f"stored claim more authoritative (ratio {ratio:.2f})",
        )
    # Comparable — flag both.
    return ContradictionResolution(
        claim=new_claim,
        status=ClaimStatus.CONTRADICTED,
        superseded_ids=[conflict.id],
        update=update,
        reason=f"contradicting claims with comparable authority (ratio {ratio:.2f})",
    )
