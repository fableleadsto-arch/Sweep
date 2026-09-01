"""
Training Safety — §23

Do not scrape or train indiscriminately on private, leaked, copyrighted,
or restricted datasets.

Respect:
- licensing
- robots.txt where applicable
- privacy
- access controls
- data retention
- deletion requests
- platform policies

For investigation functionality, build authorization and audit logging
into the architecture.

Do not design the system to bypass authentication, paywalls,
private accounts, or security controls.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AuditEntry:
    """A single audit log entry."""
    entry_id: str
    timestamp: float
    action: str  # data_access, training, retrieval, deletion, etc.
    actor: str  # who/what initiated the action
    target: str  # what was accessed
    result: str  # allowed, denied, error
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataLicense:
    """License information for a dataset."""
    dataset_name: str
    license_type: str  # open, restricted, commercial, private, unknown
    license_url: str = ""
    allows_training: bool = False
    allows_commercial: bool = False
    allows_redistribution: bool = False
    attribution_required: bool = False
    requires_share_alike: bool = False
    restrictions: list[str] = field(default_factory=list)


@dataclass
class ProvenanceRecord:
    """Provenance tracking for data items."""
    item_id: str
    source: str
    license: str
    access_date: float
    hash: str
    allowed_uses: list[str]
    restrictions: list[str]


class SafetyManager:
    """
    §23: Manages training safety, audit logging, and data governance.
    """

    def __init__(
        self,
        audit_dir: str | Path = "sweep_neural_mesh/training/audit",
    ) -> None:
        self._audit_dir = Path(audit_dir)
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._audit_log: list[AuditEntry] = []
        self._licenses: dict[str, DataLicense] = {}
        self._provenance: dict[str, ProvenanceRecord] = {}
        self._denied_actions: list[AuditEntry] = []
        self._counter = 0

        # Register default safe licenses
        self._register_default_licenses()

    def _register_default_licenses(self) -> None:
        """Register known safe dataset licenses."""
        self.register_license(DataLicense(
            dataset_name="open_facts",
            license_type="open",
            allows_training=True,
            allows_commercial=True,
            allows_redistribution=True,
        ))
        self.register_license(DataLicense(
            dataset_name="synthetic_generated",
            license_type="open",
            allows_training=True,
            allows_commercial=True,
            allows_redistribution=True,
        ))

    def register_license(self, license: DataLicense) -> None:
        """Register a dataset license."""
        self._licenses[license.dataset_name] = license

    def check_license(self, dataset_name: str) -> DataLicense | None:
        """Check if a dataset is licensed for training."""
        return self._licenses.get(dataset_name)

    def is_dataset_safe(self, dataset_name: str, use_case: str = "training") -> bool:
        """Check if a dataset can be used for a specific purpose."""
        license = self._licenses.get(dataset_name)
        if license is None:
            self._log_audit(
                action="license_check",
                actor="safety_manager",
                target=dataset_name,
                result="denied",
                reason=f"No license registered for {dataset_name}",
            )
            return False

        if use_case == "training" and not license.allows_training:
            self._log_audit(
                action="license_check",
                actor="safety_manager",
                target=dataset_name,
                result="denied",
                reason=f"License '{license.license_type}' does not allow training",
            )
            return False

        if use_case == "commercial" and not license.allows_commercial:
            self._log_audit(
                action="license_check",
                actor="safety_manager",
                target=dataset_name,
                result="denied",
                reason=f"License '{license.license_type}' does not allow commercial use",
            )
            return False

        return True

    # ══════════════════════════════════════════════════════════════════
    # AUDIT LOGGING
    # ══════════════════════════════════════════════════════════════════

    def _log_audit(
        self,
        action: str,
        actor: str,
        target: str,
        result: str,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log an audit entry."""
        self._counter += 1
        entry = AuditEntry(
            entry_id=f"AUD-{self._counter:06d}",
            timestamp=time.time(),
            action=action,
            actor=actor,
            target=target,
            result=result,
            reason=reason,
            metadata=metadata or {},
        )
        self._audit_log.append(entry)
        if result == "denied":
            self._denied_actions.append(entry)

        # Persist to disk
        self._save_audit_entry(entry)
        return entry

    def log_data_access(
        self, dataset: str, accessor: str, purpose: str
    ) -> AuditEntry:
        """Log a data access event."""
        return self._log_audit(
            action="data_access",
            actor=accessor,
            target=dataset,
            result="allowed",
            reason=f"Access for {purpose}",
        )

    def log_training(
        self, dataset: str, model: str, config: str
    ) -> AuditEntry:
        """Log a training event."""
        safe = self.is_dataset_safe(dataset, "training")
        return self._log_audit(
            action="training",
            actor=model,
            target=dataset,
            result="allowed" if safe else "denied",
            reason=f"Training with config: {config}" if safe else "License does not permit training",
        )

    def log_retrieval(self, source: str, query: str) -> AuditEntry:
        """Log a retrieval event."""
        return self._log_audit(
            action="retrieval",
            actor="web_scraper",
            target=source,
            result="allowed",
            reason=f"Query: {query[:100]}",
        )

    def log_deletion(self, item_id: str, reason: str) -> AuditEntry:
        """Log a data deletion event (GDPR compliance)."""
        return self._log_audit(
            action="deletion",
            actor="user_request",
            target=item_id,
            result="allowed",
            reason=reason,
        )

    def _save_audit_entry(self, entry: AuditEntry) -> None:
        """Save an audit entry to disk."""
        path = self._audit_dir / f"{entry.entry_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp,
                "action": entry.action,
                "actor": entry.actor,
                "target": entry.target,
                "result": entry.result,
                "reason": entry.reason,
                "metadata": entry.metadata,
            }, f, indent=2)

    # ══════════════════════════════════════════════════════════════════
    # PROVENANCE
    # ══════════════════════════════════════════════════════════════════

    def track_provenance(
        self,
        item_id: str,
        source: str,
        license: str,
        allowed_uses: list[str],
        restrictions: list[str] | None = None,
    ) -> ProvenanceRecord:
        """Track provenance for a data item."""
        record = ProvenanceRecord(
            item_id=item_id,
            source=source,
            license=license,
            access_date=time.time(),
            hash=hashlib.sha256(item_id.encode()).hexdigest()[:16],
            allowed_uses=allowed_uses,
            restrictions=restrictions or [],
        )
        self._provenance[item_id] = record
        return record

    def check_provenance(self, item_id: str) -> ProvenanceRecord | None:
        """Check provenance for a data item."""
        return self._provenance.get(item_id)

    # ══════════════════════════════════════════════════════════════════
    # PRIVACY
    # ══════════════════════════════════════════════════════════════════

    def check_privacy(self, text: str) -> dict[str, Any]:
        """Check if text contains PII (Personally Identifiable Information)."""
        import re
        pii_found = []

        # Email patterns
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            pii_found.append({"type": "email", "count": len(emails)})

        # Phone patterns
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        if phones:
            pii_found.append({"type": "phone", "count": len(phones)})

        # SSN patterns
        ssns = re.findall(r'\b\d{3}-\d{2}-\d{4}\b', text)
        if ssns:
            pii_found.append({"type": "ssn", "count": len(ssns)})

        # Credit card patterns
        cc = re.findall(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', text)
        if cc:
            pii_found.append({"type": "credit_card", "count": len(cc)})

        return {
            "contains_pii": len(pii_found) > 0,
            "pii_types": pii_found,
            "recommendation": "do_not_train" if pii_found else "safe",
        }

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════

    def summary(self) -> dict[str, Any]:
        """Generate a safety summary."""
        total_actions = len(self._audit_log)
        denied = len(self._denied_actions)
        by_action: dict[str, int] = {}
        for entry in self._audit_log:
            by_action[entry.action] = by_action.get(entry.action, 0) + 1

        return {
            "total_audit_entries": total_actions,
            "denied_actions": denied,
            "compliance_rate": (total_actions - denied) / max(total_actions, 1),
            "registered_licenses": len(self._licenses),
            "provenance_tracked": len(self._provenance),
            "actions_by_type": by_action,
            "recent_denials": [
                {"action": d.action, "target": d.target, "reason": d.reason}
                for d in self._denied_actions[-5:]
            ],
        }
