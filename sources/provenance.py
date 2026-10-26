"""Canonical metric/evidence records used across V5."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any

@dataclass
class EvidenceRecord:
    run_id: str
    company_code: str
    company_name: str
    metric_name: str
    value: Any
    unit: str | None = None
    financial_year: str | None = None
    source: str | None = None
    source_url: str | None = None
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    evidence_source: str | None = None
    evidence_locator: str | None = None
    confidence: str = "unknown"
    validation_status: str = "not_validated"
    fallback_used: bool = False
    discrepancy_flag: bool = False
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
