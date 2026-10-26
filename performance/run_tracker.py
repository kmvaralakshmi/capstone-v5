"""Run-level provenance and timing utilities for V5."""
from __future__ import annotations
import json, time, uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "processed-data" / "run_metrics"
RUN_DIR.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunTracker:
    def __init__(self, company: str | None = None, mode: str = "single"):
        self.run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        self.company = company
        self.mode = mode
        self.started_at = utc_now()
        self.stages: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.status = "running"

    @contextmanager
    def stage(self, name: str, source: str | None = None, **meta: Any):
        started = time.perf_counter()
        record = {
            "run_id": self.run_id, "company": self.company, "stage": name,
            "source": source, "started_at": utc_now(), "status": "running", **meta
        }
        try:
            yield record
            record["status"] = "success"
        except Exception as exc:
            record["status"] = "error"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            self.errors.append(record.copy())
            raise
        finally:
            record["ended_at"] = utc_now()
            record["duration_seconds"] = round(time.perf_counter() - started, 4)
            self.stages.append(record)

    def add_error(self, stage: str, error: Exception | str, **meta: Any):
        self.errors.append({
            "run_id": self.run_id, "company": self.company, "stage": stage,
            "error": str(error), "error_type": type(error).__name__ if isinstance(error, Exception) else "Error",
            **meta,
        })

    def finish(self, status: str = "completed", **metadata: Any) -> Path:
        self.status = status
        ended = utc_now()
        payload = {
            "run_id": self.run_id, "company": self.company, "mode": self.mode,
            "started_at": self.started_at, "ended_at": ended, "status": status,
            "stages": self.stages, "errors": self.errors, **metadata,
        }
        path = RUN_DIR / f"{self.run_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path
