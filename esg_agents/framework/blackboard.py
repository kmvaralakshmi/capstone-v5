"""Blackboard: the message channel agents use to talk to each other.

Agents do not import one another. An agent posts findings (coverage gaps,
suspicious companies, data-quality warnings) to the blackboard and downstream
agents read what upstream agents said about the data they are about to use.
The board is persisted to disk so every run leaves an auditable conversation.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class Blackboard:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self.messages: List[Dict[str, Any]] = []
        if self.path and self.path.exists():
            try:
                self.messages = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.messages = []

    def post(self, sender: str, topic: str, payload: Any, to: str = "all",
             severity: str = "info") -> Dict[str, Any]:
        with self._lock:
            return self._post(sender, topic, payload, to, severity)

    def _post(self, sender, topic, payload, to, severity):
        msg = {
            "id": len(self.messages) + 1,
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "from": sender, "to": to, "topic": topic,
            "severity": severity, "payload": payload,
        }
        self.messages.append(msg)
        self._save()
        return msg

    def read(self, topic: Optional[str] = None, sender: Optional[str] = None,
             to: Optional[str] = None) -> List[Dict[str, Any]]:
        out = self.messages
        if topic:
            out = [m for m in out if m["topic"] == topic]
        if sender:
            out = [m for m in out if m["from"] == sender]
        if to:
            out = [m for m in out if m["to"] in (to, "all")]
        return out

    def latest(self, topic: str, sender: Optional[str] = None) -> Optional[Any]:
        msgs = self.read(topic=topic, sender=sender)
        return msgs[-1]["payload"] if msgs else None

    def clear(self) -> None:
        self.messages = []
        self._save()

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.messages, indent=2, default=str),
                             encoding="utf-8")
