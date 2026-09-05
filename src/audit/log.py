"""Minimal structured audit trail for drift and adaptation decisions."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class AuditLog:
    def __init__(self): self.events: list[AuditEvent] = []
    def record(self, event_type: str, **payload: Any) -> AuditEvent:
        event = AuditEvent(event_type, payload); self.events.append(event); return event
    def as_dicts(self) -> list[dict[str, Any]]:
        return [{"event_type": e.event_type, "payload": e.payload, "timestamp": e.timestamp} for e in self.events]
