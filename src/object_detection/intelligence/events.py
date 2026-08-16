"""In-memory, filterable event timeline for one processing session."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Event:
    timestamp: float
    event_type: str
    message: str
    track_id: Optional[int] = None
    severity: str = "info"
    details: Optional[Dict[str, Any]] = None

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["time"] = datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")
        row.pop("timestamp")
        return row


class EventLogger:
    def __init__(self, max_events: int = 1000) -> None:
        self.max_events = max_events
        self.events: List[Event] = []

    def add(self, event: Event) -> None:
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)

    def filtered(self, track_id: Optional[int] = None, event_type: Optional[str] = None) -> List[Event]:
        return [
            event for event in self.events
            if (track_id is None or event.track_id == track_id)
            and (event_type is None or event.event_type == event_type)
        ]
