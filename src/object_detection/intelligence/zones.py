"""Configurable rectangular intelligence zones and entry/exit transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Zone:
    name: str
    kind: str
    x1: float
    y1: float
    x2: float
    y2: float

    def contains(self, point: Tuple[float, float]) -> bool:
        x, y = point
        return min(self.x1, self.x2) <= x <= max(self.x1, self.x2) and min(self.y1, self.y2) <= y <= max(self.y1, self.y2)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class ZoneTransition:
    track_id: int
    zone: Zone
    action: str


class ZoneManager:
    """Tracks membership transitions for a collection of normalized-frame zones."""

    def __init__(self, zones: Iterable[Zone] = ()) -> None:
        self.zones: List[Zone] = list(zones)
        self._membership: Dict[int, set[str]] = {}

    def update(self, track_id: int, point: Tuple[float, float]) -> List[ZoneTransition]:
        current = {zone.name for zone in self.zones if zone.contains(point)}
        previous = self._membership.get(track_id, set())
        transitions: List[ZoneTransition] = []
        lookup = {zone.name: zone for zone in self.zones}
        for name in current - previous:
            transitions.append(ZoneTransition(track_id, lookup[name], "entered"))
        for name in previous - current:
            transitions.append(ZoneTransition(track_id, lookup[name], "left"))
        self._membership[track_id] = current
        return transitions

    def active_zones(self, track_id: int) -> set[str]:
        return set(self._membership.get(track_id, set()))
