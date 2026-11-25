"""Plain data containers shared by the config flow, coordinator and entities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time


def symmetric_adjacency(
    declared: Mapping[str, Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    """Make "A is next to B" count both ways, among the rooms that exist.

    `declared` maps each room's area id to the neighbours it lists. Unknown
    areas (not a room) are dropped and a room is never its own neighbour.
    """
    rooms = set(declared)
    links: dict[str, set[str]] = {room: set() for room in rooms}
    for room, neighbours in declared.items():
        for other in neighbours:
            if other in rooms and other != room:
                links[room].add(other)
                links[other].add(room)
    return {room: tuple(sorted(found)) for room, found in links.items()}


@dataclass(frozen=True, slots=True)
class RoomConfig:
    """Static configuration of one room (a config subentry)."""

    subentry_id: str
    area_id: str
    name: str
    temperature_sensors: tuple[str, ...]
    device_sensors: tuple[str, ...] = ()
    valves: tuple[str, ...] = ()
    adjacent_area_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class RoomReading:
    """What the sensors say right now. Temperatures are in °C, None if unknown."""

    temperature: float | None  # mean of the available room sensors
    sensor_temperatures: dict[str, float | None] = field(default_factory=dict)
    device_temperatures: dict[str, float | None] = field(default_factory=dict)
    valve_open: dict[str, float | None] = field(default_factory=dict)  # 1 open, 0 closed
    adjacent_temperatures: dict[str, float | None] = field(default_factory=dict)


@dataclass(slots=True)
class RoomState:
    """Result of one control cycle for one room."""

    config: RoomConfig
    reading: RoomReading
    # Filled by later pipeline stages; None while they are stubs.
    target: object | None = None
    plan: object | None = None


@dataclass(frozen=True, slots=True)
class CentralHeatingWindow:
    """When the building's central heating runs: a season, and hours within each day."""

    season_start: date
    season_end: date
    hours_start: time
    hours_end: time

    def is_open(self, now: datetime) -> bool:
        if not self.season_start <= now.date() <= self.season_end:
            return False
        t = now.time()
        if self.hours_start <= self.hours_end:
            return self.hours_start <= t < self.hours_end
        return t >= self.hours_start or t < self.hours_end  # window wraps midnight

    def season_over(self, now: datetime) -> bool:
        return now.date() > self.season_end


@dataclass(slots=True)
class HubReading:
    """Readings that are not tied to one room."""

    outdoor_temperature: float | None = None
    central_window: CentralHeatingWindow | None = None
    central_available: bool = False
    central_hvac_action: str | None = None  # living-room thermostat: heating / idle
    ac_hvac_mode: str | None = None
    ac_hvac_action: str | None = None
    ac_power_w: float | None = None  # measured; the central heater has no equivalent
    outdoor_source: str | None = None  # "sensor", "weather" or None
    cloud_coverage: float | None = None  # %, from the weather entity
    wind_speed: float | None = None  # km/h, from the weather entity
