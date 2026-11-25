"""Which signals are recorded, and how they are named.

A series is `<entity_id>` for a plain sensor, or `<entity_id>:<kind>` for a value
derived from an entity (an attribute, or an action flag).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

TEMPERATURE = "temperature"  # a temperature sensor, °C
POWER = "power"  # a power sensor, W
HEATING = "heating"  # climate hvac_action (or mode) == heating -> 1.0 else 0.0
COOLING = "cooling"  # same for cooling
SETPOINT = "setpoint"  # climate target temperature, °C: what was asked for
ACTIVE = "active"  # climate is switched on (mode is not "off") -> 1.0 else 0.0
WEATHER_TEMPERATURE = "weather_temperature"  # weather entity, °C
CLOUD_COVERAGE = "cloud_coverage"  # weather entity, 0..100 %
WIND_SPEED = "wind_speed"  # weather entity, km/h

CLIMATE_KINDS = (SETPOINT, ACTIVE)  # recorded for every climate entity we know of
WEATHER_KINDS = (WEATHER_TEMPERATURE, CLOUD_COVERAGE, WIND_SPEED)


@dataclass(frozen=True, slots=True)
class Signal:
    series: str
    entity_id: str
    kind: str


def action_series(entity_id: str, kind: str) -> str:
    return f"{entity_id}:{kind}"


def _derived(entity_id: str, kinds: Iterable[str]) -> list[Signal]:
    return [Signal(action_series(entity_id, k), entity_id, k) for k in kinds]


def build_signals(
    *,
    room_sensors: Iterable[str],
    outdoor: str | None,
    thermostat: str | None,
    ac: str | None,
    ac_power: str | None,
    valves: Iterable[str] = (),
    weather: str | None = None,
) -> list[Signal]:
    """All signals worth recording, de-duplicated, in a stable order."""
    signals: list[Signal] = [Signal(e, e, TEMPERATURE) for e in room_sensors]
    if outdoor:
        signals.append(Signal(outdoor, outdoor, TEMPERATURE))
    if ac_power:
        signals.append(Signal(ac_power, ac_power, POWER))
    if weather:
        signals += _derived(weather, WEATHER_KINDS)
    if thermostat:
        signals += _derived(thermostat, (HEATING, *CLIMATE_KINDS))
    if ac:
        signals += _derived(ac, (HEATING, COOLING, *CLIMATE_KINDS))
    for valve in valves:
        signals += _derived(valve, (HEATING, *CLIMATE_KINDS))
    return list({s.series: s for s in signals}.values())
