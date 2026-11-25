"""State -> recorded value. Shared by the live probe and the recorder backfill."""

from __future__ import annotations

import math

from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    STATE_OFF,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfPower,
    UnitOfSpeed,
    UnitOfTemperature,
)
from homeassistant.core import State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util.unit_conversion import (
    PowerConverter,
    SpeedConverter,
    TemperatureConverter,
)

from .signals import (
    ACTIVE,
    CLOUD_COVERAGE,
    COOLING,
    HEATING,
    SETPOINT,
    TEMPERATURE,
    WEATHER_TEMPERATURE,
    WIND_SPEED,
)


def _number(raw: object) -> float | None:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _convert(value: float, unit: str | None, target: str, converter) -> float | None:
    """`value` in `unit` converted to `target`; None if the unit is unknown."""
    if not unit or unit == target:
        return value
    try:
        return converter.convert(value, unit, target)
    except (HomeAssistantError, ValueError):
        return None


def signal_value(
    kind: str, state: State | None, system_temperature_unit: str = UnitOfTemperature.CELSIUS
) -> float | None:
    """Value of a signal in canonical units (°C, W, km/h, %, or a 1.0/0.0 flag).

    `system_temperature_unit` is the unit Home Assistant uses for climate and weather
    temperatures that carry no unit of their own.
    None when the state is missing, unavailable, or the value is absent or unusable.
    """
    if state is None or state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
        return None
    attrs = state.attributes

    if kind in (HEATING, COOLING):
        action = attrs.get("hvac_action")
        if action is None:
            # Many climate entities (e.g. air conditioners) have no hvac_action:
            # fall back to the mode. auto / dry / fan_only cannot be told apart.
            action = {"heat": HEATING, "cool": COOLING, "off": "off"}.get(state.state)
            if action is None:
                return None
        return float(action == kind)

    if kind == ACTIVE:
        return float(state.state != STATE_OFF)

    if kind == SETPOINT:
        value = _number(attrs.get("temperature"))
        if value is None:
            return None
        return _convert(value, system_temperature_unit, UnitOfTemperature.CELSIUS, TemperatureConverter)

    if kind == WEATHER_TEMPERATURE:
        value = _number(attrs.get("temperature"))
        if value is None:
            return None
        unit = attrs.get("temperature_unit") or system_temperature_unit
        return _convert(value, unit, UnitOfTemperature.CELSIUS, TemperatureConverter)

    if kind == CLOUD_COVERAGE:
        return _number(attrs.get("cloud_coverage"))

    if kind == WIND_SPEED:
        value = _number(attrs.get("wind_speed"))
        if value is None:
            return None
        return _convert(
            value, attrs.get("wind_speed_unit"), UnitOfSpeed.KILOMETERS_PER_HOUR, SpeedConverter
        )

    # plain numeric sensors
    value = _number(state.state)
    if value is None:
        return None
    if kind == TEMPERATURE:
        return _convert(value, attrs.get(ATTR_UNIT_OF_MEASUREMENT), UnitOfTemperature.CELSIUS, TemperatureConverter)
    return _convert(value, attrs.get(ATTR_UNIT_OF_MEASUREMENT), UnitOfPower.WATT, PowerConverter)
