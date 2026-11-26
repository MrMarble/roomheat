"""Per-room temperature sensor (what RoomHeat currently sees)."""

from __future__ import annotations

from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import EntityCategory, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from datetime import datetime, timezone

from .const import DOMAIN
from .learn import describe, skill, skill_active
from .coordinator import RoomHeatConfigEntry, RoomHeatCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: RoomHeatConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    coordinator = entry.runtime_data
    async_add_entities([RecommendedHeatSource(coordinator, entry.entry_id)])
    for subentry_id in coordinator.data:
        async_add_entities(
            [
                RoomTemperatureSensor(coordinator, subentry_id),
                RoomThermalModelSensor(coordinator, subentry_id),
                RoomPredictedTemperatureSensor(coordinator, subentry_id),
                RoomTargetTemperatureSensor(coordinator, subentry_id),
                RoomHeatFromSensor(coordinator, subentry_id),
            ],
            config_subentry_id=subentry_id,
        )


class RoomTemperatureSensor(CoordinatorEntity[RoomHeatCoordinator], SensorEntity):
    _attr_has_entity_name = True
    _attr_name = "Temperature"
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS

    def __init__(self, coordinator: RoomHeatCoordinator, subentry_id: str) -> None:
        super().__init__(coordinator)
        self._subentry_id = subentry_id
        room = coordinator.data[subentry_id].config
        self._attr_unique_id = f"{subentry_id}_temperature"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, subentry_id)},
            name=room.name,
            suggested_area=room.name,
        )

    @property
    def available(self) -> bool:
        return super().available and self._subentry_id in self.coordinator.data

    @property
    def native_value(self) -> float | None:
        return self.coordinator.data[self._subentry_id].reading.temperature

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        state = self.coordinator.data[self._subentry_id]
        return {
            "room_sensors": state.reading.sensor_temperatures,
            "device_sensors": state.reading.device_temperatures,
            "valves_open": state.reading.valve_open,
            "adjacent_temperatures": state.reading.adjacent_temperatures,
            "outdoor_temperature": self.coordinator.hub.outdoor_temperature,
        }


def _iso(ts: float | int | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class _RoomModelEntity(CoordinatorEntity[RoomHeatCoordinator], SensorEntity):
    """Base for the per-room entities that report the thermal model."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: RoomHeatCoordinator, subentry_id: str, key: str) -> None:
        super().__init__(coordinator)
        self._subentry_id = subentry_id
        self._attr_unique_id = f"{subentry_id}_{key}"
        room = coordinator.data[subentry_id].config
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, subentry_id)}, name=room.name)

    @property
    def _model(self):
        return self.coordinator.models.states.get(self._subentry_id)

    @property
    def available(self) -> bool:
        return super().available and self._subentry_id in self.coordinator.data


class RoomThermalModelSensor(_RoomModelEntity):
    """What the model has learned about this room, and whether it can be trusted yet."""

    _attr_name = "Thermal model"
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = ["pending", "collecting", "unfit", "ready", "poor"]
    _attr_icon = "mdi:home-thermometer-outline"

    def __init__(self, coordinator: RoomHeatCoordinator, subentry_id: str) -> None:
        super().__init__(coordinator, subentry_id, "thermal_model")

    @property
    def native_value(self) -> str:
        return self._model.status if self._model else "pending"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        model = self._model
        if model is None:
            return {}
        attrs: dict[str, Any] = {
            "meaning": {
                "pending": "waiting for the first fit",
                "collecting": "not enough data yet to judge (about 3 days of readings needed)",
                "unfit": "enough readings, but no complete stretch of data to learn from (gaps in the inputs)",
                "ready": "predicts clearly better than 'the temperature stays put'",
                "poor": "does not yet beat 'the temperature stays put'",
            }[model.status],
            "data_hours": round(model.usable_hours, 1),
            "trained_at": _iso(model.trained_at),
        }
        if model.fit is not None:
            attrs |= describe(model.fit.result)
            attrs["features_dropped"] = list(model.fit.dropped)
            attrs["exposure_hours"] = model.fit.exposure
            holdout = model.fit.holdout
            if holdout is not None:
                attrs["forecast_error_1h_c"] = round(holdout.rmse_model, 3)
                attrs["stays_put_error_1h_c"] = round(holdout.rmse_persistence, 3)
                s = skill(model.fit)
                attrs["skill"] = None if s is None else round(s, 3)
                sa = skill_active(model.fit)
                attrs["skill_where_heating_changes"] = None if sa is None else round(sa, 3)
        return attrs


class RoomPredictedTemperatureSensor(_RoomModelEntity):
    """Where the room is heading in an hour if nothing changes."""

    _attr_name = "Predicted temperature (1 h)"
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_suggested_display_precision = 1

    def __init__(self, coordinator: RoomHeatCoordinator, subentry_id: str) -> None:
        super().__init__(coordinator, subentry_id, "predicted_temperature_1h")

    @property
    def available(self) -> bool:
        return super().available and self._model is not None and self._model.predicted is not None

    @property
    def native_value(self) -> float | None:
        return None if self._model is None else self._model.predicted

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        model = self._model
        if model is None:
            return {}
        return {
            "assumption": "heating, AC, valves, outdoor and neighbouring temperatures stay as they are now",
            "predicted_from": _iso(model.predicted_from),
            "predicted_for": _iso(model.predicted_for),
            "last_forecast_error_c": None if model.last_error is None else round(model.last_error, 2),
            "model_status": model.status,
        }


class _RoomPlanEntity(_RoomModelEntity):
    """Base for the entities that report the room's target and advice."""

    @property
    def _state(self):
        return self.coordinator.data[self._subentry_id]

    @property
    def _advice(self):
        return self._state.plan


class RoomTargetTemperatureSensor(_RoomPlanEntity):
    """The temperature the room should have right now, from its schedule."""

    _attr_name = "Target temperature"
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_suggested_display_precision = 1

    def __init__(self, coordinator: RoomHeatCoordinator, subentry_id: str) -> None:
        super().__init__(coordinator, subentry_id, "target_temperature")

    @property
    def native_value(self) -> float | None:
        target = self._state.target
        return None if target is None else target.temperature

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        target, room = self._state.target, self._state.config
        if target is None:
            return {}
        return {
            "level": target.level,
            "changes_at": None if target.changes_at is None else target.changes_at.isoformat(),
            "schedule": room.schedule,
            "comfort_c": room.comfort,
            "eco_c": room.eco,
        }


class RoomHeatFromSensor(_RoomPlanEntity):
    """When the room has to start heating to be at comfort temperature on time."""

    _attr_name = "Heat needed from"
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_icon = "mdi:radiator"

    def __init__(self, coordinator: RoomHeatCoordinator, subentry_id: str) -> None:
        super().__init__(coordinator, subentry_id, "heat_needed_from")

    @property
    def native_value(self) -> datetime | None:
        return None if self._advice is None else self._advice.heat_from

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        a = self._advice
        if a is None:
            return {}
        return {
            "heat_needed_now": a.needs_heat_now,
            "comfort_at": None if a.comfort_at is None else a.comfort_at.isoformat(),
            "deficit_c": a.deficit_c,
            "lead_minutes": a.lead_minutes,
            "heating_rate_c_per_h": a.rate_c_per_h,
            "heating_rate_is_assumed": a.rate_assumed,
            "note": a.reason,
        }


class RecommendedHeatSource(CoordinatorEntity[RoomHeatCoordinator], SensorEntity):
    """The single heat source advised for the whole home. Advice only."""

    _attr_has_entity_name = True
    _attr_name = "Recommended heat source"
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = ["central", "ac", "none"]
    _attr_icon = "mdi:fire-circle"

    def __init__(self, coordinator: RoomHeatCoordinator, entry_id: str) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry_id}_recommended_heat_source"
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, entry_id)}, name="RoomHeat")

    @property
    def native_value(self) -> str | None:
        rec = self.coordinator.recommendation
        return None if rec is None else rec.source

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        rec = self.coordinator.recommendation
        if rec is None:
            return {}
        return {
            "needed_from": None if rec.at is None else rec.at.isoformat(),
            "rooms": list(rec.rooms),
            "rooms_not_reached": list(rec.unserved),
            "cost_eur_per_kwh_of_heat": rec.cost_per_kwh,
            "reason": rec.reason,
        }
