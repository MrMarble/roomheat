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
from .learn import describe, skill
from .coordinator import RoomHeatConfigEntry, RoomHeatCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: RoomHeatConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    coordinator = entry.runtime_data
    for subentry_id in coordinator.data:
        async_add_entities(
            [
                RoomTemperatureSensor(coordinator, subentry_id),
                RoomThermalModelSensor(coordinator, subentry_id),
                RoomPredictedTemperatureSensor(coordinator, subentry_id),
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
    _attr_options = ["pending", "collecting", "ready", "poor"]
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
                "ready": "predicts clearly better than 'the temperature stays put'",
                "poor": "does not yet beat 'the temperature stays put'",
            }[model.status],
            "data_hours": round(model.usable_hours, 1),
            "trained_at": _iso(model.trained_at),
        }
        if model.fit is not None:
            attrs |= describe(model.fit.result)
            holdout = model.fit.holdout
            if holdout is not None:
                attrs["forecast_error_1h_c"] = round(holdout.rmse_model, 3)
                attrs["stays_put_error_1h_c"] = round(holdout.rmse_persistence, 3)
                s = skill(model.fit)
                attrs["skill"] = None if s is None else round(s, 3)
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
