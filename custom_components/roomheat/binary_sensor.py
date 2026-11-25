"""Hub-level binary sensor: is central heating available right now?"""

from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .coordinator import RoomHeatConfigEntry, RoomHeatCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: RoomHeatConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    async_add_entities([CentralHeatingAvailable(entry.runtime_data, entry.entry_id)])


class CentralHeatingAvailable(CoordinatorEntity[RoomHeatCoordinator], BinarySensorEntity):
    _attr_has_entity_name = True
    _attr_name = "Central heating available"

    def __init__(self, coordinator: RoomHeatCoordinator, entry_id: str) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry_id}_central_available"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry_id)}, name="RoomHeat"
        )

    @property
    def is_on(self) -> bool:
        return self.coordinator.hub.central_available

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        hub = self.coordinator.hub
        window = hub.central_window
        attrs: dict[str, Any] = {
            "thermostat_hvac_action": hub.central_hvac_action,
            "ac_hvac_mode": hub.ac_hvac_mode,
            "ac_hvac_action": hub.ac_hvac_action,
            "ac_power_w": hub.ac_power_w,
            "outdoor_temperature": hub.outdoor_temperature,
            "outdoor_source": hub.outdoor_source,
            "cloud_coverage": hub.cloud_coverage,
            "wind_speed": hub.wind_speed,
        }
        if window:
            attrs |= {
                "season_start": window.season_start.isoformat(),
                "season_end": window.season_end.isoformat(),
                "hours": f"{window.hours_start:%H:%M}-{window.hours_end:%H:%M}",
                "season_over": window.season_over(dt_util.now()),
            }
        return attrs
