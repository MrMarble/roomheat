"""Hub-level button: refit every room's thermal model now."""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import DOMAIN
from .coordinator import RoomHeatConfigEntry, RoomHeatCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: RoomHeatConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    async_add_entities([RefitModelsButton(entry.runtime_data, entry.entry_id)])


class RefitModelsButton(ButtonEntity):
    """Re-learns every room from the history stored so far.

    It does not fetch new history (that happens once at startup); it re-runs the fit
    on what is already stored, which is what changes as the days go by.
    """

    _attr_has_entity_name = True
    _attr_name = "Refit thermal models"
    _attr_entity_category = EntityCategory.CONFIG
    _attr_icon = "mdi:brain"

    def __init__(self, coordinator: RoomHeatCoordinator, entry_id: str) -> None:
        self._coordinator = coordinator
        self._attr_unique_id = f"{entry_id}_refit_models"
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, entry_id)}, name="RoomHeat")

    async def async_press(self) -> None:
        await self._coordinator.async_refit_models()
