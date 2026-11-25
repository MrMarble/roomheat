"""RoomHeat: model-predictive room heating/cooling (read-only for now)."""

from __future__ import annotations

import logging
from datetime import timedelta

from homeassistant.const import EVENT_HOMEASSISTANT_STOP, Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.start import async_at_started
from homeassistant.util import dt as dt_util

from .backfill import async_backfill
from .const import BACKFILL_LOOKBACK_DAYS, HISTORY_DB
from .coordinator import RoomHeatConfigEntry, RoomHeatCoordinator
from .history_store import HistoryStore

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.BINARY_SENSOR, Platform.SENSOR]
REFIT_EVERY = timedelta(hours=6)


async def async_setup_entry(hass: HomeAssistant, entry: RoomHeatConfigEntry) -> bool:
    history = HistoryStore(hass.config.path(HISTORY_DB))
    await hass.async_add_executor_job(history.setup)

    coordinator = RoomHeatCoordinator(hass, entry, history)
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = coordinator

    # Keep the partial 5-minute bucket when the entry unloads or HA stops.
    entry.async_on_unload(coordinator.async_flush_history)
    entry.async_on_unload(
        hass.bus.async_listen_once(
            EVENT_HOMEASSISTANT_STOP, coordinator.async_flush_history
        )
    )

    # Backfill from the recorder once HA is fully up (recorder ready, states loaded).
    async def _backfill(hass: HomeAssistant) -> None:
        entry.async_create_background_task(
            hass, _async_run_backfill(hass, coordinator), "roomheat_backfill"
        )

    entry.async_on_unload(async_at_started(hass, _backfill))

    # Re-learn every few hours: the data grows, and so does what the model can say.
    async def _refit(_now) -> None:
        entry.async_create_background_task(
            hass, coordinator.async_refit_models(), "roomheat_refit"
        )

    entry.async_on_unload(async_track_time_interval(hass, _refit, REFIT_EVERY))

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    # Adding/editing/removing a room subentry reloads the entry.
    entry.async_on_unload(entry.add_update_listener(_async_reload))
    return True


async def _async_run_backfill(
    hass: HomeAssistant, coordinator: RoomHeatCoordinator
) -> None:
    try:
        written = await async_backfill(
            hass,
            coordinator.history,
            coordinator.signals(),
            now_ts=dt_util.utcnow().timestamp(),
            lookback_days=BACKFILL_LOOKBACK_DAYS,
        )
    except Exception:  # noqa: BLE001 - never let a backfill problem break the entry
        _LOGGER.exception("History backfill failed")
        return
    coordinator.backfill_result = {"rows": written, "at": dt_util.utcnow().isoformat()}
    _LOGGER.info("History backfill done: %s", written)
    await coordinator.async_refit_models()  # now there is history to learn from


async def _async_reload(hass: HomeAssistant, entry: RoomHeatConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: RoomHeatConfigEntry) -> bool:
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
