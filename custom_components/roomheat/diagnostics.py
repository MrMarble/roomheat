"""Downloadable diagnostics: what is configured, and what the history store holds."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from homeassistant.core import HomeAssistant

from .coordinator import RoomHeatConfigEntry
from .learn import describe, skill
from .history_store import HistoryStore


def _iso(ts: float | int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: RoomHeatConfigEntry
) -> dict[str, Any]:
    coordinator = entry.runtime_data
    store: HistoryStore = coordinator.history
    summary = await hass.async_add_executor_job(store.summary)
    signals = {s.series: s for s in coordinator.signals()}

    series: dict[str, Any] = {}
    for name, info in sorted(summary.items()):
        series[name] = {
            **info,
            "first": _iso(info["first"]),
            "last": _iso(info["last"]),
            "still_recorded": name in signals,
        }
    for name in signals.keys() - summary.keys():  # configured but no rows yet
        series[name] = {"rows": 0, "still_recorded": True}

    models: dict[str, Any] = {}
    for key, room in coordinator.rooms().items():
        state = coordinator.models.states.get(key)
        if state is None:
            continue
        info: dict[str, Any] = {
            "status": state.status,
            "usable_hours": round(state.usable_hours, 1),
            "trained_at": _iso(state.trained_at),
            "predicted_1h": state.predicted,
            "last_forecast_error_c": state.last_error,
        }
        if state.fit is not None:
            result, holdout = state.fit.result, state.fit.holdout
            info |= {
                "learned": describe(result),
                "features_used": list(result.active),
                "training_windows": result.n,
                "forecast_error_1h_c": None if holdout is None else round(holdout.rmse_model, 3),
                "stays_put_error_1h_c": None if holdout is None else round(holdout.rmse_persistence, 3),
                "skill": None if skill(state.fit) is None else round(skill(state.fit), 3),
            }
        models[room.name] = info

    hub = coordinator.hub
    return {
        "config": dict(entry.data),
        "rooms": {
            room.name: {
                "area_id": room.area_id,
                "temperature_sensors": list(room.temperature_sensors),
                "device_sensors": list(room.device_sensors),
                "valves": list(room.valves),
                "adjacent": list(room.adjacent_area_ids),
            }
            for room in coordinator.rooms().values()
        },
        "hub_reading": {
            "central_available": hub.central_available,
            "central_window": None
            if hub.central_window is None
            else {
                "season": f"{hub.central_window.season_start} .. {hub.central_window.season_end}",
                "hours": f"{hub.central_window.hours_start} .. {hub.central_window.hours_end}",
            },
            "central_hvac_action": hub.central_hvac_action,
            "ac_hvac_mode": hub.ac_hvac_mode,
            "ac_hvac_action": hub.ac_hvac_action,
            "ac_power_w": hub.ac_power_w,
            "outdoor_temperature": hub.outdoor_temperature,
            "outdoor_source": hub.outdoor_source,
            "cloud_coverage": hub.cloud_coverage,
            "wind_speed": hub.wind_speed,
        },
        "models": models,
        "history": {
            "last_write": _iso(coordinator.last_write),
            "last_backfill": coordinator.backfill_result,
            "total_rows": sum(i.get("rows", 0) for i in series.values()),
            "series": series,
        },
    }
