"""Coordinator: reads all configured rooms and runs the control pipeline."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import date, time
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from . import control
from .history_store import SRC_LIVE, HistoryStore
from .learn import HubSpec, RoomSpec, describe
from .models import ModelManager
from .sampling import BucketAccumulator
from .signals import (
    CLOUD_COVERAGE,
    HEATING,
    POWER,
    TEMPERATURE,
    WEATHER_TEMPERATURE,
    WIND_SPEED,
    Signal,
    build_signals,
)
from .units import signal_value
from .const import (
    CONF_AC_COP,
    CONF_AC_POWER_SENSOR,
    CONF_CENTRAL_HEAT_PRICE,
    CONF_COMFORT,
    CONF_ECO,
    CONF_ELECTRICITY_PRICE,
    CONF_SCHEDULE,
    DEFAULT_AC_COP,
    DEFAULT_CENTRAL_HEAT_PRICE,
    DEFAULT_COMFORT,
    DEFAULT_ECO,
    DEFAULT_ELECTRICITY_PRICE,
    CONF_AC_UNIT,
    CONF_ADJACENT_ROOMS,
    CONF_AREA_ID,
    CONF_CENTRAL_HOURS_END,
    CONF_CENTRAL_HOURS_START,
    CONF_CENTRAL_SEASON_END,
    CONF_CENTRAL_SEASON_START,
    CONF_CENTRAL_THERMOSTAT,
    CONF_OUTDOOR_SENSOR,
    CONF_DEVICE_SENSORS,
    CONF_LEGACY_TEMPERATURE_SENSOR,
    CONF_TEMPERATURE_SENSORS,
    CONF_INTERNAL_HEAT,
    CONF_RADIATOR_IN_USE,
    CONF_VALVES,
    CONF_WEATHER_ENTITY,
    DOMAIN,
    SUBENTRY_ROOM,
    UPDATE_INTERVAL,
)
from .model import (
    CentralHeatingWindow,
    Economics,
    HubReading,
    Recommendation,
    RoomConfig,
    RoomReading,
    RoomState,
    symmetric_adjacency,
)

_LOGGER = logging.getLogger(__name__)

type RoomHeatConfigEntry = ConfigEntry[RoomHeatCoordinator]


class RoomHeatCoordinator(DataUpdateCoordinator[dict[str, RoomState]]):
    """Data is keyed by room subentry id."""

    config_entry: RoomHeatConfigEntry

    def __init__(
        self, hass: HomeAssistant, entry: RoomHeatConfigEntry, history: HistoryStore
    ) -> None:
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=DOMAIN,
            update_interval=UPDATE_INTERVAL,
        )
        self.hub = HubReading()
        self.recommendation: Recommendation | None = None
        self.history = history
        self._bucket = BucketAccumulator()
        self.models = ModelManager(history, hass.async_add_executor_job)
        self._refit_lock = asyncio.Lock()  # one refit at a time, however often it is asked for
        self.last_write: float | None = None  # unix time of the last history write
        self.backfill_result: dict[str, Any] | None = None

    # -- configuration -------------------------------------------------------

    def rooms(self) -> dict[str, RoomConfig]:
        """Build the room list from the config subentries."""
        areas = ar.async_get(self.hass)
        subentries = [
            s
            for s in self.config_entry.subentries.values()
            if s.subentry_type == SUBENTRY_ROOM
        ]
        # adjacency is symmetric: a link declared on either room counts for both
        adjacent = symmetric_adjacency(
            {s.data[CONF_AREA_ID]: s.data.get(CONF_ADJACENT_ROOMS, []) for s in subentries}
        )
        rooms: dict[str, RoomConfig] = {}
        for sub in subentries:
            area_id = sub.data[CONF_AREA_ID]
            area = areas.async_get_area(area_id)
            rooms[sub.subentry_id] = RoomConfig(
                subentry_id=sub.subentry_id,
                area_id=area_id,
                name=area.name if area else sub.title,
                temperature_sensors=tuple(
                    sub.data.get(CONF_TEMPERATURE_SENSORS)
                    or [sub.data[CONF_LEGACY_TEMPERATURE_SENSOR]]
                ),
                device_sensors=tuple(sub.data.get(CONF_DEVICE_SENSORS, [])),
                valves=tuple(sub.data.get(CONF_VALVES, [])),
                radiator_in_use=sub.data.get(CONF_RADIATOR_IN_USE, True),
                internal_heat=tuple(sub.data.get(CONF_INTERNAL_HEAT, [])),
                adjacent_area_ids=adjacent[area_id],
                schedule=sub.data.get(CONF_SCHEDULE) or None,
                comfort=float(sub.data.get(CONF_COMFORT, DEFAULT_COMFORT)),
                eco=float(sub.data.get(CONF_ECO, DEFAULT_ECO)),
            )
        return rooms

    def central_window(self) -> CentralHeatingWindow | None:
        """The configured central heating window, or None if absent/invalid."""
        data = self.config_entry.data
        try:
            return CentralHeatingWindow(
                season_start=date.fromisoformat(data[CONF_CENTRAL_SEASON_START]),
                season_end=date.fromisoformat(data[CONF_CENTRAL_SEASON_END]),
                hours_start=time.fromisoformat(data[CONF_CENTRAL_HOURS_START]),
                hours_end=time.fromisoformat(data[CONF_CENTRAL_HOURS_END]),
            )
        except (KeyError, ValueError):
            return None

    # -- reading -------------------------------------------------------------

    def _read_temperature(self, entity_id: str | None) -> float | None:
        """Read a temperature sensor in °C; None when missing/unavailable."""
        if not entity_id:
            return None
        return signal_value(TEMPERATURE, self.hass.states.get(entity_id))

    def _read_power_w(self, entity_id: str | None) -> float | None:
        """Read a power sensor in watts; None when missing/unavailable."""
        if not entity_id:
            return None
        return signal_value(POWER, self.hass.states.get(entity_id))

    def _read_climate(self, entity_id: str | None) -> tuple[str | None, str | None]:
        """(hvac mode, hvac_action) of a climate entity; None where unknown."""
        if not entity_id:
            return None, None
        state = self.hass.states.get(entity_id)
        if state is None or state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            return None, None
        return state.state, state.attributes.get("hvac_action")

    def _read_room(self, room: RoomConfig) -> RoomReading:
        """Read a room's own sensors (adjacent temperatures are filled in later)."""
        sensors = {e: self._read_temperature(e) for e in room.temperature_sensors}
        available = [v for v in sensors.values() if v is not None]
        return RoomReading(
            temperature=sum(available) / len(available) if available else None,
            sensor_temperatures=sensors,
            device_temperatures={
                e: self._read_temperature(e) for e in room.device_sensors
            },
            valve_open={
                e: signal_value(HEATING, self.hass.states.get(e)) for e in room.valves
            },
        )

    def _read_schedule(self, room: RoomConfig) -> tuple[str | None, Any]:
        """(state, next_event) of the room's schedule helper; (None, None) if unreadable."""
        state = self.hass.states.get(room.schedule) if room.schedule else None
        if state is None or state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            return None, None
        next_event = state.attributes.get("next_event")
        return state.state, dt_util.as_local(next_event) if next_event else None

    def _learned_rate(self, subentry_id: str) -> float | None:
        """Learned central heating gain in °C/h for a room, if the model has one."""
        model = self.models.states.get(subentry_id)
        if model is None or model.fit is None:
            return None
        return describe(model.fit.result)["central_heating_gain_c_per_h"]

    # -- cycle ---------------------------------------------------------------

    async def _async_update_data(self) -> dict[str, RoomState]:
        now = dt_util.now()
        rooms = self.rooms()
        by_area = {room.area_id: room for room in rooms.values()}

        conf = self.config_entry.data
        window = self.central_window()
        _, central_action = self._read_climate(conf.get(CONF_CENTRAL_THERMOSTAT))
        ac_mode, ac_action = self._read_climate(conf.get(CONF_AC_UNIT))
        outdoor = self._read_temperature(conf.get(CONF_OUTDOOR_SENSOR))
        outdoor_source = "sensor" if outdoor is not None else None
        weather = self.hass.states.get(conf.get(CONF_WEATHER_ENTITY) or "")
        weather_temp = signal_value(WEATHER_TEMPERATURE, weather, self._temp_unit)
        if outdoor is None and weather_temp is not None:
            outdoor, outdoor_source = weather_temp, "weather"
        self.hub = HubReading(
            outdoor_temperature=outdoor,
            outdoor_source=outdoor_source,
            cloud_coverage=signal_value(CLOUD_COVERAGE, weather),
            wind_speed=signal_value(WIND_SPEED, weather),
            central_window=window,
            central_available=window.is_open(now) if window else False,
            central_hvac_action=central_action,
            ac_hvac_mode=ac_mode,
            ac_hvac_action=ac_action,
            ac_power_w=self._read_power_w(conf.get(CONF_AC_POWER_SENSOR)),
        )

        # 1. read every room first, so adjacent temperatures are consistent
        readings = {sid: self._read_room(r) for sid, r in rooms.items()}
        for sid, room in rooms.items():
            readings[sid].adjacent_temperatures = {
                area_id: readings[by_area[area_id].subentry_id].temperature
                for area_id in room.adjacent_area_ids
                if area_id in by_area
            }

        econ = Economics(
            central_price=float(conf.get(CONF_CENTRAL_HEAT_PRICE, DEFAULT_CENTRAL_HEAT_PRICE)),
            electricity_price=float(conf.get(CONF_ELECTRICITY_PRICE, DEFAULT_ELECTRICITY_PRICE)),
            ac_cop=float(conf.get(CONF_AC_COP, DEFAULT_AC_COP)),
            ac_available=bool(conf.get(CONF_AC_UNIT)),
        )

        # 2..3 resolve target and plan; 4. actuation does not exist yet
        states: dict[str, RoomState] = {}
        for sid, room in rooms.items():
            reading = readings[sid]
            target = control.resolve_target(room, *self._read_schedule(room))
            plan = control.plan(
                room, reading, target, now, self._learned_rate(sid)
            )
            states[sid] = RoomState(room, reading, target, plan)

        self.recommendation = control.recommend(
            {sid: (st.config, st.plan) for sid, st in states.items()}, self.hub, econ, now
        )
        await self._async_record(now.timestamp())
        return states

    @property
    def _temp_unit(self) -> str:
        return self.hass.config.units.temperature_unit

    # -- history -------------------------------------------------------------

    def signals(self) -> list[Signal]:
        """Everything worth recording, derived from the current configuration."""
        conf = self.config_entry.data
        return build_signals(
            room_sensors=[
                e
                for r in self.rooms().values()
                for e in (*r.temperature_sensors, *r.device_sensors)
            ],
            valves=[v for r in self.rooms().values() for v in r.valves],
            internal_heat=[e for r in self.rooms().values() for e in r.internal_heat],
            outdoor=conf.get(CONF_OUTDOOR_SENSOR),
            thermostat=conf.get(CONF_CENTRAL_THERMOSTAT),
            ac=conf.get(CONF_AC_UNIT),
            ac_power=conf.get(CONF_AC_POWER_SENSOR),
            weather=conf.get(CONF_WEATHER_ENTITY),
        )

    async def _async_record(self, ts: float) -> None:
        """Probe every signal now; finished 5-minute buckets are written to disk."""
        probe = {
            s.series: signal_value(
                s.kind, self.hass.states.get(s.entity_id), self._temp_unit
            )
            for s in self.signals()
        }
        rows = self._bucket.add(ts, probe)
        await self._async_write(rows)
        if rows:  # a 5-minute bucket just closed: refresh the 1-hour prediction
            await self._async_predict(ts)

    # -- thermal model -------------------------------------------------------

    def _model_specs(self) -> tuple[HubSpec, list[RoomSpec]]:
        conf = self.config_entry.data
        hub = HubSpec(
            outdoor=conf.get(CONF_OUTDOOR_SENSOR),
            weather=conf.get(CONF_WEATHER_ENTITY),
            thermostat=conf.get(CONF_CENTRAL_THERMOSTAT),
            ac=conf.get(CONF_AC_UNIT),
            ac_power=conf.get(CONF_AC_POWER_SENSOR),
        )
        window = self.central_window()
        central_open = (
            None
            if window is None
            else lambda ts: window.is_open(dt_util.as_local(dt_util.utc_from_timestamp(ts)))
        )
        hub = replace(hub, central_open=central_open)
        rooms = [
            RoomSpec(
                key=r.subentry_id,
                area_id=r.area_id,
                name=r.name,
                sensors=r.temperature_sensors,
                valves=r.valves,
                radiator_in_use=r.radiator_in_use,
                internal_heat=r.internal_heat,
                adjacent_area_ids=r.adjacent_area_ids,
            )
            for r in self.rooms().values()
        ]
        return hub, rooms

    async def async_refit_models(self, *_: Any) -> None:
        """Re-learn every room from the history store (slow; runs in an executor)."""
        if self._refit_lock.locked():
            _LOGGER.debug("A refit is already running; not starting another")
            return
        async with self._refit_lock:
            hub, rooms = self._model_specs()
            now = dt_util.utcnow().timestamp()
            try:
                await self.models.refit(hub, rooms, now)
                await self.models.predict(hub, rooms, now)  # show predictions right away
            except Exception:  # noqa: BLE001 - never break the entry over the model
                _LOGGER.exception("Refitting the thermal models failed")
                return
        _LOGGER.info(
            "Thermal models refitted: %s",
            {self.rooms()[k].name: s.status for k, s in self.models.states.items() if k in self.rooms()},
        )
        self.async_update_listeners()

    async def _async_predict(self, ts: float) -> None:
        hub, rooms = self._model_specs()
        try:
            await self.models.predict(hub, rooms, ts)
        except Exception:  # noqa: BLE001
            _LOGGER.exception("Predicting room temperatures failed")

    async def async_flush_history(self, *_: Any) -> None:
        """Write the partial bucket (called on unload and on Home Assistant stop)."""
        await self._async_write(self._bucket.flush())

    async def _async_write(self, rows: list[tuple[str, int, float]]) -> None:
        if not rows:
            return
        try:
            await self.hass.async_add_executor_job(
                self.history.write, [(s, ts, v, SRC_LIVE) for s, ts, v in rows]
            )
            self.last_write = dt_util.utcnow().timestamp()
        except Exception:  # noqa: BLE001 - a full disk must not stop the control loop
            _LOGGER.exception("Could not write %d history rows", len(rows))
