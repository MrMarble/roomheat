"""Config flow: one hub entry, one `room` subentry per area."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import selector

from .const import (
    CONF_ADJACENT_ROOMS,
    CONF_AREA_ID,
    CONF_AC_POWER_SENSOR,
    CONF_CENTRAL_HEAT_PRICE,
    CONF_AC_UNIT,
    CONF_CENTRAL_HOURS_END,
    CONF_CENTRAL_HOURS_START,
    CONF_CENTRAL_SEASON_END,
    CONF_CENTRAL_SEASON_START,
    CONF_CENTRAL_THERMOSTAT,
    CONF_DEVICE_SENSORS,
    CONF_ELECTRICITY_PRICE,
    CONF_LEGACY_TEMPERATURE_SENSOR,
    CONF_OUTDOOR_SENSOR,
    CONF_TEMPERATURE_SENSORS,
    CONF_INTERNAL_HEAT,
    CONF_RADIATOR_IN_USE,
    CONF_VALVES,
    CONF_WEATHER_ENTITY,
    DEFAULT_CENTRAL_HEAT_PRICE,
    DEFAULT_CENTRAL_HOURS_END,
    DEFAULT_CENTRAL_HOURS_START,
    DEFAULT_ELECTRICITY_PRICE,
    DOMAIN,
    SUBENTRY_ROOM,
)

_TEMP_SENSOR_SELECTOR = selector.EntitySelector(
    selector.EntitySelectorConfig(domain="sensor", device_class="temperature")
)
_TEMP_SENSORS_SELECTOR = selector.EntitySelector(
    selector.EntitySelectorConfig(
        domain="sensor", device_class="temperature", multiple=True
    )
)


class RoomHeatConfigFlow(ConfigFlow, domain=DOMAIN):
    """Hub: holds the global (non-room) settings."""

    VERSION = 1

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        return {SUBENTRY_ROOM: RoomSubentryFlow}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = _validate_hub(user_input)
            if not errors:
                return self.async_create_entry(title="RoomHeat", data=user_input)
        return self.async_show_form(
            step_id="user",
            data_schema=_hub_schema(user_input or {}),
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = _validate_hub(user_input)
            if not errors:
                # replace (not merge) so clearing an optional entity really clears it
                return self.async_update_reload_and_abort(entry, data=user_input)
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=_hub_schema(user_input or entry.data),
            errors=errors,
        )


def _validate_hub(user_input: dict[str, Any]) -> dict[str, str]:
    if user_input[CONF_CENTRAL_SEASON_START] > user_input[CONF_CENTRAL_SEASON_END]:
        return {"base": "season_order"}  # ISO dates compare correctly as strings
    return {}


def _hub_schema(d: Any) -> vol.Schema:
    def suggested(key: str) -> dict[str, Any]:
        return {"suggested_value": d.get(key)}

    price = selector.NumberSelector(
        selector.NumberSelectorConfig(
            min=0,
            max=2,
            step=0.001,
            mode=selector.NumberSelectorMode.BOX,
            unit_of_measurement="€/kWh",
        )
    )
    climate = selector.EntitySelector(selector.EntitySelectorConfig(domain="climate"))
    return vol.Schema(
        {
            vol.Optional(CONF_OUTDOOR_SENSOR, description=suggested(CONF_OUTDOOR_SENSOR)): _TEMP_SENSOR_SELECTOR,
            vol.Optional(CONF_WEATHER_ENTITY, description=suggested(CONF_WEATHER_ENTITY)): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="weather")
            ),
            vol.Optional(CONF_CENTRAL_THERMOSTAT, description=suggested(CONF_CENTRAL_THERMOSTAT)): climate,
            vol.Optional(CONF_AC_UNIT, description=suggested(CONF_AC_UNIT)): climate,
            vol.Optional(CONF_AC_POWER_SENSOR, description=suggested(CONF_AC_POWER_SENSOR)): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="power")
            ),
            vol.Required(CONF_CENTRAL_SEASON_START, description=suggested(CONF_CENTRAL_SEASON_START)): selector.DateSelector(),
            vol.Required(CONF_CENTRAL_SEASON_END, description=suggested(CONF_CENTRAL_SEASON_END)): selector.DateSelector(),
            vol.Required(
                CONF_CENTRAL_HOURS_START,
                default=d.get(CONF_CENTRAL_HOURS_START, DEFAULT_CENTRAL_HOURS_START),
            ): selector.TimeSelector(),
            vol.Required(
                CONF_CENTRAL_HOURS_END,
                default=d.get(CONF_CENTRAL_HOURS_END, DEFAULT_CENTRAL_HOURS_END),
            ): selector.TimeSelector(),
            vol.Required(
                CONF_CENTRAL_HEAT_PRICE,
                default=d.get(CONF_CENTRAL_HEAT_PRICE, DEFAULT_CENTRAL_HEAT_PRICE),
            ): price,
            vol.Required(
                CONF_ELECTRICITY_PRICE,
                default=d.get(CONF_ELECTRICITY_PRICE, DEFAULT_ELECTRICITY_PRICE),
            ): price,
        }
    )


def _suggest_temperature_sensor(hass: HomeAssistant, area_id: str) -> str | None:
    """First temperature sensor assigned to the area (directly or via its device)."""
    ent_reg = er.async_get(hass)
    dev_reg = dr.async_get(hass)
    for entry in ent_reg.entities.values():
        if entry.domain != "sensor" or entry.disabled:
            continue
        if (entry.device_class or entry.original_device_class) != "temperature":
            continue
        entity_area = entry.area_id
        if entity_area is None and entry.device_id:
            device = dev_reg.async_get(entry.device_id)
            entity_area = device.area_id if device else None
        if entity_area == area_id:
            return entry.entity_id
    return None


class RoomSubentryFlow(ConfigSubentryFlow):
    """Add / reconfigure a room."""

    def __init__(self) -> None:
        self._area_id: str | None = None

    def _other_room_areas(self, exclude_area_id: str) -> list[selector.SelectOptionDict]:
        """Areas of the other already-configured rooms, as selector options."""
        areas = ar.async_get(self.hass)
        options: list[selector.SelectOptionDict] = []
        for sub in self._get_entry().subentries.values():
            other = sub.data.get(CONF_AREA_ID)
            if sub.subentry_type != SUBENTRY_ROOM or other in (None, exclude_area_id):
                continue
            area = areas.async_get_area(other)
            options.append(
                selector.SelectOptionDict(
                    value=other, label=area.name if area else sub.title
                )
            )
        return options

    def _room_schema(
        self, area_id: str, defaults: dict[str, Any]
    ) -> vol.Schema:
        room_sensors = defaults.get(CONF_TEMPERATURE_SENSORS) or (
            [defaults[CONF_LEGACY_TEMPERATURE_SENSOR]]
            if defaults.get(CONF_LEGACY_TEMPERATURE_SENSOR)
            else vol.UNDEFINED
        )
        fields: dict[Any, Any] = {
            vol.Required(
                CONF_TEMPERATURE_SENSORS, default=room_sensors
            ): _TEMP_SENSORS_SELECTOR,
            vol.Optional(
                CONF_DEVICE_SENSORS, default=defaults.get(CONF_DEVICE_SENSORS, [])
            ): _TEMP_SENSORS_SELECTOR,
            vol.Optional(
                CONF_VALVES, default=defaults.get(CONF_VALVES, [])
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="climate", multiple=True)
            ),
            vol.Optional(
                CONF_RADIATOR_IN_USE, default=defaults.get(CONF_RADIATOR_IN_USE, True)
            ): selector.BooleanSelector(),
            vol.Optional(
                CONF_INTERNAL_HEAT, default=defaults.get(CONF_INTERNAL_HEAT, [])
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="power", multiple=True)
            ),
        }
        options = self._other_room_areas(area_id)
        if options:
            fields[
                vol.Optional(
                    CONF_ADJACENT_ROOMS,
                    default=defaults.get(CONF_ADJACENT_ROOMS, []),
                )
            ] = selector.SelectSelector(
                selector.SelectSelectorConfig(options=options, multiple=True)
            )
        return vol.Schema(fields)

    def _used_area_ids(self) -> set[str]:
        return {
            sub.data[CONF_AREA_ID]
            for sub in self._get_entry().subentries.values()
            if sub.subentry_type == SUBENTRY_ROOM
        }

    # -- add ---------------------------------------------------------------

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            area_id = user_input[CONF_AREA_ID]
            if area_id in self._used_area_ids():
                errors[CONF_AREA_ID] = "already_configured"
            else:
                self._area_id = area_id
                return await self.async_step_sensors()
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {vol.Required(CONF_AREA_ID): selector.AreaSelector()}
            ),
            errors=errors,
        )

    async def async_step_sensors(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        assert self._area_id is not None
        area = ar.async_get(self.hass).async_get_area(self._area_id)
        if user_input is not None:
            return self.async_create_entry(
                title=area.name if area else self._area_id,
                data={CONF_AREA_ID: self._area_id, **user_input},
                unique_id=self._area_id,
            )
        suggestion = _suggest_temperature_sensor(self.hass, self._area_id)
        return self.async_show_form(
            step_id="sensors",
            data_schema=self._room_schema(
                self._area_id,
                {CONF_TEMPERATURE_SENSORS: [suggestion]} if suggestion else {},
            ),
            description_placeholders={"area": area.name if area else self._area_id},
        )

    # -- edit --------------------------------------------------------------

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        subentry = self._get_reconfigure_subentry()
        area_id = subentry.data[CONF_AREA_ID]  # the area itself is not editable
        if user_input is not None:
            return self.async_update_and_abort(
                self._get_entry(),
                subentry,
                data={CONF_AREA_ID: area_id, **user_input},
            )
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self._room_schema(area_id, dict(subentry.data)),
            description_placeholders={"area": subentry.title},
        )
