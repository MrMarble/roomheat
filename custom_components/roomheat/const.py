"""Constants for RoomHeat."""

from __future__ import annotations

from datetime import timedelta

DOMAIN = "roomheat"

SUBENTRY_ROOM = "room"

# Hub entry data
CONF_OUTDOOR_SENSOR = "outdoor_temperature_sensor"
CONF_WEATHER_ENTITY = "weather_entity"  # outdoor fallback + cloud cover and wind
CONF_CENTRAL_THERMOSTAT = "central_thermostat"  # climate entity in the living room
CONF_AC_UNIT = "ac_unit"  # the single ducted heat-pump climate entity
CONF_AC_POWER_SENSOR = "ac_power_sensor"  # measured electrical power of the AC
CONF_CENTRAL_SEASON_START = "central_season_start"  # ISO date, edited every year
CONF_CENTRAL_SEASON_END = "central_season_end"
CONF_CENTRAL_HOURS_START = "central_hours_start"  # ISO time
CONF_CENTRAL_HOURS_END = "central_hours_end"
CONF_CENTRAL_HEAT_PRICE = "central_heat_price"  # EUR per kWh of heat (marginal; fixed fee is sunk)
CONF_ELECTRICITY_PRICE = "electricity_price"  # EUR per kWh, final price incl. taxes
CONF_AC_COP = "ac_cop"  # average heating COP of the heat pump
DEFAULT_AC_COP = 3.0
DEFAULT_CENTRAL_HEAT_PRICE = 0.126
DEFAULT_ELECTRICITY_PRICE = 0.11
DEFAULT_CENTRAL_HOURS_START = "09:00:00"
DEFAULT_CENTRAL_HOURS_END = "22:00:00"

# Room subentry data
CONF_AREA_ID = "area_id"
CONF_TEMPERATURE_SENSORS = "temperature_sensors"  # in the room, averaged
CONF_RADIATOR_IN_USE = "radiator_in_use"  # False: the room is not heated by the central heating
CONF_INTERNAL_HEAT = "internal_heat_sensors"  # power sensors of appliances that warm the room
CONF_VALVES = "valves"  # radiator valve climate entities; open = hvac_action heating
CONF_DEVICE_SENSORS = "device_temperature_sensors"  # e.g. radiator valve; not averaged
CONF_LEGACY_TEMPERATURE_SENSOR = "temperature_sensor"  # pre-0.1 single sensor
CONF_SCHEDULE = "schedule"  # schedule helper: on = comfort, off = eco
CONF_COMFORT = "comfort_temperature"
CONF_ECO = "eco_temperature"
DEFAULT_COMFORT = 21.0
DEFAULT_ECO = 17.0
CONF_ADJACENT_ROOMS = "adjacent_rooms"  # list of area_ids

UPDATE_INTERVAL = timedelta(minutes=1)  # probe interval; rows are stored per 5 minutes
BACKFILL_LOOKBACK_DAYS = 365
HISTORY_DB = ".storage/roomheat_history.db"
