"""From the history store to fitted rooms and predictions. Pure Python.

Shared by the Home Assistant integration and `tools/fit_model.py`, so the numbers
you see in HA are exactly the ones the offline tool computes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .dataset import RoomFit, RoomSeries, fit_room, make_rows
from .sampling import SAMPLE_SECONDS
from .signals import COOLING, HEATING, WEATHER_TEMPERATURE, action_series
from .thermal import STEP_HOURS, FitResult, feature_values, simulate

STEP = SAMPLE_SECONDS
MIN_USABLE_STEPS = 3 * 288  # fewer than ~3 days of room readings and a fit is noise
READY_SKILL = 0.2  # forecast must beat "stays put" by this much to count as ready
FRESH_STEPS = 3  # a prediction needs a reading no older than this many steps


@dataclass(frozen=True, slots=True)
class RoomSpec:
    key: str
    area_id: str
    name: str
    sensors: tuple[str, ...]
    valves: tuple[str, ...] = ()
    adjacent_area_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HubSpec:
    outdoor: str | None = None
    weather: str | None = None
    thermostat: str | None = None
    ac: str | None = None
    ac_power: str | None = None


def needed_series(hub: HubSpec, rooms: Sequence[RoomSpec]) -> list[str]:
    names = {s for r in rooms for s in r.sensors}
    names |= {action_series(v, HEATING) for r in rooms for v in r.valves}
    names |= {n for n in (hub.outdoor, hub.ac_power) if n}
    if hub.thermostat:
        names.add(action_series(hub.thermostat, HEATING))
    if hub.ac:
        names |= {action_series(hub.ac, HEATING), action_series(hub.ac, COOLING)}
    if hub.weather:
        names.add(action_series(hub.weather, WEATHER_TEMPERATURE))
    return sorted(names)


def mean_of(values: Sequence[float | None]) -> float | None:
    got = [v for v in values if v is not None]
    return sum(got) / len(got) if got else None


@dataclass(slots=True)
class Dataset:
    start: int
    n: int
    rooms: dict[str, RoomSeries]  # by RoomSpec.key
    outdoor: list[float | None]
    boiler: list[float | None]
    ac_heat: list[float | None]
    ac_cool: list[float | None]


def build_dataset(
    rows: Mapping[str, Mapping[int, float]],
    hub: HubSpec,
    rooms: Sequence[RoomSpec],
    start: int,
    end: int,
) -> Dataset:
    """Align everything on the 5-minute grid [start, end) and build each room's inputs."""
    grid = range(start, end, STEP)
    n = len(grid)

    def column(series: str | None) -> list[float | None]:
        data = rows.get(series or "", {})
        return [data.get(ts) for ts in grid]

    outdoor = column(hub.outdoor)
    if hub.weather:  # the weather entity stands in where the sensor has gaps
        fallback = column(action_series(hub.weather, WEATHER_TEMPERATURE))
        outdoor = [o if o is not None else w for o, w in zip(outdoor, fallback)]
    boiler = column(hub.thermostat and action_series(hub.thermostat, HEATING))

    kw = [None if p is None else p / 1000 for p in column(hub.ac_power)]

    def ac_input(flag: list[float | None]) -> list[float | None]:
        if not any(v is not None for v in flag):
            return [None] * n
        if not hub.ac_power:
            return flag
        return [None if f is None or p is None else f * p for f, p in zip(flag, kw)]

    ac_heat = ac_input(column(hub.ac and action_series(hub.ac, HEATING)))
    ac_cool = ac_input(column(hub.ac and action_series(hub.ac, COOLING)))

    temps = {
        r.area_id: [mean_of([rows.get(s, {}).get(ts) for s in r.sensors]) for ts in grid]
        for r in rooms
    }
    out: dict[str, RoomSeries] = {}
    for r in rooms:
        neighbours = [
            mean_of([temps[a][i] for a in r.adjacent_area_ids if a in temps]) for i in range(n)
        ]
        valve_cols = [column(action_series(v, HEATING)) for v in r.valves]
        valve = [mean_of([c[i] for c in valve_cols]) for i in range(n)] if valve_cols else None
        out[r.key] = RoomSeries(temps[r.area_id], outdoor, neighbours, boiler, ac_heat, ac_cool, valve)
    return Dataset(start, n, out, outdoor, boiler, ac_heat, ac_cool)


def usable_steps(series: RoomSeries) -> int:
    return sum(v is not None for v in series.temp)


def skill(fit: RoomFit | None) -> float | None:
    """1 - model error / "stays put" error on unseen data. None without a hold-out."""
    if fit is None or fit.holdout is None or not fit.holdout.rmse_persistence:
        return None
    return 1 - fit.holdout.rmse_model / fit.holdout.rmse_persistence


def status(fit: RoomFit | None, usable: int) -> str:
    """collecting: not enough data to judge. ready: beats "stays put". poor: does not."""
    s = skill(fit)
    if usable < MIN_USABLE_STEPS or s is None:
        return "collecting"
    return "ready" if s >= READY_SKILL else "poor"


def describe(result: FitResult) -> dict[str, float | int | None]:
    """The fitted numbers in units a person can reason about (None: not learnable)."""
    c, active, per_hour = result.coef, result.active, 1 / STEP_HOURS

    def tau(name: str) -> float | None:
        return round(STEP_HOURS / c[name], 1) if name in active and c[name] > 1e-6 else None

    def gain(name: str) -> float | None:
        return round(c[name] * per_hour, 3) if name in active else None

    return {
        "outdoor_time_constant_h": tau("outdoor"),
        "neighbour_time_constant_h": tau("neighbours"),
        "central_heating_gain_c_per_h": gain("boiler"),
        "radiator_lag_min": result.tau_boiler_steps * STEP // 60 if "boiler" in active else None,
        "ac_heating_gain_c_per_h_per_kw": gain("ac_heat"),
        "ac_cooling_gain_c_per_h_per_kw": gain("ac_cool"),
        "unmodelled_heat_c_per_h": round((c["bias_up"] - c["bias_down"]) * per_hour, 3),
    }


def predict_ahead(
    result: FitResult, series: RoomSeries, horizon: int
) -> tuple[int, float, float] | None:
    """Temperature `horizon` steps after the newest usable reading, if inputs stay as they are.

    Returns (index of that reading in the series, temperature then, predicted temperature).
    None when the newest usable reading is stale.
    """
    rows = make_rows(series, result.tau_boiler_steps)
    for i in range(len(rows) - 1, -1, -1):
        row = rows[i]
        if row.t is None or feature_values(row, result.active) is None:
            continue
        if len(rows) - 1 - i > FRESH_STEPS:
            return None
        pred = simulate(result, [row] * horizon, 0, horizon)
        return None if pred is None else (i, row.t, pred)
    return None
