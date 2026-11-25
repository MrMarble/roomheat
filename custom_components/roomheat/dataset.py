"""Turn per-signal time series into model rows, and search the boiler lag.

Pure Python. All series are aligned on one 5-minute grid; None marks gaps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .thermal import (
    FEATURES,
    STEP_HOURS,
    Evaluation,
    FitResult,
    StepRow,
    accumulate,
    evaluate,
    fit,
    result_from,
)

# candidate lags (in 5-minute steps) of the radiator's response: 0, 15, 30, 60, 120, 240 min
BOILER_TAU_GRID = (0, 3, 6, 12, 24, 48)


@dataclass(slots=True)
class RoomSeries:
    """One room's inputs on the grid. Missing inputs are all-None lists."""

    temp: list[float | None]
    outdoor: list[float | None]
    neighbours: list[float | None]
    boiler: list[float | None]  # boiler-on fraction 0..1
    ac_heat: list[float | None]
    ac_cool: list[float | None]
    # Fraction of the room's radiator valves that are open (0..1). None means the
    # room has no controllable valve, so its radiators count as always open.
    valve: list[float | None] | None = None
    # Measured appliance heat in the room (kW). None means the room has no such sensor.
    internal: list[float | None] | None = None


def lowpass(values: Sequence[float | None], tau_steps: int) -> list[float | None]:
    """First-order low-pass. tau_steps = 0 returns the input unchanged.

    A gap yields None for that step and leaves the filter state untouched.
    """
    alpha = 1.0 / (1 + tau_steps)
    out: list[float | None] = []
    state: float | None = None
    for v in values:
        if v is None:
            out.append(None)
            continue
        state = v if state is None else state + alpha * (v - state)
        out.append(state)
    return out


def gated_boiler(series: RoomSeries) -> list[float | None]:
    """Heat the radiators are asked to deliver: boiler on AND the valve open."""
    if series.valve is None:
        return series.boiler
    return [
        None if b is None or v is None else b * v
        for b, v in zip(series.boiler, series.valve)
    ]


def make_rows(series: RoomSeries, tau_boiler_steps: int) -> list[StepRow]:
    boiler = lowpass(gated_boiler(series), tau_boiler_steps)
    return [
        StepRow(t, o, n, b, h, c, i)
        for t, o, n, b, h, c, i in zip(
            series.temp,
            series.outdoor,
            series.neighbours,
            boiler,
            series.ac_heat,
            series.ac_cool,
            series.internal or [None] * len(series.temp),
        )
    ]


def _varies(values: Sequence[float | None]) -> bool:
    return any(v for v in values if v is not None)  # some non-None, non-zero value


def active_features(series: RoomSeries) -> tuple[str, ...]:
    """Features the data can actually identify (an input that is never present, or
    never non-zero, cannot be learned)."""
    present = {
        "outdoor": any(v is not None for v in series.outdoor),
        "neighbours": any(v is not None for v in series.neighbours),
        "boiler": _varies(gated_boiler(series)),
        "ac_heat": _varies(series.ac_heat),
        "ac_cool": _varies(series.ac_cool),
        "internal": series.internal is not None and _varies(series.internal),
        "bias_up": True,
        "bias_down": True,
    }
    return tuple(f for f in FEATURES if present[f])


def _slice(series: RoomSeries, a: int, b: int) -> RoomSeries:
    return RoomSeries(
        series.temp[a:b],
        series.outdoor[a:b],
        series.neighbours[a:b],
        series.boiler[a:b],
        series.ac_heat[a:b],
        series.ac_cool[a:b],
        None if series.valve is None else series.valve[a:b],
        None if series.internal is None else series.internal[a:b],
    )


def exposure_hours(series: RoomSeries) -> dict[str, float]:
    """Hours in which each input was actually doing something. A coefficient learned from
    an input that was rarely on is not to be trusted, and this makes that visible."""

    def hours(values: Sequence[float | None], threshold: float) -> float:
        return round(sum(1 for v in values if v is not None and v > threshold) * STEP_HOURS, 1)

    return {
        "boiler": hours(gated_boiler(series), 0.05),
        "ac_heat": hours(series.ac_heat, 0.0),
        "ac_cool": hours(series.ac_cool, 0.0),
        "internal": hours(series.internal or [], 0.0),
    }


@dataclass(frozen=True, slots=True)
class RoomFit:
    result: FitResult
    train_steps: int
    holdout: Evaluation | None  # None if there was no hold-out or it had no usable windows
    dropped: tuple[str, ...] = ()  # features that were identifiable but did not help, so were left out
    exposure: dict[str, float] = field(default_factory=dict)  # hours each input was active


# Leaving a feature out is accepted unless it makes the validation forecast worse than
# this much: between equal models the simpler one generalises better.
KEEP_MARGIN = 1.02
OPTIONAL_FEATURES = ("outdoor", "neighbours", "boiler", "ac_heat", "ac_cool", "internal")


def _select_features(
    train: RoomSeries, tau: int, active: Sequence[str], span: int, horizon: int
) -> tuple[str, ...]:
    """Backward elimination on a validation stretch kept inside the training part.

    Fits on the first two thirds of `train` and scores each candidate on the last third,
    so the final hold-out stays untouched. A feature is dropped when the forecast is no
    worse (within KEEP_MARGIN) without it.
    """
    active = tuple(active)
    rows = make_rows(train, tau)
    cut = int(len(rows) * 2 / 3)
    ne = accumulate(rows[:cut], active, span)
    if ne is None:
        return active
    validation = rows[cut:]

    def score(names: Sequence[str]) -> float:
        idx = [active.index(n) for n in names]
        result = result_from(ne.subset(idx), names, span, tau)
        ev = evaluate(result, validation, horizon)
        return ev.rmse_model if ev else float("inf")

    current, best = list(active), score(active)
    if best == float("inf"):  # no usable validation stretch: keep everything
        return active
    while True:
        candidates = [(score([n for n in current if n != f]), f) for f in current if f in OPTIONAL_FEATURES]
        if not candidates:
            break
        s, f = min(candidates)
        if s > best * KEEP_MARGIN:
            break
        current.remove(f)
        best = min(best, s)
    return tuple(current)


def fit_room(
    series: RoomSeries,
    *,
    span: int = 6,
    horizon: int = 12,
    holdout_fraction: float = 0.25,
    select: bool = True,
) -> RoomFit | None:
    """Fit one room: pick the boiler lag and the useful features on the training part,
    score the result on the rest.

    `horizon` is the open-loop prediction length (12 steps = 1 hour).

    Only the stretch that actually holds readings is used. The caller examines a long
    window (e.g. 60 days) but a new install has data for the last few days only, and
    splitting the whole window by position would put all of it in the hold-out part.
    """
    known = [i for i, v in enumerate(series.temp) if v is not None]
    if not known:
        return None
    series = _slice(series, known[0], known[-1] + 1)
    n = len(series.temp)
    split = int(n * (1 - holdout_fraction)) if holdout_fraction else n
    train = _slice(series, 0, split)
    active = active_features(train)

    best: FitResult | None = None
    taus = BOILER_TAU_GRID if "boiler" in active else (0,)
    for tau in taus:
        result = fit(make_rows(train, tau), active, span=span, tau_boiler_steps=tau)
        if result is not None and (best is None or result.sse < best.sse):
            best = result
    if best is None:
        return None

    chosen = _select_features(train, best.tau_boiler_steps, active, span, horizon) if select else tuple(active)
    tau = best.tau_boiler_steps if "boiler" in chosen else 0
    rows = make_rows(series, tau)  # one pass, so the lag filter is continuous across the split
    final = fit(rows[:split], chosen, span=span, tau_boiler_steps=tau) or best

    holdout = None
    if holdout_fraction and n - split > horizon:
        holdout = evaluate(final, rows[split:], horizon)
    dropped = tuple(f for f in active if f not in final.active)
    return RoomFit(final, split, holdout, dropped, exposure_hours(series))
