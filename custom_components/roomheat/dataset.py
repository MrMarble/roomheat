"""Turn per-signal time series into model rows, and search the boiler lag.

Pure Python. All series are aligned on one 5-minute grid; None marks gaps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .thermal import (
    FEATURES,
    Evaluation,
    FitResult,
    StepRow,
    evaluate,
    fit,
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
        StepRow(t, o, n, b, h, c)
        for t, o, n, b, h, c in zip(
            series.temp,
            series.outdoor,
            series.neighbours,
            boiler,
            series.ac_heat,
            series.ac_cool,
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
    )


@dataclass(frozen=True, slots=True)
class RoomFit:
    result: FitResult
    train_steps: int
    holdout: Evaluation | None  # None if there was no hold-out or it had no usable windows


def fit_room(
    series: RoomSeries,
    *,
    span: int = 6,
    horizon: int = 12,
    holdout_fraction: float = 0.25,
) -> RoomFit | None:
    """Fit one room: pick the boiler lag on the training part, score on the rest.

    `horizon` is the open-loop prediction length (12 steps = 1 hour).
    """
    n = len(series.temp)
    split = int(n * (1 - holdout_fraction)) if holdout_fraction else n
    train, test = _slice(series, 0, split), _slice(series, split, n)
    active = active_features(series)

    best: FitResult | None = None
    taus = BOILER_TAU_GRID if "boiler" in active else (0,)
    for tau in taus:
        result = fit(make_rows(train, tau), active, span=span, tau_boiler_steps=tau)
        if result is not None and (best is None or result.sse < best.sse):
            best = result
    if best is None:
        return None

    holdout = None
    if holdout_fraction and test.temp:
        holdout = evaluate(best, make_rows(test, best.tau_boiler_steps), horizon)
    return RoomFit(best, split, holdout)
