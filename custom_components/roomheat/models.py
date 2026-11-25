"""Keeps each room's fitted model and its latest prediction. Pure Python.

Refitting is slow, so it runs now and then; predicting is cheap, so it runs after
every 5-minute bucket. Both read the history store in an executor.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .dataset import RoomFit, fit_room
from .history_store import HistoryStore
from .learn import (
    MIN_USABLE_STEPS,
    STEP,
    HubSpec,
    RoomSpec,
    build_dataset,
    needed_series,
    predict_ahead,
    status,
    usable_steps,
)
from .thermal import FitResult

Run = Callable[..., Awaitable[Any]]  # hass.async_add_executor_job


@dataclass(slots=True)
class RoomModelState:
    status: str = "pending"  # pending until the first refit has run
    fit: RoomFit | None = None
    usable_hours: float = 0.0
    trained_at: float | None = None
    predicted: float | None = None
    predicted_from: int | None = None  # timestamp of the reading it starts from
    predicted_for: int | None = None  # timestamp it predicts
    last_error: float | None = None  # actual - predicted, newest prediction that came due
    pending: list[tuple[int, float]] = field(default_factory=list)  # (due ts, predicted)


class ModelManager:
    def __init__(
        self,
        store: HistoryStore,
        run: Run,
        lookback_days: int = 60,
        horizon_steps: int = 12,
    ) -> None:
        self._store = store
        self._run = run
        self._lookback = lookback_days * 86400
        self.horizon = horizon_steps  # 12 steps = 1 hour
        self.states: dict[str, RoomModelState] = {}

    # -- refit -----------------------------------------------------------

    def _refit_sync(self, hub: HubSpec, rooms: Sequence[RoomSpec], start: int, end: int):
        rows = self._store.read_many(needed_series(hub, rooms), start, end)
        data = build_dataset(rows, hub, rooms, start, end)
        out = {}
        for room in rooms:
            series = data.rooms[room.key]
            usable = usable_steps(series)
            fit = fit_room(series, horizon=self.horizon) if usable >= MIN_USABLE_STEPS else None
            out[room.key] = (fit, usable)
        return out

    async def refit(self, hub: HubSpec, rooms: Sequence[RoomSpec], now_ts: float) -> None:
        end = int(now_ts) // STEP * STEP
        results = await self._run(self._refit_sync, hub, rooms, end - self._lookback, end)
        for key, (fit, usable) in results.items():
            state = self.states.setdefault(key, RoomModelState())
            state.fit, state.usable_hours = fit, usable * STEP / 3600
            state.trained_at, state.status = now_ts, status(fit, usable)
        for gone in self.states.keys() - results.keys():
            del self.states[gone]

    # -- predict ---------------------------------------------------------

    def _predict_sync(
        self, hub: HubSpec, rooms: Sequence[RoomSpec], fits: dict[str, FitResult], start: int, end: int
    ):
        rows = self._store.read_many(needed_series(hub, rooms), start, end)
        data = build_dataset(rows, hub, rooms, start, end)
        out = {}
        for room in rooms:
            if room.key not in fits:
                continue
            series = data.rooms[room.key]
            actual = {start + i * STEP: t for i, t in enumerate(series.temp) if t is not None}
            out[room.key] = (predict_ahead(fits[room.key], series, self.horizon), actual)
        return out

    async def predict(self, hub: HubSpec, rooms: Sequence[RoomSpec], now_ts: float) -> None:
        fits = {k: s.fit.result for k, s in self.states.items() if s.fit is not None}
        if not fits:
            return
        end = int(now_ts) // STEP * STEP + STEP
        start = end - (self.horizon + 96) * STEP  # enough lag history for the radiator filter
        results = await self._run(self._predict_sync, hub, rooms, fits, start, end)
        for key, (prediction, actual) in results.items():
            state = self.states[key]
            for due, pred in list(state.pending):  # score predictions that have come due
                if due in actual:
                    state.last_error = actual[due] - pred
                    state.pending.remove((due, pred))
                elif due < start:
                    state.pending.remove((due, pred))
            if prediction is None:
                state.predicted = state.predicted_from = state.predicted_for = None
                continue
            index, _, pred = prediction
            base = start + index * STEP
            state.predicted, state.predicted_from = pred, base
            state.predicted_for = base + self.horizon * STEP
            if (state.predicted_for, pred) not in state.pending:
                state.pending.append((state.predicted_for, pred))
