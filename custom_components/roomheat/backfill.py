"""Fill the history store from Home Assistant's recorder, so it never starts empty.

For every recorded signal this fills the gap between the newest stored row (or
`lookback_days` ago, on a first run) and now:
  1. hourly long-term statistics  (reach back furthest; sensors only)
  2. recorder state history       (5-minute grid; replaces 1 where both exist)
Live rows always win over both (see history_store). Running it again is harmless.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from functools import partial

from homeassistant.components.recorder.history import get_significant_states
from homeassistant.components.recorder.statistics import statistics_during_period
from homeassistant.core import HomeAssistant
from homeassistant.helpers.recorder import get_instance

from .history_store import SRC_STATES, SRC_STATISTICS, HistoryStore
from .sampling import SAMPLE_SECONDS, resample_hold
from .signals import POWER, TEMPERATURE, Signal
from .units import signal_value

_LOGGER = logging.getLogger(__name__)

STATE_CHUNK = timedelta(days=30)  # bounds memory when the recorder keeps a lot
STAT_UNITS = {"temperature": "°C", "power": "W"}


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


async def async_backfill(
    hass: HomeAssistant,
    store: HistoryStore,
    signals: Sequence[Signal],
    *,
    now_ts: float,
    lookback_days: int,
) -> dict[str, int]:
    """Backfill all signals. Returns rows written per source name."""
    written = {"statistics": 0, "states": 0}
    if not signals:
        return written

    instance = get_instance(hass)
    if not await instance.async_db_ready:
        _LOGGER.warning("Recorder database is not ready; skipping history backfill")
        return written

    temp_unit = hass.config.units.temperature_unit
    end = int(now_ts) // SAMPLE_SECONDS * SAMPLE_SECONDS
    floor = end - lookback_days * 86400
    last = await hass.async_add_executor_job(
        store.last_timestamps, [s.series for s in signals]
    )
    # rows are needed strictly after what is already stored
    starts = {s.series: max(last.get(s.series, floor), floor) for s in signals}
    earliest = min(starts.values())
    if earliest >= end:
        return written

    # 1. hourly statistics
    stat_signals = [s for s in signals if s.kind in (TEMPERATURE, POWER)]
    if stat_signals:
        try:
            stats = await instance.async_add_executor_job(
                partial(
                    statistics_during_period,
                    hass,
                    _dt(earliest),
                    _dt(end),
                    {s.entity_id for s in stat_signals},
                    "hour",
                    STAT_UNITS,
                    {"mean"},
                )
            )
        except Exception:  # noqa: BLE001 - backfill is best effort
            _LOGGER.exception("Statistics backfill failed")
            stats = {}
        rows = [
            (s.series, int(row["start"]), row["mean"], SRC_STATISTICS)
            for s in stat_signals
            for row in stats.get(s.entity_id, [])
            if row.get("mean") is not None and int(row["start"]) > starts[s.series]
        ]
        if rows:
            await hass.async_add_executor_job(store.write, rows)
            written["statistics"] = len(rows)

    # 2. state history, in chunks
    entity_ids = sorted({s.entity_id for s in signals})
    chunk_start = earliest
    while chunk_start < end:
        chunk_end = min(chunk_start + int(STATE_CHUNK.total_seconds()), end)
        try:
            history = await instance.async_add_executor_job(
                partial(
                    get_significant_states,
                    hass,
                    _dt(chunk_start),
                    _dt(chunk_end),
                    entity_ids,
                    None,  # filters
                    True,  # include_start_time_state: needed for the hold at the chunk edge
                    False,  # significant_changes_only
                    False,  # minimal_response
                    False,  # no_attributes: units and hvac_action live there
                )
            )
        except Exception:  # noqa: BLE001 - backfill is best effort
            _LOGGER.exception("State backfill failed for chunk starting %s", chunk_start)
            history = {}

        rows = []
        for s in signals:
            points = sorted(
                (st.last_updated_timestamp, signal_value(s.kind, st, temp_unit))
                for st in history.get(s.entity_id, [])
            )
            if not points:
                continue
            for ts, value in resample_hold(
                points, max(chunk_start, starts[s.series] + 1), chunk_end
            ):
                rows.append((s.series, ts, value, SRC_STATES))
        if rows:
            await hass.async_add_executor_job(store.write, rows)
            written["states"] += len(rows)
        chunk_start = chunk_end

    return written
