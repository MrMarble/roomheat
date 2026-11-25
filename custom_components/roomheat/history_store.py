"""Persistent observation log (SQLite, stdlib only).

One row per (series, 5-minute bucket). A series is an entity id, or
`<climate entity>:heating` / `:cooling` for a climate entity's hvac_action.

`src` ranks how trustworthy a row is; a lower rank is never overwritten by a
higher one, so live data always beats backfilled data:
    0 live probes   1 recorder states   2 recorder hourly statistics
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Sequence

SRC_LIVE = 0
SRC_STATES = 1
SRC_STATISTICS = 2

_UPSERT = """
INSERT INTO samples (series, ts, value, src) VALUES (?, ?, ?, ?)
ON CONFLICT (series, ts) DO UPDATE SET value = excluded.value, src = excluded.src
WHERE excluded.src <= samples.src
"""


class HistoryStore:
    """Thread-safe; call every method from an executor, never the event loop."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def setup(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS samples (
                    series TEXT NOT NULL,
                    ts     INTEGER NOT NULL,
                    value  REAL NOT NULL,
                    src    INTEGER NOT NULL,
                    PRIMARY KEY (series, ts)
                ) WITHOUT ROWID"""
            )

    def write(self, rows: Iterable[tuple[str, int, float, int]]) -> None:
        """Upsert (series, ts, value, src) rows."""
        with self._lock, self._connect() as conn:
            conn.executemany(_UPSERT, list(rows))

    def last_timestamps(self, series: Sequence[str]) -> dict[str, int]:
        """Newest stored timestamp per series (series with no rows are absent)."""
        with self._lock, self._connect() as conn:
            found = {
                s: ts
                for s in series
                if (ts := conn.execute(
                    "SELECT MAX(ts) FROM samples WHERE series = ?", (s,)
                ).fetchone()[0]) is not None
            }
        return found

    def read_many(
        self, series: Sequence[str], start: int, end: int, max_src: int = SRC_STATES
    ) -> dict[str, dict[int, float]]:
        """{series: {ts: value}} for rows in [start, end) no coarser than `max_src`.

        The default leaves out hourly statistics, which are too coarse to fit on.
        """
        out: dict[str, dict[int, float]] = {s: {} for s in series}
        if not series:
            return out
        marks = ",".join("?" * len(series))
        with self._lock, self._connect() as conn:
            for name, ts, value in conn.execute(
                f"SELECT series, ts, value FROM samples WHERE src <= ? AND ts >= ? AND ts < ? "
                f"AND series IN ({marks})",
                (max_src, start, end, *series),
            ):
                out[name][ts] = value
        return out

    def read(self, series: str, start: int, end: int) -> list[tuple[int, float, int]]:
        """(ts, value, src) rows of one series in [start, end)."""
        with self._lock, self._connect() as conn:
            return conn.execute(
                "SELECT ts, value, src FROM samples WHERE series = ? AND ts >= ? AND ts < ? ORDER BY ts",
                (series, start, end),
            ).fetchall()

    def summary(self) -> dict[str, dict[str, int]]:
        """Per series: rows, first and last timestamp, rows per source (diagnostics)."""
        with self._lock, self._connect() as conn:
            out: dict[str, dict[str, int]] = {
                s: {"rows": n, "first": a, "last": b}
                for s, n, a, b in conn.execute(
                    "SELECT series, COUNT(*), MIN(ts), MAX(ts) FROM samples GROUP BY series"
                )
            }
            for series, src, n in conn.execute(
                "SELECT series, src, COUNT(*) FROM samples GROUP BY series, src"
            ):
                out[series][("live", "states", "statistics")[src]] = n
        return out

    def counts(self) -> dict[str, int]:
        """Rows per series (diagnostics)."""
        with self._lock, self._connect() as conn:
            return dict(conn.execute("SELECT series, COUNT(*) FROM samples GROUP BY series"))
