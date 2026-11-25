"""Pure sampling helpers (no Home Assistant imports, unit-testable)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

SAMPLE_SECONDS = 300  # one stored row per series every 5 minutes
MAX_HOLD_SECONDS = 6 * 3600  # never carry a value across a longer silence


class BucketAccumulator:
    """Averages the per-minute probes into 5-minute buckets.

    Averaging (rather than keeping the last probe) keeps short on/off cycles of
    the boiler or AC visible as a fraction between 0 and 1.
    A row's timestamp is the *start* of its bucket.
    """

    def __init__(self, step: int = SAMPLE_SECONDS) -> None:
        self._step = step
        self._bucket: int | None = None
        self._acc: dict[str, list[float]] = {}  # series -> [sum, count]

    def add(
        self, ts: float, values: Mapping[str, float | None]
    ) -> list[tuple[str, int, float]]:
        """Add one probe. Returns the finished rows when it starts a new bucket."""
        bucket = int(ts // self._step) * self._step
        done: list[tuple[str, int, float]] = []
        if self._bucket is not None and bucket != self._bucket:
            done = self.flush()
        self._bucket = bucket
        for series, value in values.items():
            if value is None:
                continue
            acc = self._acc.setdefault(series, [0.0, 0])
            acc[0] += value
            acc[1] += 1
        return done

    def flush(self) -> list[tuple[str, int, float]]:
        """Return and clear the rows of the current (possibly partial) bucket."""
        rows: list[tuple[str, int, float]] = []
        if self._bucket is not None:
            rows = [
                (series, self._bucket, total / n)
                for series, (total, n) in self._acc.items()
                if n
            ]
        self._acc = {}
        self._bucket = None
        return rows


def resample_hold(
    points: Sequence[tuple[float, float | None]],
    start: int,
    end: int,
    step: int = SAMPLE_SECONDS,
    max_hold: int = MAX_HOLD_SECONDS,
) -> list[tuple[int, float]]:
    """Zero-order-hold resample of state changes onto the step grid.

    `points` are (timestamp, value) state changes sorted by time; a None value
    means "unavailable". Returns (grid_ts, value) for grid times in [start, end)
    that have a known value no older than `max_hold`.
    """
    out: list[tuple[int, float]] = []
    i, n = 0, len(points)
    last: tuple[float, float | None] | None = None
    t = -(-start // step) * step  # first grid time >= start
    while t < end:
        while i < n and points[i][0] <= t:
            last = points[i]
            i += 1
        if last is not None and last[1] is not None and t - last[0] <= max_hold:
            out.append((t, last[1]))
        t += step
    return out
