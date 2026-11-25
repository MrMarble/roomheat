"""Thermal model of one room. Pure Python, no Home Assistant imports.

Over one 5-minute step the room temperature changes by

    dT = a_out · (T_outdoor − T)          heat exchange with the outside
       + a_nb  · (T_neighbours − T)       heat exchange with adjacent rooms
       + b_boiler · u_boiler              central heating (filtered, see below)
       + b_ac_heat · u_ac_heat            AC heating  (kW × heating flag)
       − b_ac_cool · u_ac_cool            AC cooling  (kW × cooling flag)
       + b_internal · u_internal          measured appliances in the room (kW), if any
       + c                                everything unmodelled (sun, people, other rooms…)

This is linear in the unknown coefficients, so it is learned by non-negative
least squares. All coefficients are physically signed (a, b ≥ 0), which stops the
fit from inventing, say, a boiler that cools the room. The unmodelled term c can
have either sign, so it is two non-negative features, `bias_up` and `bias_down`.

Radiators keep giving heat after the boiler stops, so the boiler input is passed
through a first-order low-pass filter. Its time constant is the one non-linear
parameter and is found by a small grid search (`dataset.py`, `fit_room`).

The statistics are sums (XᵀX, Xᵀy, yᵀy), so the same code serves both an offline
fit over the whole store and, later, an online learner that adds one step at a
time and decays old data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .sampling import SAMPLE_SECONDS

STEP_HOURS = SAMPLE_SECONDS / 3600

FEATURES = (
    "outdoor",
    "neighbours",
    "boiler",
    "ac_heat",
    "ac_cool",
    "internal",
    "bias_up",
    "bias_down",
)


@dataclass(slots=True)
class StepRow:
    """Everything known at one 5-minute step. None means "no data"."""

    t: float | None  # room temperature
    outdoor: float | None = None  # outdoor temperature
    neighbours: float | None = None  # mean temperature of adjacent rooms
    boiler: float | None = None  # filtered boiler-on fraction, 0..1
    ac_heat: float | None = None  # AC heating input (kW x flag)
    ac_cool: float | None = None  # AC cooling input (kW x flag), positive number
    internal: float | None = None  # measured appliance heat in the room, kW


def feature_values(row: StepRow, names: Sequence[str], t: float | None = None) -> list[float] | None:
    """Feature vector for a row, or None if any input it needs is missing.

    `t` overrides the row's temperature (used when simulating).
    """
    temp = row.t if t is None else t
    if temp is None:
        return None
    out: list[float] = []
    for name in names:
        if name == "outdoor":
            v = None if row.outdoor is None else row.outdoor - temp
        elif name == "neighbours":
            v = None if row.neighbours is None else row.neighbours - temp
        elif name == "boiler":
            v = row.boiler
        elif name == "ac_heat":
            v = row.ac_heat
        elif name == "ac_cool":
            v = None if row.ac_cool is None else -row.ac_cool
        elif name == "internal":
            v = row.internal
        elif name == "bias_up":
            v = 1.0
        elif name == "bias_down":
            v = -1.0
        else:
            raise ValueError(f"unknown feature {name!r}")
        if v is None:
            return None
        out.append(v)
    return out


# -- least squares ---------------------------------------------------------


def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve a·x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            if f:
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


class NormalEquations:
    """Sufficient statistics of a least-squares problem, optionally forgetting."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.xtx = [[0.0] * n for _ in range(n)]
        self.xty = [0.0] * n
        self.yty = 0.0
        self.weight = 0.0  # effective number of samples

    def add(self, x: Sequence[float], y: float, w: float = 1.0) -> None:
        for i in range(self.n):
            wxi = w * x[i]
            row = self.xtx[i]
            for j in range(self.n):
                row[j] += wxi * x[j]
            self.xty[i] += wxi * y
        self.yty += w * y * y
        self.weight += w

    def subset(self, idx: Sequence[int]) -> "NormalEquations":
        """The same problem restricted to some of the features."""
        sub = NormalEquations(len(idx))
        sub.xtx = [[self.xtx[i][j] for j in idx] for i in idx]
        sub.xty = [self.xty[i] for i in idx]
        sub.yty, sub.weight = self.yty, self.weight
        return sub

    def decay(self, factor: float) -> None:
        """Multiply all statistics by `factor` (< 1 forgets old data)."""
        for row in self.xtx:
            for j in range(self.n):
                row[j] *= factor
        self.xty = [v * factor for v in self.xty]
        self.yty *= factor
        self.weight *= factor

    def sse(self, beta: Sequence[float]) -> float:
        quad = sum(
            beta[i] * self.xtx[i][j] * beta[j]
            for i in range(self.n)
            for j in range(self.n)
        )
        return self.yty - 2 * sum(b * v for b, v in zip(beta, self.xty)) + quad

    def solve_nnls(self, ridge: float = 1e-6) -> list[float]:
        """Non-negative least squares by exact enumeration of active sets.

        The optimum is the unconstrained solution on its own support, so trying
        every support and keeping the best feasible one is exact. Fine for the
        handful of features used here (2^7 subsets).
        """
        n = self.n
        mean_diag = sum(self.xtx[i][i] for i in range(n)) / n if n else 0.0
        lam = ridge * mean_diag + 1e-12
        best = [0.0] * n
        best_sse = self.yty
        for mask in range(1, 1 << n):
            idx = [i for i in range(n) if mask >> i & 1]
            a = [[self.xtx[i][j] + (lam if i == j else 0.0) for j in idx] for i in idx]
            sol = _solve(a, [self.xty[i] for i in idx])
            if sol is None or any(v < 0 for v in sol):
                continue
            beta = [0.0] * n
            for i, v in zip(idx, sol):
                beta[i] = v
            sse = self.sse(beta)
            if sse < best_sse:
                best, best_sse = beta, sse
        return best


# -- fitting ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FitResult:
    coef: dict[str, float]  # every feature; inactive ones are 0.0
    active: tuple[str, ...]
    tau_boiler_steps: int
    span: int
    n: int  # training windows
    sse: float  # of the fit
    sse_persistence: float  # of predicting "no change"


def accumulate(rows: Sequence[StepRow], names: Sequence[str], span: int = 6) -> NormalEquations | None:
    """Least-squares statistics over every complete `span`-step window of `rows`.

    The target is the temperature change over `span` steps and the regressors are
    the feature sums over those steps. Using 30 minutes instead of 5 keeps real
    sensors' 0.1 °C resolution from drowning the signal. None if no window is complete.
    """
    names = tuple(names)
    feats = [feature_values(r, names) for r in rows]
    ne = NormalEquations(len(names))
    for i in range(len(rows) - span):
        t0, t1 = rows[i].t, rows[i + span].t
        if t0 is None or t1 is None:
            continue
        window = feats[i : i + span]
        if any(f is None for f in window):
            continue
        x = [sum(f[k] for f in window) for k in range(len(names))]  # type: ignore[index]
        ne.add(x, t1 - t0)
    return ne if ne.weight else None


def result_from(
    ne: NormalEquations, names: Sequence[str], span: int, tau_boiler_steps: int, ridge: float = 1e-6
) -> FitResult:
    beta = ne.solve_nnls(ridge)
    coef = {name: 0.0 for name in FEATURES}
    coef.update(zip(names, beta))
    return FitResult(
        coef=coef,
        active=tuple(names),
        tau_boiler_steps=tau_boiler_steps,
        span=span,
        n=int(ne.weight),
        sse=ne.sse(beta),
        sse_persistence=ne.yty,
    )


def fit(
    rows: Sequence[StepRow],
    active: Sequence[str],
    span: int = 6,
    tau_boiler_steps: int = 0,
    ridge: float = 1e-6,
) -> FitResult | None:
    """Fit the coefficients on `rows` (consecutive 5-minute steps)."""
    ne = accumulate(rows, active, span)
    return None if ne is None else result_from(ne, active, span, tau_boiler_steps, ridge)


def step_delta(result: FitResult, row: StepRow, t: float | None = None) -> float | None:
    """Predicted temperature change over one step."""
    x = feature_values(row, result.active, t)
    if x is None:
        return None
    return sum(result.coef[name] * v for name, v in zip(result.active, x))


def simulate(result: FitResult, rows: Sequence[StepRow], start: int, steps: int) -> float | None:
    """Open-loop prediction of the temperature at `start + steps`.

    Uses the measured inputs (outdoor, neighbours, boiler, AC) but only the
    temperature at `start`, which is how the planner will use the model.
    """
    t = rows[start].t
    if t is None:
        return None
    for j in range(steps):
        d = step_delta(result, rows[start + j], t)
        if d is None:
            return None
        t += d
    return t


# An input counts as "changing" inside a window when it moves by more than this
# (boiler as a 0..1 fraction, the others in kW). Those windows are where a model
# earns its keep; in a flat stretch "the temperature stays put" is hard to beat.
CHANGE_THRESHOLD = {"boiler": 0.3, "ac_heat": 0.3, "ac_cool": 0.3, "internal": 0.2}


@dataclass(frozen=True, slots=True)
class Evaluation:
    n: int
    rmse_model: float
    rmse_persistence: float
    n_active: int = 0  # windows in which heating, AC or appliances change
    rmse_model_active: float | None = None
    rmse_persistence_active: float | None = None


def _inputs_change(rows: Sequence[StepRow], i: int, horizon: int, names: Sequence[str]) -> bool:
    for name in names:
        vals = [v for v in (getattr(rows[i + j], name) for j in range(horizon)) if v is not None]
        if vals and max(vals) - min(vals) > CHANGE_THRESHOLD[name]:
            return True
    return False


def evaluate(result: FitResult, rows: Sequence[StepRow], horizon: int, stride: int = 3) -> Evaluation | None:
    """Open-loop error `horizon` steps ahead, against "the temperature stays put".

    Also scored separately on the windows where heating, AC or appliances change.
    """
    drivers = [n for n in result.active if n in CHANGE_THRESHOLD]
    se_model = se_pers = a_model = a_pers = 0.0
    n = n_active = 0
    for i in range(0, len(rows) - horizon, stride):
        t0, actual = rows[i].t, rows[i + horizon].t
        if t0 is None or actual is None:
            continue
        pred = simulate(result, rows, i, horizon)
        if pred is None:
            continue
        m, p = (pred - actual) ** 2, (t0 - actual) ** 2
        se_model += m
        se_pers += p
        n += 1
        if _inputs_change(rows, i, horizon, drivers):
            a_model += m
            a_pers += p
            n_active += 1
    if n == 0:
        return None
    return Evaluation(
        n,
        math.sqrt(se_model / n),
        math.sqrt(se_pers / n),
        n_active,
        math.sqrt(a_model / n_active) if n_active else None,
        math.sqrt(a_pers / n_active) if n_active else None,
    )
