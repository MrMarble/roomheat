"""A synthetic room with known physics, to check that the fit recovers it."""

from __future__ import annotations

import math
import random

from _pure import load

D = load("dataset")

TRUE = {
    "outdoor": 0.004,      # 5 min step: time constant ~ 21 h
    "neighbours": 0.006,
    "boiler": 0.05,        # +0.6 C/h with the (filtered) boiler fully on
    "ac_heat": 0.03,       # per kW
    "ac_cool": 0.04,       # per kW
    "bias_up": 0.0005,
}
TRUE_TAU_STEPS = 6          # radiator lag: 30 min
STEPS_PER_DAY = 288


def make(days: int = 21, seed: int = 1, quantise: float = 0.1, sensor_noise: float = 0.03, gate: bool = False):
    """Returns (RoomSeries as the sensors would report it, true temperature list).

    With gate=True the room has a radiator valve that is open about 55% of the time,
    in blocks of roughly an hour, and only boiler time with the valve open heats it.
    """
    rnd = random.Random(seed)
    n = days * STEPS_PER_DAY
    hour = lambda k: (k % STEPS_PER_DAY) / 12
    outdoor = [8 + 5 * math.sin(2 * math.pi * (hour(k) - 9) / 24) + 2 * math.sin(k / 700) for k in range(n)]
    neighbours = [19 + 1.2 * math.sin(2 * math.pi * (hour(k) - 12) / 24) + 0.8 * math.sin(k / 400) for k in range(n)]

    boiler = []
    on = False
    for k in range(n):
        in_window = 9 <= hour(k) < 22 and (k // STEPS_PER_DAY) % 7 < 6  # off one day a week
        if in_window:
            if rnd.random() < 0.08:  # thermostat cycles every ~hour
                on = not on
        else:
            on = False
        boiler.append(1.0 if on else 0.0)

    ac_heat, ac_cool = [], []
    for k in range(n):
        h = hour(k)
        evening = (k // STEPS_PER_DAY) % 3 == 0 and 22 <= h < 24
        morning = (k // STEPS_PER_DAY) % 3 == 1 and 6 <= h < 8
        ac_heat.append(1.2 if evening else 0.0)
        ac_cool.append(1.5 if morning else 0.0)

    valve = None
    if gate:
        valve, is_open = [], True
        for _ in range(n):
            if rnd.random() < 0.08:
                is_open = rnd.random() < 0.55
            valve.append(1.0 if is_open else 0.0)
    delivered = boiler if valve is None else [b * v for b, v in zip(boiler, valve)]
    filt = D.lowpass(delivered, TRUE_TAU_STEPS)
    t = 18.0
    truth = []
    for k in range(n):
        truth.append(t)
        t += (
            TRUE["outdoor"] * (outdoor[k] - t)
            + TRUE["neighbours"] * (neighbours[k] - t)
            + TRUE["boiler"] * filt[k]
            + TRUE["ac_heat"] * ac_heat[k]
            - TRUE["ac_cool"] * ac_cool[k]
            + TRUE["bias_up"]
        )

    q = lambda v: round((v + rnd.gauss(0, sensor_noise)) / quantise) * quantise if quantise else v
    series = D.RoomSeries(
        temp=[q(v) for v in truth],
        outdoor=[q(v) for v in outdoor],
        neighbours=[q(v) for v in neighbours],
        boiler=boiler,
        ac_heat=ac_heat,
        ac_cool=ac_cool,
        valve=valve,
    )
    return series, truth
