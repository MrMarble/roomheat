"""Run with:  python3 tests/test_model.py   (no pytest, no Home Assistant needed)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synthetic as S  # noqa: E402
from _pure import load  # noqa: E402

T, D, M = load("thermal"), load("dataset"), load("model")
G, H = load("signals"), load("history_store")
L, MM = load("learn"), load("models")


def close(got, want, rel):
    assert abs(got - want) <= rel * abs(want), f"{got} vs {want} (±{rel:.0%})"


def test_lowpass():
    assert D.lowpass([0, 1, 1, 0], 0) == [0, 1, 1, 0]
    out = D.lowpass([1.0] + [0.0] * 40, 6)
    assert out[0] == 1.0 and out[1] < out[0] and out[-1] < 0.01  # decays
    gap = D.lowpass([1.0, None, 0.0], 1)
    assert gap == [1.0, None, 0.5]  # a gap yields None and leaves the state alone


def test_nnls_respects_signs_and_is_exact_when_truth_is_feasible():
    rnd = random.Random(0)
    # truth feasible -> recovered
    ne = T.NormalEquations(3)
    for _ in range(500):
        x = [rnd.gauss(0, 1) for _ in range(3)]
        ne.add(x, 2.0 * x[0] + 0.5 * x[2])
    b = ne.solve_nnls()
    assert abs(b[0] - 2.0) < 1e-4 and abs(b[1]) < 1e-4 and abs(b[2] - 0.5) < 1e-4, b
    # truth needs a negative coefficient -> clipped to zero, never negative
    ne = T.NormalEquations(2)
    for _ in range(500):
        x0 = rnd.gauss(0, 1)
        x = [x0, x0 + 0.3 * rnd.gauss(0, 1)]
        ne.add(x, 2.0 * x[0] - 1.0 * x[1])
    b = ne.solve_nnls()
    assert all(v >= 0 for v in b), b


def test_nnls_online_decay_forgets_old_regime():
    ne = T.NormalEquations(1)
    for _ in range(300):
        ne.add([1.0], 5.0)
    ne.decay(0.001)
    for _ in range(300):
        ne.add([1.0], 1.0)
    assert abs(ne.solve_nnls()[0] - 1.0) < 0.05


def test_fit_recovers_known_physics_exactly_without_noise():
    series, _ = S.make(quantise=0, sensor_noise=0)
    rf = D.fit_room(series, span=1)
    assert rf.result.tau_boiler_steps == S.TRUE_TAU_STEPS
    for name in ("outdoor", "neighbours", "boiler", "ac_heat", "ac_cool"):
        close(rf.result.coef[name], S.TRUE[name], 0.01)
    assert rf.holdout.rmse_model < 0.01


def test_fit_is_robust_to_sensor_resolution_and_noise():
    series, _ = S.make()  # 0.1 C steps + noise
    rf = D.fit_room(series)
    assert rf.result.tau_boiler_steps in (3, 6, 12)  # the grid neighbours of the truth
    for name in ("outdoor", "neighbours", "boiler", "ac_heat", "ac_cool"):
        close(rf.result.coef[name], S.TRUE[name], 0.10)
    # the point of the model: far better than "temperature stays put" an hour ahead
    assert rf.holdout.rmse_model < 0.3 * rf.holdout.rmse_persistence


def test_valve_state_explains_when_the_radiator_really_heats():
    series, _ = S.make(gate=True)
    with_gate = D.fit_room(series)
    close(with_gate.result.coef["boiler"], S.TRUE["boiler"], 0.10)
    assert with_gate.holdout.rmse_model < 0.3 * with_gate.holdout.rmse_persistence

    series.valve = None  # same data, but the valve state is unknown
    blind = D.fit_room(series)
    # without the gate the boiler looks weaker and the forecast is clearly worse
    assert blind.result.coef["boiler"] < 0.8 * S.TRUE["boiler"]
    assert blind.holdout.rmse_model > 1.5 * with_gate.holdout.rmse_model


def test_fit_survives_missing_data():
    series, _ = S.make(seed=3)
    rnd = random.Random(9)
    for lst in (series.temp, series.outdoor, series.neighbours):
        for i in range(len(lst)):
            if rnd.random() < 0.1:
                lst[i] = None
    rf = D.fit_room(series)
    close(rf.result.coef["boiler"], S.TRUE["boiler"], 0.15)
    assert rf.holdout is not None


def test_unidentifiable_inputs_are_dropped():
    series, _ = S.make()
    series.ac_heat = [None] * len(series.temp)         # AC never reported
    series.ac_cool = [0.0] * len(series.temp)          # never cooled
    series.neighbours = [None] * len(series.temp)      # room with no neighbours
    active = D.active_features(series)
    assert "ac_heat" not in active and "ac_cool" not in active and "neighbours" not in active
    assert {"outdoor", "boiler", "bias_up"} <= set(active)
    assert D.fit_room(series) is not None


def test_signals_cover_every_entity_once_with_stable_names():
    sig = G.build_signals(
        room_sensors=["sensor.a", "sensor.a"], outdoor="sensor.o", thermostat="climate.t",
        ac="climate.ac", ac_power="sensor.p", valves=["climate.v1", "climate.v2"], weather="weather.home",
    )
    names = [s.series for s in sig]
    assert len(names) == len(set(names))
    assert names.count("sensor.a") == 1
    for want in ("climate.t:heating", "climate.t:setpoint", "climate.t:active",
                 "climate.ac:cooling", "climate.v2:heating", "climate.v2:setpoint",
                 "weather.home:weather_temperature", "weather.home:cloud_coverage", "weather.home:wind_speed"):
        assert want in names, want
    assert "climate.v1:cooling" not in names  # valves only heat
    assert {s.entity_id for s in sig if s.series == "weather.home:cloud_coverage"} == {"weather.home"}


def test_store_summary_counts_rows_by_source():
    import os, tempfile
    st = H.HistoryStore(os.path.join(tempfile.mkdtemp(), "s.db")); st.setup()
    st.write([("a", 300, 1.0, 0), ("a", 600, 1.0, 1), ("a", 900, 1.0, 1), ("b", 300, 2.0, 2)])
    info = st.summary()
    assert info["a"] == {"rows": 3, "first": 300, "last": 900, "live": 1, "states": 2}, info["a"]
    assert info["b"]["statistics"] == 1 and info["b"]["rows"] == 1


TS0 = 1_700_000_100  # a multiple of 300


def _store_with_two_rooms(days=21, gate=True, night_calls=False):
    """A real store on disk holding a synthetic 'sala' (heated) and 'vecina' (its neighbour)."""
    import os, tempfile
    series, _ = S.make(days=days, gate=gate)
    rows = []

    def put(name, vals):
        rows.extend((name, TS0 + 300 * i, float(v), H.SRC_LIVE) for i, v in enumerate(vals) if v is not None)

    put("sensor.sala", series.temp); put("sensor.vecina", series.neighbours)
    asks = list(series.boiler)
    if night_calls:  # the thermostat also asks for heat at night, when the building gives none
        rnd, on = __import__("random").Random(5), False
        for k in range(len(asks)):
            if not 9 <= (k % 288) / 12 < 22:
                on = (not on) if rnd.random() < 0.08 else on
                asks[k] = 1.0 if on else 0.0
    put("sensor.out", series.outdoor); put("climate.boiler:heating", asks)
    put("climate.valve:heating", series.valve or [1.0] * len(series.temp))
    put("sensor.ac_w", [(h + c) * 1000 for h, c in zip(series.ac_heat, series.ac_cool)])
    put("climate.ac:heating", [1.0 if h else 0.0 for h in series.ac_heat])
    put("climate.ac:cooling", [1.0 if c else 0.0 for c in series.ac_cool])
    store = H.HistoryStore(os.path.join(tempfile.mkdtemp(), "m.db")); store.setup(); store.write(rows)
    window = (lambda ts: 9 <= ((ts - TS0) // 300 % 288) / 12 < 22) if night_calls else None
    hub = L.HubSpec(outdoor="sensor.out", thermostat="climate.boiler", ac="climate.ac",
                    ac_power="sensor.ac_w", central_open=window)
    rooms = [
        L.RoomSpec("k_sala", "sala", "Sala", ("sensor.sala",), ("climate.valve",), ("vecina",)),
        L.RoomSpec("k_vecina", "vecina", "Vecina", ("sensor.vecina",), (), ("sala",)),
    ]
    return store, hub, rooms


async def _run(f, *a):
    return f(*a)


def test_manager_refits_a_room_from_the_store_and_describes_it():
    import asyncio
    store, hub, rooms = _store_with_two_rooms()
    mgr = MM.ModelManager(store, _run)
    asyncio.run(mgr.refit(hub, rooms, TS0 + 21 * 86400))
    st = mgr.states["k_sala"]
    assert st.status == "ready", (st.status, L.skill(st.fit))
    d = L.describe(st.fit.result)
    close(d["central_heating_gain_c_per_h"], S.TRUE["boiler"] * 12, 0.10)
    close(d["outdoor_time_constant_h"], (1 / 12) / S.TRUE["outdoor"], 0.15)
    assert d["radiator_lag_min"] in (15, 30, 60)
    assert st.usable_hours > 400


def test_a_short_history_at_the_end_of_a_long_window_still_fits():
    """Real case: 60 days are examined but only the last ~10 hold data."""
    import asyncio
    store, hub, rooms = _store_with_two_rooms(days=10)
    mgr = MM.ModelManager(store, _run)               # looks back 60 days by default
    asyncio.run(mgr.refit(hub, rooms, TS0 + 10 * 86400))
    st = mgr.states["k_sala"]
    assert st.fit is not None, "enough readings but no fit: the split ignored where the data is"
    assert st.status in ("ready", "poor"), st.status
    close(st.fit.result.coef["boiler"], S.TRUE["boiler"], 0.15)


def test_enough_readings_but_nothing_to_learn_from_is_reported_as_unfit():
    assert L.status(None, L.MIN_USABLE_STEPS + 1) == "unfit"
    assert L.status(None, 10) == "collecting"


def test_manager_says_collecting_when_there_is_too_little_data():
    import asyncio
    store, hub, rooms = _store_with_two_rooms(days=1)
    mgr = MM.ModelManager(store, _run)
    asyncio.run(mgr.refit(hub, rooms, TS0 + 86400))
    assert mgr.states["k_sala"].status == "collecting" and mgr.states["k_sala"].fit is None


def test_manager_predicts_an_hour_ahead_and_scores_it_when_it_comes_due():
    import asyncio
    store, hub, rooms = _store_with_two_rooms()
    mgr = MM.ModelManager(store, _run)
    now = TS0 + 18 * 86400
    asyncio.run(mgr.refit(hub, rooms, now))
    asyncio.run(mgr.predict(hub, rooms, now))
    st = mgr.states["k_sala"]
    assert st.predicted is not None and st.predicted_for - st.predicted_from == 3600
    assert st.last_error is None and len(st.pending) == 1          # nothing has come due yet
    asyncio.run(mgr.predict(hub, rooms, now + 3600))               # an hour later the first one is due
    assert st.last_error is not None and abs(st.last_error) < 0.6, st.last_error


def test_heat_only_counts_while_central_heating_is_available():
    import asyncio, dataclasses
    store, hub, rooms = _store_with_two_rooms(night_calls=True)
    now = TS0 + 21 * 86400
    mgr = MM.ModelManager(store, _run)
    asyncio.run(mgr.refit(hub, rooms, now))
    gated = mgr.states["k_sala"].fit.result.coef["boiler"]
    close(gated, S.TRUE["boiler"], 0.10)

    naive = MM.ModelManager(store, _run)  # same data, but taking the thermostat at its word
    asyncio.run(naive.refit(dataclasses.replace(hub, central_open=None), rooms, now))
    blind = naive.states["k_sala"].fit.result.coef["boiler"]
    assert blind < 0.85 * S.TRUE["boiler"], blind  # "heating" nights with no heat weaken the boiler


def test_stale_readings_give_no_prediction():
    import asyncio
    store, hub, rooms = _store_with_two_rooms()
    mgr = MM.ModelManager(store, _run)
    asyncio.run(mgr.refit(hub, rooms, TS0 + 18 * 86400))
    asyncio.run(mgr.predict(hub, rooms, TS0 + 30 * 86400))         # long after the last stored reading
    assert mgr.states["k_sala"].predicted is None


def test_appliance_power_is_learned_as_heat():
    series, _ = S.make(internal=True)
    rf = D.fit_room(series)
    assert "internal" in rf.result.active and "internal" not in rf.dropped
    close(rf.result.coef["internal"], S.TRUE["internal"], 0.15)
    close(L.describe(rf.result)["internal_heat_gain_c_per_h_per_kw"], S.TRUE["internal"] * 12, 0.15)
    # without the sensor the daily pattern is unexplained and the forecast is worse
    series.internal = None
    blind = D.fit_room(series)
    assert blind.holdout.rmse_model > 1.3 * rf.holdout.rmse_model


def test_a_room_whose_radiator_is_off_ignores_the_central_heating():
    import asyncio
    store, hub, rooms = _store_with_two_rooms()
    off = [L.RoomSpec(r.key, r.area_id, r.name, r.sensors, r.valves, r.adjacent_area_ids, radiator_in_use=False)
           if r.key == "k_sala" else r for r in rooms]
    mgr = MM.ModelManager(store, _run)
    asyncio.run(mgr.refit(hub, off, TS0 + 21 * 86400))
    fit = mgr.states["k_sala"].fit
    assert "boiler" not in fit.result.active and fit.exposure["boiler"] == 0.0


def test_a_feature_that_does_nothing_is_left_out_and_reported():
    import math
    series, _ = S.make(seed=2)
    rnd, on, decoy = random.Random(4), 0.0, []
    for _ in series.temp:
        if rnd.random() < 0.02:
            on = 1.5 if rnd.random() < 0.4 else 0.0
        decoy.append(on)
    filt, t, temp = D.lowpass(series.boiler, S.TRUE_TAU_STEPS), 18.0, []
    for k in range(len(series.temp)):  # a temperature with no cooling effect at all
        temp.append(round(t / 0.1) * 0.1)
        t += (S.TRUE["outdoor"] * (series.outdoor[k] - t) + S.TRUE["neighbours"] * (series.neighbours[k] - t)
              + S.TRUE["boiler"] * filt[k] + S.TRUE["bias_up"])
    series.temp, series.ac_cool, series.ac_heat = temp, decoy, [0.0] * len(temp)
    rf = D.fit_room(series)
    assert "ac_cool" in rf.dropped and "ac_cool" not in rf.result.active
    assert "boiler" in rf.result.active  # the real effects are kept


def test_real_effects_survive_feature_selection():
    rf = D.fit_room(S.make()[0])
    assert rf.dropped == (), rf.dropped


def test_exposure_hours_say_how_long_each_input_was_on():
    series, _ = S.make()
    ex = D.fit_room(series).exposure
    assert 100 < ex["boiler"] < 125 and ex["ac_heat"] == 14.0 and ex["ac_cool"] == 14.0 and ex["internal"] == 0.0


def test_status_is_judged_where_heating_changes_when_there_is_enough_of_it():
    rf = D.fit_room(S.make()[0])
    assert rf.holdout.n_active >= L.MIN_ACTIVE_WINDOWS
    assert L.skill_active(rf) is not None and L.judging_skill(rf) == L.skill_active(rf)
    flat = D.fit_room(S.make()[0])               # same fit, but pretend almost nothing changed
    import dataclasses
    quiet = dataclasses.replace(flat, holdout=dataclasses.replace(flat.holdout, n_active=3))
    assert L.skill_active(quiet) is None and L.judging_skill(quiet) == L.skill(quiet)


def test_symmetric_adjacency():
    adj = M.symmetric_adjacency({"a": [], "b": ["a", "ghost", "b"], "c": ["b"]})
    assert adj == {"a": ("b",), "b": ("a", "c"), "c": ("b",)}, adj


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {name}: {exc!r}")
    sys.exit(1 if failed else 0)
