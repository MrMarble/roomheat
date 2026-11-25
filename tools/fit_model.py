#!/usr/bin/env python3
"""Fit each room's thermal model from the RoomHeat history store.

    python3 tools/fit_model.py --db PATH --entries PATH [--days 60]

--db       roomheat_history.db  (HA config dir: .storage/roomheat_history.db)
--entries  core.config_entries  (HA config dir: .storage/core.config_entries)
Both default to the dev instance in dev/config. The database is copied first so
the running Home Assistant is never disturbed. It uses the same code as the
integration, so these are the numbers HA shows.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pure import load  # noqa: E402

L, D, H, M = load("learn"), load("dataset"), load("history_store"), load("model")

STEP = L.STEP


def load_config(entries_path: Path) -> tuple[dict, list]:
    data = json.loads(entries_path.read_text())["data"]["entries"]
    entry = next(e for e in data if e["domain"] == "roomheat")
    subs = [s for s in entry.get("subentries", []) if s["subentry_type"] == "room"]
    adjacency = M.symmetric_adjacency({s["data"]["area_id"]: s["data"].get("adjacent_rooms", []) for s in subs})
    rooms = [
        L.RoomSpec(
            key=s["data"]["area_id"],
            area_id=s["data"]["area_id"],
            name=s["title"],
            sensors=tuple(s["data"].get("temperature_sensors") or [s["data"]["temperature_sensor"]]),
            valves=tuple(s["data"].get("valves", [])),
            adjacent_area_ids=adjacency[s["data"]["area_id"]],
        )
        for s in subs
    ]
    return entry["data"], rooms


def open_store_copy(db: Path, tmp: str) -> "H.HistoryStore":
    for suffix in ("", "-wal", "-shm"):  # copy so a live HA is not touched
        if (src := Path(str(db) + suffix)).exists():
            shutil.copy(src, Path(tmp) / (db.name + suffix))
    return H.HistoryStore(str(Path(tmp) / db.name))


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=root / "dev/config/.storage/roomheat_history.db")
    ap.add_argument("--entries", type=Path, default=root / "dev/config/.storage/core.config_entries")
    ap.add_argument("--days", type=float, default=365, help="use only the most recent N days")
    ap.add_argument("--horizon-hours", type=float, default=1.0, help="open-loop prediction length for the score")
    ap.add_argument("--holdout", type=float, default=0.25, help="fraction of the newest data kept for scoring")
    args = ap.parse_args()

    conf, rooms = load_config(args.entries)
    if not rooms:
        print("No rooms are configured yet.")
        return 1
    hub = L.HubSpec(
        outdoor=conf.get("outdoor_temperature_sensor"),
        weather=conf.get("weather_entity"),
        thermostat=conf.get("central_thermostat"),
        ac=conf.get("ac_unit"),
        ac_power=conf.get("ac_power_sensor"),
    )

    with tempfile.TemporaryDirectory() as tmp:
        store = open_store_copy(args.db, tmp)
        rows = store.read_many(L.needed_series(hub, rooms), 0, 2**40)

    stamps = [ts for r in rooms for s in r.sensors for ts in rows.get(s, {})]
    if not stamps:
        print("The history store has no 5-minute data for the room sensors yet.")
        return 1
    end = max(stamps) + STEP
    start = max(min(stamps), end - int(args.days * 86400)) // STEP * STEP
    data = L.build_dataset(rows, hub, rooms, start, end)
    n = data.n
    print(f"History: {n} steps = {n * STEP / 86400:.1f} days  ({_fmt(start)} -> {_fmt(end)})\n")

    def share(col) -> str:
        return f"{100 * sum(v is not None for v in col) / n:3.0f}%"

    print("Coverage:  outdoor", share(data.outdoor), "| boiler", share(data.boiler),
          "| AC heat", share(data.ac_heat), "| AC cool", share(data.ac_cool))
    print(f"           boiler on for {sum(v for v in data.boiler if v) * STEP / 3600:.0f} h,"
          f" AC heating {sum(1 for v in data.ac_heat if v) * STEP / 3600:.0f} h\n")

    horizon = max(1, round(args.horizon_hours * 3600 / STEP))
    for room in rooms:
        series = data.rooms[room.key]
        usable = L.usable_steps(series)
        print(f"== {room.name}   sensors: {len(room.sensors)}   valves: {len(room.valves)}   neighbours: "
              f"{', '.join(room.adjacent_area_ids) or 'none'}   temperature data: {share(series.temp)}")
        if usable < L.MIN_USABLE_STEPS:
            print(f"   not enough data yet ({usable * STEP / 3600:.0f} h of readings; "
                  f"need about {L.MIN_USABLE_STEPS * STEP / 3600:.0f} h). Keep collecting.\n")
            continue
        rf = D.fit_room(series, horizon=horizon, holdout_fraction=args.holdout)
        if rf is None:
            print("   no complete windows to fit (too many gaps).\n")
            continue
        report(rf, horizon)
    return 0


def report(rf, horizon: int) -> None:
    d = L.describe(rf.result)

    def line(label: str, text: str) -> None:
        print(f"   {label:26s} {text}")

    if d["outdoor_time_constant_h"] is not None:
        line("exchange with outdoors", f"time constant {d['outdoor_time_constant_h']:5.1f} h")
    elif "outdoor" in rf.result.active:
        line("exchange with outdoors", "no measurable exchange")
    if d["neighbour_time_constant_h"] is not None:
        line("exchange with neighbours", f"time constant {d['neighbour_time_constant_h']:5.1f} h")
    elif "neighbours" in rf.result.active:
        line("exchange with neighbours", "no measurable exchange")
    if d["central_heating_gain_c_per_h"] is not None:
        line("central heating", f"+{d['central_heating_gain_c_per_h']:.2f} °C/h when on   (radiator lag {d['radiator_lag_min']} min)")
    if d["ac_heating_gain_c_per_h_per_kw"] is not None:
        line("AC heating", f"+{d['ac_heating_gain_c_per_h_per_kw']:.2f} °C/h per kW")
    if d["ac_cooling_gain_c_per_h_per_kw"] is not None:
        line("AC cooling", f"-{d['ac_cooling_gain_c_per_h_per_kw']:.2f} °C/h per kW")
    line("unmodelled heat", f"{d['unmodelled_heat_c_per_h']:+.2f} °C/h")
    h, s = rf.holdout, L.skill(rf)
    if h and s is not None:
        print(f"   {horizon * STEP // 60}-min-ahead error on unseen data: model {h.rmse_model:.2f} °C vs "
              f"'stays put' {h.rmse_persistence:.2f} °C  (skill {s:+.0%}, {h.n} checks)\n")
    else:
        print("   (no hold-out score: not enough data left after the training split)\n")


def _fmt(ts: int) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


if __name__ == "__main__":
    sys.exit(main())
