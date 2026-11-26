"""Control pipeline stages (pure: no Home Assistant imports).

One cycle per room is:  read -> resolve_target -> plan -> actuate

`resolve_target` and `plan` only advise. `actuate` does not exist: the
integration is read-only for now.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .model import Advice, Economics, HubReading, Recommendation, RoomConfig, RoomReading, Target

HYSTERESIS_C = 0.3  # closer than this to the target counts as there
ASSUMED_RATE_C_PER_H = 1.0  # used until the model has learned a heating gain
MAX_LEAD_H = 6.0


def resolve_target(
    room: RoomConfig, schedule_state: str | None, next_event: datetime | None
) -> Target:
    """The room's temperature target right now.

    Schedule on = comfort, off = eco. A room with no schedule is always at
    comfort. A configured schedule that cannot be read falls back to eco.
    """
    if room.schedule is None:
        return Target(room.comfort, "comfort")
    if schedule_state == "on":
        return Target(room.comfort, "comfort", next_event)
    if schedule_state == "off":
        return Target(room.eco, "eco", next_event)
    return Target(room.eco, "eco")


def recommend(
    advice: dict[str, tuple[RoomConfig, Advice | None]],
    hub: HubReading,
    econ: Economics,
    now: datetime,
) -> Recommendation:
    """One heat source for the whole home, from the rooms that need heat soonest.

    Central heating warms every room whose radiator is open, and the AC is one
    ducted unit, so running both at once only wastes energy. The decision is made
    for the moment the first room needs heat (now, if one already does), among
    the rooms needing heat at that moment.
    """
    due = [(r, a) for r, a in advice.values() if a is not None and a.heat_from is not None]
    if not due:
        return Recommendation("none", None, (), (), {}, "no room needs heat")
    at = min(a.heat_from for _, a in due)
    driving = [r for r, a in due if a.heat_from <= max(at, now)]
    window = hub.central_window
    central_ok = window is not None and window.is_open(at)
    reaches = [r for r in driving if r.radiator_in_use]
    costs: dict[str, float] = {}
    if central_ok and reaches:
        costs["central"] = econ.central_price
    if econ.ac_available and econ.ac_cop > 0:
        costs["ac"] = econ.electricity_price / econ.ac_cop
    names = tuple(r.name for r in driving)
    if not costs:
        return Recommendation("none", at, names, names, {}, "no heat source is available then")
    source = min(costs, key=costs.__getitem__)
    unserved = tuple(r.name for r in driving if source == "central" and not r.radiator_in_use)
    return Recommendation(
        source, at, names, unserved, {k: round(v, 4) for k, v in costs.items()},
        "cheapest per kWh of heat among " + ", ".join(sorted(costs)),
    )


def plan(
    room: RoomConfig,
    reading: RoomReading,
    target: Target | None,
    now: datetime,
    learned_rate: float | None = None,
) -> Advice | None:
    """When the room will need heat. The whole home gets one source, from `recommend`.

    The heating rate is the learned central gain when there is one. The estimate
    ignores the room cooling in the meantime, so it is a floor on the lead time.
    """
    temp = reading.temperature
    if temp is None or target is None:
        return None
    assumed = not (learned_rate and learned_rate > 0.05)
    rate = ASSUMED_RATE_C_PER_H if assumed else learned_rate

    def none(reason: str, deficit: float = 0.0, comfort_at: datetime | None = None) -> Advice:
        return Advice(False, None, comfort_at, round(deficit, 2), None, rate, assumed, reason)

    if target.level == "comfort":
        deficit = target.temperature - temp
        if deficit <= HYSTERESIS_C:
            return none("at comfort temperature")
        start, comfort_at, lead = now, now, 0.0
    else:
        comfort_at = target.changes_at
        if comfort_at is None:
            return none("no comfort period ahead is known")
        deficit = room.comfort - temp
        if deficit <= HYSTERESIS_C:
            return none("already warm enough for the next comfort period", comfort_at=comfort_at)
        lead = min(deficit / rate, MAX_LEAD_H)
        start = max(now, comfort_at - timedelta(hours=lead))

    return Advice(
        needs_heat_now=start <= now,
        heat_from=start,
        comfort_at=comfort_at,
        deficit_c=round(deficit, 2),
        lead_minutes=round(lead * 60),
        rate_c_per_h=round(rate, 2),
        rate_assumed=assumed,
        reason="heat needed now" if start <= now else "pre-heat to reach comfort on time",
    )


# actuate(): intentionally absent. The integration is read-only for now.
