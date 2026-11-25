"""Control pipeline stages.

One cycle per room is:  read -> resolve_target -> plan -> actuate

Only `read` (in the coordinator) is implemented. The other stages are the
extension points, kept as small pure functions so the thermal model / MPC can be
developed and unit-tested without Home Assistant.
"""

from __future__ import annotations

from datetime import datetime

from .model import HubReading, RoomConfig, RoomReading


def resolve_target(
    room: RoomConfig, reading: RoomReading, hub: HubReading, now: datetime
) -> object | None:
    """Decide the desired temperature (and time ranges) for a room.

    TODO: this is where RoomHeat differs from RoomMind (scheduling / targets).
    """
    return None


def plan(
    room: RoomConfig,
    reading: RoomReading,
    target: object | None,
    hub: HubReading,
    now: datetime,
) -> object | None:
    """Run the learned thermal model / MPC and return what should be done.

    `hub.central_available` says whether central heating can run right now; the
    plan has to look ahead across its 09:00/22:00 edges (pre-heat, then AC).

    TODO: thermal model (learns how the room temperature moves, using the outdoor
    and adjacent-room temperatures) + MPC optimiser.
    """
    return None


# actuate(): intentionally absent. The integration is read-only for now.
