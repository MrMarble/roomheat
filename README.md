# RoomHeat

A Home Assistant integration that **learns how each room's temperature moves** and
predicts where it is heading. It is built for a home where heating is central and
only available some hours of the year, with an air-conditioning heat pump as the
other source. It is inspired by [RoomMind](https://github.com/snazzybean/roommind).

> **Status: alpha, read-only.** RoomHeat reads your sensors and learns. It does
> **not** control anything yet: no thermostat, valve or AC is ever commanded.

## What it does today

- **Rooms from areas.** Add a room per Home Assistant area, with its temperature
  sensors, optional radiator-valve sensors, radiator valves and adjacent rooms.
  Two optional per-room settings tell the model what it cannot see: *Central heating
  radiator in use* (turn it off for a room whose radiator is normally off) and
  *Appliance power sensors* (computers and the like, whose heat it then learns).
- **Central heating window.** Season dates and daily hours; it knows when central
  heating is available. The AC covers the rest.
- **A history of its own.** Every 5 minutes it stores room and outdoor
  temperatures, valve and boiler states, AC power, setpoints and weather. On
  first start it backfills from the recorder, so it does not begin empty.
- **A thermal model per room.** Learns exchange with outdoors and neighbours, how
  much the central heating and the AC heat the room (gated by the valves), and
  how long radiators keep giving heat. Everything is learned from your data.
- **Targets and advice.** Give a room a `schedule` helper and comfort/eco
  temperatures: while the schedule is on it targets comfort, otherwise eco (no
  schedule = always comfort). RoomHeat then reports *Target temperature*, *Heat
  needed from* (when to start heating to reach comfort on time) and a hub-level *Recommended
  heat source* (one for the whole home, never both: central or AC, cheapest per kWh of heat: central price vs.
  electricity price / heat-pump COP, and only central when its window is open).
  Advice only; nothing is controlled. Until the model has learned a heating gain
  the lead time assumes 1 °C/h and says so.
- **Entities.** Per room: *Thermal model* (status and learned numbers),
  *Predicted temperature (1 h)* and *Temperature*. Plus *Central heating
  available* and a **Refit thermal models** button, which re-learns every room
  now instead of waiting for the next automatic refit (every 6 hours). Settings → Devices & Services → RoomHeat → Download diagnostics
  shows what the history holds.

## Install (HACS)

1. HACS → ⋮ → **Custom repositories**.
2. Repository: `https://github.com/MrMarble/roomheat`, type **Integration**.
3. Install **RoomHeat**, restart Home Assistant.
4. Settings → Devices & Services → Add Integration → **RoomHeat**.

Then add rooms with **Add room** on the integration page.

**Tip:** the recorder keeps only 10 days of state history by default. Set
`recorder: purge_keep_days: 60` (or more) *before* installing, so the first
backfill has more to work with. RoomHeat keeps its own history afterwards.

## Requirements

Home Assistant 2025.3 or newer (config subentries). Developed and tested on
2026.11; older versions are untested. No extra Python packages.

## Development

See [DEV.md](DEV.md): a Docker dev instance, the unit tests
(`python3 tests/test_model.py`), and an offline fitting tool.

## License

MIT. See [LICENSE](LICENSE).
