# Dev instance

```
docker compose up -d
docker compose logs -f           # watch for errors
docker compose restart           # after Python changes
```

UI: http://localhost:8124 (create a throwaway user on first run).

## Pull sensors from your real HA
0. Install "Remote Home-Assistant" from HACS in the dev instance (HACS itself must be installed there first, see hacs.xyz).
1. Real HA: Profile -> Security -> create a long-lived access token.
2. Dev HA: Settings -> Devices & Services -> Add Integration -> "Remote Home-Assistant".
3. Enter the real HA host/port and the token.
4. In the integration's options, add an entity filter for only the sensors
   you need (e.g. `sensor.living_room_temperature`), and a prefix if wanted.
   Do NOT mirror climate/switch entities, and avoid forwarding services,
   so the dev instance can't act on your real home.

## Thermal model

The model (`thermal.py`, `dataset.py`, `learn.py`, `models.py`) is pure Python with
no Home Assistant imports, and **runs inside the integration**: after the history
backfill, and every 6 hours, each room is refitted from the history store. After
every 5-minute bucket the temperature one hour ahead is predicted.

Each room's device gets two entities:

- **Thermal model** (diagnostic): `pending` / `collecting` / `unfit` / `ready` / `poor`.
  `collecting` means under about 3 days of readings. `ready` means the 1-hour
  forecast beats "the temperature stays put" by at least 20% on data it was not
  trained on, judged over the windows where heating changes when there are enough
  of them. The model also drops inputs that do not help on a validation stretch. Attributes hold the learned numbers: outdoor and neighbour time
  constants, central heating gain and radiator lag, AC gains, unmodelled heat,
  forecast error and skill.
- **Predicted temperature (1 h)**: where the room is heading if heating, AC,
  valves and the weather stay as they are. `last_forecast_error_c` is how far off
  the previous prediction turned out to be.

The **Refit thermal models** button (on the RoomHeat device) re-runs the fit now on the
history stored so far. Settings -> Devices & Services -> RoomHeat -> Download diagnostics
shows what the history store holds and each room's model.

### Development tools (not part of the integration)

```
python3 tests/test_model.py        # unit tests: synthetic rooms with known physics
python3 tools/fit_model.py         # the same fit as HA, on files copied from an install
python3 tools/fit_model.py --db PATH --entries PATH --days 60
```

On a real install the two files are `<config>/.storage/roomheat_history.db` and
`<config>/.storage/core.config_entries`.
