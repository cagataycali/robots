# Safety envelope with zenoh absent

Measurement backing the tests that pin `Mesh`'s four `except ImportError` arms on
the safety-envelope path.

- `driver.py` — drives the end-to-end resume sequence with `zenoh` hidden from the
  import system and prints the fleet state as one JSON line.
- `capture.py` — runs the driver twice: once on the branch as it ships, once with
  M2 applied (`_safety_wire_zid` binding the proof to the local zid even though
  the fallback body it publishes carries no `source_zid`). Asserts every claim.
- `compose.py` — renders the figure; every drawn number is asserted against
  `facts.json` first, text placement is bounds-checked and the border verified.
- `facts.json` — the measured dump.

The decisive rows: on the branch the receiver clears its lockout
(`FLEET AVAILABLE`); under M2 the issuer still reports `{"status": "ok"}`, the
receiver raises nothing, and the lockout stays set — a silent availability
failure reported as success on both ends.
