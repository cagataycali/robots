# `review/` — operator-side pentest assets for PR #195

This directory holds the offensive-testing assets that accompany PR #195
(`mesh/zenoh-native-security`). Everything here is **operator-side**, not
a regression suite — these are the attack scripts a red-teamer would
run against a production deployment to verify the wire-layer claims hold.

The in-tree pin tests (`tests/mesh/test_*`) are the regression gate;
this dir is the adversarial counterpart.

## Contents

* [`rogue_fleet/`](./rogue_fleet/) — 14 process-isolated rogue robots,
  one per attack vector. Each rogue is its own pid + tempdir, fired from
  `run_fleet.py` which brings up a real `Mesh` victim per scenario.
  See `rogue_fleet/README.md` for the layout and
  `rogue_fleet/XRAY.md` for the codebase data-flow + attack-surface map.

## Quickstart

```bash
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install -e ".[mesh,dev]" cryptography
python review/rogue_fleet/run_fleet.py
# expect: 14/14 defences held in ~9 seconds
```

## Why outside `tests/`

* These scripts spawn real subprocesses, real Zenoh sessions, and real
  PKI material per scenario. They take ~9s end-to-end — too heavy for
  pytest's per-test budget but cheap enough for an operator to run
  pre-deployment.
* Findings get **promoted** to in-tree `tests/mesh/test_*` pin tests
  when they catch a regression — keeping the test suite tight while
  the offensive kit stays comprehensive.
