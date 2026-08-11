# robot_mesh validate-before-HITL guards -- measurement artifacts

Reproducible measurement behind the PR "pin both validate-before-HITL contract
guards in robot_mesh".

* `capture.py`  -- drives both gated actions with a `validate_command` that
  returns `None` (the refactor the guards' comment names), with and without each
  guard present, and records what reached the transport. Also re-derives the
  mutation matrix. Writes `art-facts.json`.
* `compose.py`  -- builds the figure; every rendered number is read from
  `art-facts.json` and asserted before the PNG is saved.
* `mutation_table.py` -- the standalone 5-mutation x 2-arm table.
* `art-facts.json` -- the measured data.

Run from a checkout root with `PYTHONPATH=<root> python3 capture.py`.
Each script restores `strands_robots/tools/robot_mesh.py` byte-identically.
