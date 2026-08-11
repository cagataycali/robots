# Independent verification of PR #2143 — shared LeRobot cache isolation

Measured on a machine with real `mujoco` / `torch` / `lerobot` installed.

- `inspectprobe.py` — pytest plugin separating a *resolution* of
  `$HF_LEROBOT_HOME/{repo_id}` from an on-disk *inspection* of it
  (`_prepare_create_target`, the step that stats the directory). Run with
  `-p inspectprobe`.
- `capture.py` — per tree, records the resolve/inspect split and the
  planted-cache outcome; writes JSON. Plants a dataset at the two inspected ids
  and removes it in a `finally`, asserting the cleanup.
- `compose_pr2143.py` — builds the figure from the two dumps, asserting every
  rendered number first.
- `art_main.json` — pristine `9e0b77b9`. `art_theirs.json` — PR #2143 at `c9a2e57`.
