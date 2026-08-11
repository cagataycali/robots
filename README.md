# Shared LeRobot cache isolation — measurement scripts

- `inspectprobe.py` — pytest plugin separating a *resolution* of
  `$HF_LEROBOT_HOME/{repo_id}` from an on-disk *inspection* of it
  (`_prepare_create_target`). Run with `-p inspectprobe`.
- `capture.py` — measures the resolve/inspect split and the planted-cache
  outcome for one tree; writes JSON. Run once per tree.
- `compose.py` — builds the figure from the two dumps, asserting every
  rendered number first.
- `art_main.json` / `art_branch.json` — the two measurements.
