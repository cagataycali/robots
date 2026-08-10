# Artifact: recording-rate refusal on the two MuJoCo-only rollout entry points

- `capture.py` drives `start_policy` and `run_multi_policy` against a real
  30 fps LeRobot recording while capturing at 50 Hz, renders the scene at each
  stage, and asserts every relation the figure claims.
- `compose.py` builds `rate_refusal.png`, re-deriving each cell from
  `facts.json` so a stale panel cannot ship.
- `mutate.py` applies the two mutation styles (drop the guard / keep the call
  and discard the refusal) and reports each against the new cases and the 79
  pre-existing ones.

Run with `MUJOCO_GL=egl` from a checkout root.
