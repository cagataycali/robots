# move_to on a body-framed end-effector

`capture.py` runs the real MuJoCo scene (headless EGL) and writes
`measurements.json`; `compose.py` builds `move-to-body-frame.png` and asserts
every number it prints against that dump. `mutate.py` / `mutate5.py` run the
mutation table: each regression is applied to
`strands_robots/simulation/mujoco/motion_primitives.py` with an AST-scoped
anchor, measured against the new module and the 161 pre-existing
motion-primitive tests, then reverted byte-identically.
