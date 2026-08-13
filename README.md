# create_world timestep domain across backends

Measurements for the PR pinning the `create_world(timestep=...)` domain on every
simulation backend. Nothing here is hand-typed: `capture.py` measures the shipped
code and writes `measurements.json`, and `compose.py` asserts every rendered
number against that file before saving the figure.

- `capture.py` - drives all six (backend x knob) cells with `dt = -0.002`, probes
  `IsaacConfig(physics_dt=...)` for the four values whose config verdict differs
  from `create_world`'s, and renders a world genuinely built through
  `create_world(timestep=0.002)` in MuJoCo headless (`MUJOCO_GL=egl`).
- `compose.py` - the figure, with layout and value guards.
- `mutate.py` - the mutation table: eight regressions applied to the shipped
  guards, each run against the new cells and against this module's pre-PR version
  (fetched from the merge base), restored byte-identically afterwards.
- `measurements.json` - the raw dump every number is read from.
