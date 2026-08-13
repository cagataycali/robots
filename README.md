# Newton solver articulation domain

Measurements behind the change that refuses a Newton solver which cannot drive an
articulated robot. Everything here was run on an NVIDIA Jetson AGX Thor (CUDA
13.0) against newton 1.5.0 / warp 1.16.0, using the two-hinge arm in
`probe_arm.xml`.

## Probes

- `probe_solver_matrix.py` / `facts_solver_matrix.json` -- drives all eight
  solvers Newton resolves through `create_world`, `add_robot`, `send_action` and
  `step`, recording joint travel and every status.
- `probe_vbd_with_coloring.py` / `facts_vbd_with_coloring.json` -- measures the
  remedy the `vbd` Newton error names (`ModelBuilder.set_coloring` / `color()`):
  it finalizes, constructs and steps, and the arm still does not move.
- `probe_gravity_only.py` / `facts_gravity_only.json` -- steps under gravity with
  nothing commanded, separating "ignores targets" from "does not integrate rigid
  bodies at all".
- `probe_body_vs_joint_state.py` -- reads body poses beside joint state, to rule
  out bodies drifting while joint observations stay flat.
- `probe_gated_assertions.py` -- runs the Newton-gated assertions of the new test
  module under a real Newton install, since that environment has no pytest.

## Figure

- `capture.py` sweeps four cameras and picks the framing by measurement, then
  renders the same commanded pose under an accepted solver and a refused one.
- `compose.py` builds `newton_solver_domain.png` and asserts every number it
  draws against `facts.json`.
- `art_arm.xml` is the chunkier scene used for the figure; the quoted travel
  numbers elsewhere come from `probe_arm.xml`.
