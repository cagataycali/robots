### Tests: a render test's GL gate is the shared probe's marker, not a local rebuild

`ROBOT_TEST_MUJOCO=0` is the force-skip an operator sets to keep a known-bad
runner from attempting GL at all -- the host class where a second offscreen
renderer construction aborts the interpreter uncatchably instead of raising. It
is read by `tests/simulation/mujoco/_gl_probe.gl_available` and by nothing else,
so the 17 modules that rebuilt the gate locally from the private
`strands_robots.simulation.mujoco.backend._can_render` never saw it: with the
variable set, 56 GL-dependent cases across those modules still ran and
constructed renderers. Those gates were also invisible to the render-gating scan
in `tests/test_mujoco_render_assertions_are_gl_gated.py`, which recognises
gating through the shared probe, so a render assertion added to any of them
would have been reported as ungated.

All 17 now take `requires_gl` from the shared probe, and
`tests/test_the_gl_gate_on_a_render_test_has_one_owner.py` keeps it that way:
exactly one module in the tree may build a GL-gating skip marker. Reading
`_can_render` is unchanged and unreported -- its own contract tests do that; the
discriminator is building a second gate out of it.
