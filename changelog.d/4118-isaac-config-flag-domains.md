### Fixed: IsaacConfig grades its timestep and posture fields at construction

`IsaacConfig.__post_init__` graded `physics_dt`/`rendering_dt` with a bare
`<= 0` and left `headless`/`ground_plane`/`verbose` unchecked. The comparison is
`False` for `nan` and `inf` and read `True` as a dt of 1.0 (a one-second physics
step), so a dt no integrator can advance by was stored to be caught a call later
under a knob the caller never spelled, or not at all; a non-boolean flag such as
`headless="off"` reached the `SimulationApp` launch dict verbatim while the
`STRANDS_ISAAC_HEADLESS` env door already refused `"maybe"`. Both dt fields now
pass through `positive_finite_number_error` and the three flags through
`boolean_flag_error` - the shared domains their sibling surfaces already use - so
each is refused at construction under its own name.
