### Fixed: a MuJoCo scene rebuild defines every buffer it grows, and a refused duplicate insert is rolled back by identity

`mujoco 3.14.0` changed two things `add_robot` / `add_object` / `add_camera`
relied on, and `main` was red from the first CI run that resolved it.

`spec.recompile(model, data)` now carries `qacc_warmstart`, `eq_active`, both
applied-force buffers and mocap poses by element, and when the spec was grown
by attaching a sub-spec that carries a `<keyframe>` - every robot description
in the registry - the slices of the attached elements come back as heap
garbage (`nan`, `6.98e-316`, `1.12e+219`; the pre-existing slices intact).
`_recompile_preserving_state` defined the new tail of `qpos`, `qvel`, `ctrl`
and `act`, the four buffers the 3.5-3.13 transfer left undefined, and assumed
the rest zero, so a second robot's dofs entered the solver with a NaN warm
start and the scene reported `Nan, Inf or huge value in QACC at DOF 0` against
an arm that was parked and healthy - on the runs where the freed memory
happened to be non-finite. The rebuild now trusts the compiler with nothing a
reset would write: `qacc_warmstart` is zeroed on the new dofs, `eq_active` and
`mocap_pos`/`mocap_quat` take their declared values, and both applied-force
buffers are zeroed whole before the name-keyed snapshot is re-applied.

A refused `add_body(name=...)` / `add_camera(name=...)` against a name the scene
already holds still appends the orphan, but the failed rename now preserves the
element's previous name, so the orphan is nameless (and at the origin) where it
used to carry the colliding name. The rollback found the surplus by that name,
so on 3.14 it deleted nothing and the live spec kept a nameless body after
every refused collision - and compiled. `SpecBuilder.snapshot_bodies` /
`remove_bodies_not_in` (and the camera pair) replace `count_*_named` /
`remove_surplus_*`: the surplus is whatever is present after the insert and
absent from the snapshot taken before it, the same set on every build.

Both are pinned independently of heap luck: a recompile proxy that writes the
3.14 garbage into every new slice, and the identity rollback measured against a
nameless orphan and against a worldbody camera that is not last in enumeration.
