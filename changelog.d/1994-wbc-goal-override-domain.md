### Fixed: a WBC per-call `height` / `target_orientation` is held to the domain of the config default it overrides

#1991 gave `WBCConfig.height_cmd` and `rpy_cmd` a value domain. Both have a
per-call spelling on `WBCPolicy.get_actions` - `height` and
`target_orientation` - which writes the SAME command slot and was unchecked, so
the same number was refused as a default and accepted as an override of that
default. The sibling `target_velocity` on that method has always been guarded,
which made this an asymmetry rather than a uniform gap.

Measured on the real conversion, no `onnxruntime` needed: `WBCConfig` refused
`height_cmd=nan`/`inf`/`True`/`'0.8'` while `get_actions(height=...)` accepted
every one of them - `nan` straight into `command[3]`, and `True` as a silent
1 m base-height command. The damage does not stay on its tick: the returned
action is stored as `_prev_action` and fed into every later observation frame,
so ONE unusable goal made all 15 joint targets non-finite and kept them
non-finite for every subsequent tick, with the caller's next perfectly usable
goal applied exactly as asked and no error at any point. Those targets reach
`data.ctrl`, so the observable failure was a diverged MuJoCo state attributable
to no parameter. Only `reset()` cleared it.

`height` must now be a finite number and every `target_orientation` component
finite - the same `finite_number_error` / `finite_vector_error` the config
composes, so the two surfaces cannot drift apart by editing one of them. Length
is deliberately not constrained, so a 4- or 6-component orientation is still
truncated as before. The guard runs as the first statement of
`_resolve_command`, itself the first statement of `get_actions`, so a refused
goal pushes no observation frame, leaves `_prev_action` untouched, advances no
gait phase and runs no inference. `WBCGaitPolicy` overrides `_resolve_command`
wholesale and is covered too; a structural scan pins that a third
implementation cannot skip the domain.
