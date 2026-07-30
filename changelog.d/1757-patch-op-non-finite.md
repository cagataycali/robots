### Fixed: `patch_scene_mjcf` refuses a numeric op field that is not finite

Every numeric field a patch op writes into the compiled MJCF - `pos`, `quat`,
`size`, `rgba` - is now checked for finite numeric components before the spec is
touched, the same domain `add_object`, `add_camera` and `move_object` already
apply to those fields.

MuJoCo does not reject a `nan`/`inf` pose, extent or colour (its one exception is
a `nan` geom size), so the component was written verbatim into the model and the
patch reported success. `{"op": "set_body_pos", "pos": [nan, 0, 0.3]}` on a body
owning a freejoint left `qpos` and `qvel` non-finite after the next step, with the
patch, the step and the observation read all reporting success - so the same field
`move_object` refused was accepted by `set_body_pos`, for one identical write to
`body_pos`.
