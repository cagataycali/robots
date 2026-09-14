### Fixed: removing a robot no longer drops a scene loaded from a file

`remove_robot` does not delete the departing robot's bodies from the live
`MjSpec` - deleting an `spec.attach()`-ed body segfaults MuJoCo at interpreter
shutdown - so it rebuilds the base scene from the `robots` / `objects` /
`cameras` registry and re-attaches the survivors. A scene installed by
`load_scene` is not in that registry: its bodies, lights, tendons and equality
constraints exist only in the compiled spec, so the rebuild omitted every one of
them and still reported `"success"`. Measured on a two-prop scene, both props
left the model while a `cube` added afterwards through `add_object` survived,
because the registry did hold that one.

`remove_robot` now reads the `scene_loaded` marker `load_scene` already recorded
and refuses, ahead of the cooperative policy stop and the registry pop, so the
world is left exactly as it was found. The message names the cause and both ways
forward (re-`load_scene` and `add_robot` only the robots you want, or
`replace_scene_mjcf` for a wholesale swap). The additive verbs are unchanged -
they mutate the loaded spec in place and preserve it - and removal from a
registry-built world is unaffected.
