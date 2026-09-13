### Fixed: a declared embodiment is refused when there is no preprocessor to apply it to

An `embodiment` is a caller's statement of how their observation maps onto the
model: which camera feeds which image feature (`obs_rename`), which scalar keys
compose `observation.state` and in what order (`state_keys`, `dim_policy`), and
what units each side speaks (`state_units` / `action_units`). All of it is
installed as *preprocessor* steps -- `apply_embodiment` sets the rename step's
map and inserts `strands_pack_state` after it.

`ProcessorBridge.from_pretrained` skips a pipeline whose config the checkpoint
omits, and `is_active` is `has_preprocessor or has_postprocessor`. So a
checkpoint shipping only `policy_postprocessor.json` reached the configure hook
with nothing to configure: `apply_embodiment` returned at debug having installed
neither step, and `_configure_embodiment` then latched the map anyway and logged
`Embodiment 'so101' configured`. The bridge's own `Embodiment ... applied` line
was absent -- the claim was made without the act, and nothing warned.

What ran instead was the legacy path, and only half of the declared map went
missing. Measured on a real ACT checkpoint with `embodiment="so101"` and a
MuJoCo SO-101 at `qpos [-0.6, 0.5, -0.4, 0.2, 0.15, 0.9]`: the model received
those radians verbatim, where the declared map converts them to
`[-34.38, 28.65, -22.92, 11.46, 8.59, 55.99]` (arm degrees, gripper
`RANGE_0_100`) -- a 57.3x scale error on every arm column. Meanwhile
`_tensor_to_action_dicts` reads the same latched map and *did* convert the
returned action back with `model_action_to_sim`, dividing by 57.3 on the way
out. The two cameras also fell to positional routing rather than the declared
`obs_rename`. `so100` and `so101` are the two built-in maps that declare
`"degrees"`, so they are exactly the ones this silence lands on.

A declared embodiment that cannot be installed is now refused, naming the
missing `policy_preprocessor.json`, the action-side conversion that would still
have run, and the two ways out (a checkpoint that ships a preprocessor, or
dropping `embodiment=`). The load path already had the right handler for this:
the pipeline is discarded, the raw obs/action flow takes over with *both* halves
of the conversion consistently absent, and `processor_overrides` callers get a
`RuntimeError` instead of a warning.

The map `_configure_embodiment` synthesises from `robot_state_keys` is
deliberately exempt: it carries native units and the same keys the legacy path
already binds, so leaving it unapplied drops nothing -- and refusing it would
discard a postprocessor the checkpoint really did ship.
