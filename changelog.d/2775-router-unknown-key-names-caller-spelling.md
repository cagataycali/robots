### Fixed: an unknown router key is named by the spelling the caller wrote

`_dispatch_action` rewrites `_FIELD_ALIASES` to method parameter names before validating, so every
refusal downstream of that rewrite reports the name it validated rather than the field the payload
carried. `_reported_param_name` exists to undo that for the caller, and four of the five refusal
sites in `_validate_and_build_kwargs` consult it. The unknown-parameter branch interpolated
`unknown[0]` -- a key taken straight out of the post-rewrite payload -- so the offending key was
reported canonically while the `Valid:` list beside it, built through the helper in the same
statement, was reported in caller spellings.

Every alias reproduces it, because the rewrite is unconditional and the target is a parameter of one
action rather than all of them: sending `torque_vec` to any action but `apply_force` was answered
`Unknown parameter 'torque'`, and `checkpoint_name`, `camera_names` and `joint_positions` were
answered `name`, `cameras` and `positions`. Three of those four canonical names are not published
properties at all, so a schema-constrained caller was named a field it is not permitted to emit and
had no route from the message back to the one that works -- the dead-end diagnostic
`_reported_param_name` was introduced to remove, surviving in the one branch that read its loop
variable directly.

The branch now routes through the same helper. A key the alias map does not mention still reports
itself, and a caller who writes the canonical parameter directly is still named by it: the
substitution is driven by what the payload carries, not by the map, so it cannot answer either
caller with a field they did not send.

The contract's structural guard is replaced rather than extended. It forbade two literal format
strings and asserted a minimum count of helper calls, which grades how a slot is written and how
many there are -- not where a slot's value came from -- so it stayed green for the one raw site it
had not been told about, and no count could have moved. The replacement asks the dataflow question
instead: a name assigned from an expression over `remapped` or `received` holds post-rewrite keys,
and no refusal in the function may interpolate one. The behavioural sweep beside it is derived from
`_FIELD_ALIASES`, so an alias added later is graded on arrival without a second list to keep true.
