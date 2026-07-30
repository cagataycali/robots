### Fixed: a MuJoCo entity name that is not a string is reported instead of crashing the process

`mujoco.mj_name2id` declares its name parameter as `const char *`, and the pybind11
binding maps Python `None` onto a NULL pointer rather than rejecting it. MuJoCo
dereferences that pointer while comparing names, so the call does not raise - it
terminates the interpreter with SIGSEGV. Five agent-callable methods passed a
caller-supplied name straight into that binding, so one argument of the wrong type
killed the process: `get_body_state`, `set_body_properties`, `apply_force`,
`attach_bodies`, and `set_joint_positions` (where the name arrives as a mapping key).
Nothing above the call could recover - the agent-tool envelope, the caller's `except`
clauses and any open recording all died with it.

Three further methods - `move_object`, `remove_object` and `remove_camera` - reached
the shared "did you mean" reporter instead, where `difflib.get_close_matches` raised a
bare `TypeError` past the envelope those methods document as their only failure channel.

Every lookup now goes through `mj_name_to_id`, which resolves a non-string name to
"not found" so the existing unknown-entity message reports it, naming the value and
listing what the model does contain. An AST check pins that no module reaches the
binding directly, so a lookup added later inherits the guard.
