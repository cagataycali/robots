### Fixed: the 0.7 removal warning names the provider to type

`create_policy("protomotions")` warned "instead use the wbc provider", which
says what to switch to but not what to write. The warning now has the same
shape as the registry's removed-provider refusals - `policy_provider
'protomotions' is removed in 0.7: use policy_provider='wbc' for Unitree G1
whole-body control.` - and the `kimodo` notice names the `Policy` subclass a
replayed offline motion goes through, since no in-tree provider replaces it.
