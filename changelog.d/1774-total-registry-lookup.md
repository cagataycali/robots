### Fixed: an entity name that is not a string is reported instead of raising `TypeError`

Every simulation entity is addressed by name, and each backend resolves a
caller-supplied name against a name-keyed registry. A bare `name in registry` /
`registry.get(name)` is not total: for a name that is not hashable (a list, a
dict, a set) the lookup itself raised `TypeError: unhashable type`, so the
unknown-entity error path it guards was never reached and the exception escaped
the agent-tool dict those methods document as their only failure channel.
Twenty-one MuJoCo entry points were affected, `move_object`, `send_action`,
`render` and `move_to` among them, and Newton had the same lookups.

The lookup is now total: `registered` and `registry_entry` resolve a name that
cannot be a registry key to "absent", and the caller reports it with the message
it already had, so no error text changed and a name that does resolve is
unaffected. Entity *creation* is unchanged - claiming a name that cannot be one
needs a contract for what a name may be, not a total lookup.
