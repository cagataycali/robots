### Fixed: `list_providers()` no longer lists a `register_policy` alias as a provider

An alias passed to `register_policy(..., aliases=[...])` now appears only in
`list_aliases()`, as the registry's own aliases always have, so
`set(list_providers()) | set(list_aliases())` names each spelling once and the
real robot's unknown-provider refusal lists provider names only. The alias
still resolves through `create_policy()` and still appears in its
did-you-mean hint.
