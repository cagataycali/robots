### Fixed: create_policy names the provider for every policy the README lists

`create_policy("GR00T N1.7")`, `create_policy("whole-body control")` and
`create_policy("scripted")` raised a bare "Unknown policy provider" with no
suggestion, because no provider is spelled that way and difflib found nothing
close. Each now raises one sentence naming where it runs: GR00T N1.7 through
`lerobot_local(policy_type='groot')`, whole-body control through `wbc` (or
`holosoma`), a scripted policy through a `Policy` subclass and
`register_policy`. Refused spellings now match ignoring case, spaces, `-`, `_`
and `.`, so `gr00t-n1.7` and `Whole_Body_Control` are caught too.
