### Fixed: `use_unitree` gates every verb it does not know to be a read

The first F-001 fix put every known mutative Unitree SDK verb behind the
operator gate, but the decision of *whether to ask* was a denylist:
`_is_mutative` answered true only for names starting with one of 24 hardcoded
prefixes, and everything else - not a read, not private, just spelled
differently - dispatched with no prompt. `_execute` calls any public method of
the client and `list_operations` advertises every one, so a verb added on the
next SDK bump (`Recover`, `Trigger`, `ArmTask`, ...) would have been
discoverable, dispatchable and ungated, silently (CWE-862). The classifier is
now an allowlist: a name is a read only when it is in `READONLY_WHITELIST` or
is a `Get*` / `Check*`; every other public verb stops for an operator, unknown
ones included. `MUTATIVE_PREFIXES` stays as documentation and as a floor the
tests hold. `HIGH_DANGER_OPS`, `STRANDS_UNITREE_COMMAND_ALLOW`,
`BYPASS_TOOL_CONSENT` and the never-gated `loco.StopMove` are unchanged.
