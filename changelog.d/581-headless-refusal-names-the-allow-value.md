### Fixed: the command gate's headless refusal names the allowlist value

With no operator reachable, a gated command now says which value pre-approves
it - `STRANDS_ROBOT_COMMAND_ALLOW=execute (or …=*)`, the spelling the tool's
own matcher accepts - and, when the variable is already set to something that
does not match, says so. Before, the refusal named only the variable, even to
a caller who had just set it.
