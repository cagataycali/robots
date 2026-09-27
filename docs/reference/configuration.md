# Configuration

Every environment variable the package reads, generated from the source at build time. After this page you can find the variable behind any behaviour you want to change, see which module reads it and what the code assumes when it is unset.

The package has no config file. Behaviour is set three ways, in this order of precedence:

1. Keyword arguments on the call: `Robot("so101", mode="real", driver="strands")`.
2. Environment variables, listed below. A variable named in the `grant` column of [refusal codes](refusal-codes.md) lifts one refusal; the `STRANDS_*_COMMAND_ALLOW` family pre-approves commands for the operator gate.
3. Defaults in the code, shown in the `default` column when the read passes a literal. `unset` means the code handles the missing variable itself; open the module named in `read in` for the branch.

Two rules apply everywhere. A boolean variable accepts `1`, `true`, `yes` (case-insensitive) and treats anything else as off. A variable that names a security posture (`STRANDS_MESH_AUTH_MODE=none`, `DEVICE_CONNECT_ALLOW_INSECURE`, `BYPASS_TOOL_CONSENT`) is read once and logged at WARNING when it weakens the default.

The `meaning` column is the first sentence in a docstring or comment of the reading module that names the variable. Where the source has no such sentence the column points at the module instead.

{{env_vars}}
