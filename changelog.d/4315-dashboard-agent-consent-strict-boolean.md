### Fixed: an agent-console consent answer is the JSON boolean true or a refusal

`/ws/agent` answered a motion interrupt with `bool(frame.get("approve"))`, so
the string `"false"`, `"no"`, `1` or a non-empty list read as consent (f025).
`approve` and `always` are now parsed strictly: `true` or the string `"true"`
means yes, anything else means no, so a misspelt refusal never becomes a yes.
