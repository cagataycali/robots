### Changed: the dashboard's fleet agent uses what each peer advertises

The sim peer tool in the dashboard mirrored a package copy of the simulation's
spec. It is now a projection of what the peer serves: presence carries the
peer's `tool_spec_hash`, the console fetches the spec once per hash with
`describe_tool` (a child peer reuses its parent's), and the tool's action enum
and parameter schema are that spec, verbs first. A function rides the `call`
rail; a denied function, an unknown one or a parameter the function does not
take is refused before the wire with the peer's own reason. A peer that
advertises nothing (real hardware, an older build) keeps the static tool.
