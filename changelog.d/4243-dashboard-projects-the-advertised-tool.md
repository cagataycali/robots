### Changed: the dashboard's fleet agent uses what each peer advertises

The sim peer tool in the dashboard mirrored a package copy of the simulation's
spec. It is now a projection of what the peer serves: presence carries the
peer's `tool_spec_hash`, the console fetches the spec once per hash with
`describe_tool` (a child peer reuses its parent's), and the tool's action enum
and parameter schema are that spec, verbs first. A function rides the `call`
rail; a denied function, an unknown one or a parameter the function does not
take is refused before the wire with the peer's own reason. A peer that
advertises nothing (real hardware, an older build) keeps the static tool.

### Added: `--mesh-listen`, the dashboard as the mesh anchor

`strands_robots dashboard --mesh-listen tcp/0.0.0.0:7447` sets `ZENOH_LISTEN`
before the mesh session opens, so the dashboard is the first node robots on the
LAN attach to. The header shows the line they need
(`ZENOH_CONNECT=tcp/<lan-ip>:7447`, click to copy) and `/api/mesh/config`
carries it as `anchor_hint`. A malformed endpoint is refused before anything
starts.
