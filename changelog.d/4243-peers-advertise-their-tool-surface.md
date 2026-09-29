### Added: a mesh peer advertises the tool it serves, and `call` invokes it

`sim_call` admitted simulation actions from a list kept in the mesh package. The
authority now sits on the peer: a simulation's `wire_tool_spec()` is its
published tool minus a deny table kept next to the spec (`wire_surface.json`,
one reason per entry: world-replacing actions, peer-host windows and paths,
rollouts that ride `execute`/`start`/`stop`, dataset reads, and the params
that name a host path, raw MJCF or an egress switch). Presence carries the
spec's `tool_spec_hash`; the read-only `describe_tool` mesh action returns the
spec; `call` (`function`, `params`) is bounded on the wire for shape and size
the way a Device Connect RPC is, and the peer refuses any function or param it
does not advertise, with the table's reason. Real hardware advertises nothing
and refuses; motion stays on `execute`/`start` with the operator gate.
`sim_call` remains as an alias onto `call`.
