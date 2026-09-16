### Fixed: the real robot's missing-`policy_port` refusal names the provider and the in-process way

`execute` / `start` without a `policy_port` now says which provider dials a
server (`groot`, marked as the default when it was not chosen), to pass the
port it listens on, and - with no server running - to choose a provider that
builds in process (`mock`, `lerobot_local`). The old remedy pointed at
`run_policy` with a `policy_object`, neither of which this tool's caller can
reach; an agent reading it asked the operator for a port instead of retrying
with `mock`.
