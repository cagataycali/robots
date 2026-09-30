### Fixed: robot_mesh's headless refusal names what pre-approves the call

A gated `robot_mesh` action refused for want of an operator now names the `STRANDS_MESH_HITL_ACTIONS` value that pre-approves that one action and `BYPASS_TOOL_CONSENT=true`, and the bypass is honoured the way every other operator gate honours it: the interrupt is skipped with a WARNING and an audit row, the rate limit stays. The agents page no longer says stopping is never gated while its table gates `robot_mesh` `stop`. (#4156)
