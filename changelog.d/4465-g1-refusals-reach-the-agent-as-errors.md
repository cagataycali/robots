### Fixed: a refused `use_unitree` or G1 lookup call reaches the agent as a failed call

`use_unitree`, `g1_joints`, `g1_motion_gates`, `g1_arm_actions` and
`g1_error_codes` returned a flat `{"status": "error", "message": ...}` dict.
Strands reads `status` only off a dict that also carries `content`, so every
refusal - an unknown service, a gate-refused `ZeroTorque`, a `SetVelocity` that
failed on the bus - reached the model as a successful call. Their answers now
arrive as `{"status": ..., "content": [{"json": <the same dict>}]}`, the shape
driver envelopes already have, and the tool result's `status` is the answer's.
Python callers read the fields from `result["content"][0]["json"]`.
