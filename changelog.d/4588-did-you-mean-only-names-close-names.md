### Fixed: "Did you mean" in a simulation refusal no longer names unrelated actions

An unknown action, parameter, robot, object, camera or model name used to get
up to three suggestions that only shared a few letters with it:
`action="pick"` was answered with `run_policy, stop_policy, eval_policy`, and
`action="grab"` with `set_gravity`. The shared suggestion helper now uses
difflib's default cutoff (0.6, was 0.4). A name that is the start of a known one
(`policy` -> `policy_provider`, `arm` -> `arm/base`) is still suggested. A name
with no close match gets no suggestion, and the refusal still says where the
full list is.
