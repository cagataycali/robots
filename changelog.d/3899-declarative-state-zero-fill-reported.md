### Fixed: a zero-filled state dim on the declarative `embodiment` path is reported

`missing_state_keys_used` - the `run_policy` / `eval_policy` flag a collection
loop gates on, documented as the signature of a robot moving on meaningless
inputs - read `False` for every `embodiment`-driven rollout, and `strict_keys=True`
packed the zero instead of refusing it. The declarative path composes
`observation.state` inside LeRobot's pipeline, in the injected
`strands_pack_state` step, and that step had no route back to the policy; only
the generic `robot_state_keys` path honoured either surface. On aloha with
`lerobot/act_aloha_sim_transfer_cube_human` and `embodiment="aloha"` that was two
of fourteen dims carrying no reading (`left/gripper` / `right/gripper` are
actuator names the sim reports as finger joints) under a `success` envelope whose
binding flags all read healthy. The step now records what it zero-filled on the
processor bridge and raises under `strict_keys`, sharing one message with the
warning it already emitted, so one flag and one posture mean the same thing
however `observation.state` is composed.
