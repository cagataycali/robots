### Fixed: a MuJoCo world that diverged is reported, not stepped through as a success

When `mj_step` finds a NaN, inf or huge joint position, velocity or acceleration,
MuJoCo resets every robot and object to the model's initial pose and the clock
to zero, and says so only on stderr. `step`, `send_action`, `move_to`,
`set_gripper`, `rotate_wrist` and `run_multi_policy` now answer
`status="error"` naming the divergence, the joint MuJoCo flagged and `reset` as
the recovery, with `{"diverged": true}` in the JSON block. `run_policy`,
`eval_policy`, `evaluate_benchmark` and `replay_episode` stop on it instead of
finishing on a reset world: an evaluation drops the diverged episode, reports
`physics_error`, and discards its unsaved recording frames. Before, a
`1e12 N` shove answered `success` with the clock running backwards, and an
evaluation scored 2/2 episodes over a world MuJoCo had reset.
