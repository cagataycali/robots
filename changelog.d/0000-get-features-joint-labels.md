### Fixed: `get_features` sidecar surfaces the registry's `joint_labels`

The MuJoCo `get_features(robot_name=...)` discovery call already shipped a
per-robot sidecar at `.content[1].json.features.robots.<name>` carrying
`joint_names`, `n_joints`, `n_actuators`, `data_config` and `source`, but it
dropped the registry's `joint_labels` block that `get_robot_state` on the SAME
object had been rendering since harness#520. On `so101` -- whose upstream
MJCF names its servos `1`..`6` -- the agent-callable "what does this robot
have?" surface read `joint_names: ['1','2','3','4','5','6']` with no bridge to
the `shoulder_pan`..`gripper` vocabulary the sibling `send_action` refusal and
`get_robot_state` already carried. An agent reading `get_features` for the
first time learned the arm had six integer joints with no hint which one was
the gripper.

The sidecar is pure discovery metadata (never consumed by a recorder or by
`Policy.set_robot_state_keys` -- those read `robot_action_keys` whose shape is
locked for dataset-column stability by harness#712 / harness#768), so the fix
is additive and column-stable. `get_features` now carries
`"joint_labels": self._robot_joint_labels(robot)` beside `joint_names`, and
the human-readable text block adds `joint_labels: 1 (shoulder_pan), ...` on
the line below each robot's summary -- the same shape
`_world_readiness_sentence` adopted in harness#758. Robots whose registry
entry declares no labels (panda, unitree_g1, unitree_go2, ...) get `{}` in the
sidecar and no new text line, so their legacy format is byte-identical.
