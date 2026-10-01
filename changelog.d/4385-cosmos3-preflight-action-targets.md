### Fixed: `cosmos3` refuses, before the server is dialled, a configuration whose actions no actuator would receive

`Cosmos3Policy` keys actions by its embodiment's layout names
(`joint_0..joint_6, gripper`), renamed only through `action_mapping` or
`robot=`. A robot named otherwise received step dicts naming none of its
actuators: `send_action` dropped every command while `run_policy` reported the
rollout ran. The provider now ships a `preflight`: when neither
`action_mapping` nor `robot` is given and no layout name is among the
observation keys, the configuration is a `status=error` envelope naming the
layout, the robot's keys and both remedies. An explicit mapping is not
second-guessed; an unknown robot or embodiment stays the constructor's
refusal. (#4190)
