### Fixed

- **policies/cosmos3**: `action_mapping` is now required to be a rename. Two
  action columns arriving at one actuator name used to collapse into a single
  entry of the per-step action dict, dropping the losing column's command with
  `status="success"` - reachable from a single mapping entry aimed at another
  column's own name (`{"joint_0": "joint_1"}` on the DROID
  `[joint_0..joint_6, gripper]` layout), and worse in reverse
  (`{"gripper": "joint_6"}` commanded `joint_6` with the gripper's value and left
  the gripper uncommanded). Construction already validated the mapping's keys;
  it now validates the targets too, refusing before any request reaches the
  server and naming the colliding columns and the target they arrive at.
