### Added: SO-arm joints can be addressed by label

The SO-101 asset names its joints `1`..`6` and the SO-100's `Rotation`..`Jaw`,
while the same arms' driver and LeRobot datasets speak `shoulder_pan`,
`shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. An agent
asked to lift the shoulder sent `set_joint_positions {"Shoulder_Lift": 0.3}`,
was refused, and exported the MJCF to guess which servo id it meant.

`registry/robots.json` entries may now carry `joint_labels` (`{asset joint:
label}`; SO-101 and SO-100 do), read through `strands_robots.registry.joint_labels()`.
In the MuJoCo backend `get_robot_state` prints `1 (shoulder_pan): pos=…` and
adds a `joint_labels` map to its JSON (state keys unchanged), and
`set_joint_positions` / `set_joint_velocities` accept a label as a dict key -
bare, `<robot>/<label>`, in any case - resolved to that robot's joint, so
`hold=True` moves the right servo too. A key that resolves to nothing is still
refused all-or-nothing, and the refusal now lists the labels beside the asset
names. Robots without labels are unchanged.
