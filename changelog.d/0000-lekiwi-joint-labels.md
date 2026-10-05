### Fixed: LeKiwi's arm takes the SO-100 joint labels

The `lekiwi` registry entry now carries `joint_labels`, so the
`shoulder_pan` .. `gripper` dict an `so100` accepts drives LeKiwi's arm too
(its wheels keep their actuator names). Before, `send_action` refused the whole
batch as unresolved keys on the same arm.
