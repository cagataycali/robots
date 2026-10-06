### Fixed: a robot page's joint-label table says the labels are accepted as well as the printed keys

The `lekiwi`, `so100` and `so101` pages print `robot_action_keys` (actuator
names such as `Rotation`) and then show a table of labels such as
`shoulder_pan`. The table's right column was headed "`send_action` label",
which read as the one name to use. It is now headed "also accepted by
`send_action`": both spellings drive the same joint, and the keys the fence
prints stay the ones a recording or a policy's state is keyed by.
