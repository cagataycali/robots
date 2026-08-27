### Fixed: a joint driven through a spatial tendon is no longer read as free

MJCF spells a tendon two ways, and only one of them names joints. A `<fixed>` tendon lists joint
coordinates, so its wrap entries are joint ids. A `<spatial>` tendon lists a route of sites and wrap
geoms, so its entries come from other id spaces entirely. `tendon_joint_ids` kept only the
`mjWRAP_JOINT` entries, so every spatial tendon read as driving nothing at all whatever it was wired
to -- the "reads an actuated joint as free" failure `actuator_driven_joint_ids` names in its own
docstring, for the tendon kind that the standard cable-driven hand is actuated with throughout.

Three surfaces resolve a tendon through that one rule, and each got a wrong answer from it.
`send_action` looks up the actuator driving a joint when the action key is a joint name, found none,
and dropped the value with `joint has no driving actuator` -- the silent gripper drop issue #318 was
filed to remove. `actuate_robot` saw nothing driven and would add a position servo per joint on top
of the cable already pulling it, the double-actuation its own refusal exists to prevent.
`joint_drive_map` placed those joints in neither the servo nor the other-drive map, so a pose write
treated a cable-driven finger as free. Measured on the shipped `aero_hand` asset (16 joints, 7
actuators, 6 of them cables) loaded through `load_scene`, six actuators reported no driven joint
while MuJoCo's own `qfrc_actuator` moves 15 of the hand's 16 joints from their `ctrl`, leaving one
commandable joint of sixteen.

A spatial tendon's length is the distance along its route, so what it drives is every joint between
the bodies that route touches -- the path from each of them up to their deepest common ancestor, above
which they move together and no distance among them changes. Sites are the points the cable is
anchored to and wrap geoms are obstacles it bends around, and moving either changes the length, so
both end the span; `mjWRAP_PULLEY` carries a divisor rather than an id and contributes nothing. The
rule stays in the shared reader so the two directions -- which joints an actuator drives, and which
actuator drives a joint -- cannot disagree about what a tendon reaches.

Graded against `qfrc_actuator` across the 554 MJCF files in the asset cache: the reading now agrees
for all 75 tendon-transmission actuators, where it agreed for 57 of 75 before, and the 57 joint-wrap
readings are byte-identical. Only the three `tetheria_aero_hand_open` files change.
