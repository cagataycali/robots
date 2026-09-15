### Fixed: `talos` names the model that carries the 32 actuators the catalog advertises

Menagerie's `pal_talos` pack splits the robot across four documents: `talos.xml`
is a shared include holding the bodies alone and declares no `<actuator>`, while
the drivable models are `talos_position.xml` (32 `<position>`) and
`talos_motor.xml` (32 `<motor>`), each with a matching `scene_position.xml` /
`scene_motor.xml`. The pack ships no `scene.xml`.

The catalog entry named `talos.xml` and `scene.xml`. The missing scene fell
through to the bare include, which still compiles - the robot loaded, seated on
the terrain, rendered and reported its 45 joints, so `add_robot` returned success
and went on to advertise `run_policy` on it. It had `nu == 0`: `robot_action_keys`
was empty, and every command was dropped for want of a driving actuator. A robot
the catalog describes as a 32-DOF humanoid could not be moved, and holding its
declared pose for two seconds dropped `base_link` 824 mm as it collapsed.

The entry now names `talos_position.xml` and `scene_position.xml`, so both entry
points reach the 32 position actuators the description states. Holding the same
pose for two seconds now moves `base_link` 18 mm. The position variant is the one
named, not the torque variant of equal actuator count, because a caller sending a
joint angle to a `<motor>` commands that many newton-metres instead.
