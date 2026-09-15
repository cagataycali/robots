### Fixed: `set_gripper(state="close")` says what the fingers closed on

The reply "gripper commanded close" read the same whether the fingers met an
object or air, and an agent that closed next to a cube lifted and reported a
pick. The MuJoCo backend now reads the contacts after the last tick: bodies
outside the robot that touch the finger subtree are named with their contact
counts ("Closed on 'red_cube' (11 contacts)"), and a close that touched nothing
says so and points at the fix ("Closed on nothing: no object is touching the
fingers, so a lift now carries nothing - move_to the object first"). The
fingers are the bodies the gripper drives, read through the shared
transmission resolver, so a tendon gripper - one `ctrl` coupling both jaws,
which is how every Franka here is actuated - resolves to its pads rather than
to no body at all. A drive whose transmission names no body reports nothing
either way, since an empty contact reading would be a claim about fingers that
were never located. The json payload gains `holding` and `finger_contacts`.
Opening, and backends that do not read contacts, reply as before.
