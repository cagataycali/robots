### Fixed: terrain seating moves the robot's own floating base, not a task object the robot declares

`_robot_free_base_joint_id` answers "which free joint is this robot's floating
base?" for two consumers: `_seat_floating_bases_on_terrain`, which WRITES `qpos`
at whatever joint it is handed, and the `start_recording` schema, which only
tests the answer against `>= 0`. The same question is answered twice more,
inlined, by `_get_sim_observation` and `get_robot_state`.

Those two loops spell the precedence as "the ownership-checked resolver decides;
the named scan only supplies a candidate", and both carry the same eleven-line
comment explaining why: a robot whose MJCF ships a free-jointed task object
under its own namespace - a payload, a kick ball, the grasping cube a Menagerie
manipulation scene declares - names that joint in `robot.joint_names` too, so a
scan that CHOOSES from that list can land on the object.

The finder did not apply that precedence. It returned the first free joint named
in `joint_names` and consulted the ownership-checked resolver only when the scan
found nothing, so the three surfaces disagreed on one compiled model. Measured
on a `stairs` world with a mobile base whose own `<freejoint>` is unnamed (so it
is absent from `joint_names` entirely) carrying a free-jointed payload at
`x=3.5`:

| | before | after |
| --- | --- | --- |
| finder named | `carrier/payload_free` | the chassis `<freejoint>` |
| `get_observation` `base_pos` | the chassis | the chassis |
| base raised by seating | `+0.0000 m` | `+0.0400 m` |
| payload moved by seating | `+0.0800 m` | `+0.0000 m` |

Terrain under the chassis is `0.04 m` and under the payload `0.08 m`, so seating
sampled the wrong `(x, y)` and applied the offset to the wrong body: the robot
kept its flat-ground height and is buried `0.04 m` under the raised terrain -
the one state the seating pass exists to prevent - while a task object levitated.
The observation surfaces were right throughout, so the seated pose and the
observed pose described different bodies.

The named scan now records a candidate instead of returning one, and the
ownership-checked resolver decides; its `-1` still does not erase a candidate the
scan found, which is what keeps a fixed-base arm carrying a free-jointed prop
reporting the same joint it reported before. That makes the finder's own
docstring true again - it claimed to mirror the inlined detection.

The recording-schema consumer is unaffected: the finder and the loops can differ
on WHICH joint, never on whether one exists, so `finder(...) >= 0` and
`"base_pos" in get_observation(...)` agreed before and still agree. A test pins
that.

`tests/simulation/mujoco/test_terrain_seating_moves_the_robots_own_base.py` pins
the rows above plus the two controls that keep the fix from becoming an
over-correction - an ordinary named floating base is still resolved, and a
fixed-base arm carrying a prop still reports the scanned joint. The two
pre-existing pins on this finder assert only `>= 0` and `jnt_type == mjJNT_FREE`,
both of which the prop's joint satisfied, which is why the disagreement was not
observable before.

Production cost is two executable statements.
