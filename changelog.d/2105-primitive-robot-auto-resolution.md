### Quality: pin the motion primitives' documented `robot_name` default

`move_to` / `set_gripper` / `rotate_wrist` each document `robot_name` as
"defaults to the single robot in the world (errors if ambiguous)", and each
documents "Never raises.". Every existing primitive test named the robot
explicitly, so nothing held that default: not the resolution itself, not the
ambiguity refusal, and not the `ValueError`-to-envelope conversion the
never-raises contract rests on. All three documented outcomes - one robot, zero
robots, many robots - are now pinned on all three primitives against a real
MuJoCo scene. Four mutations of the shared preamble, including one that silently
resolves the first of two robots rather than refusing, pass the existing 77
primitive tests and are each caught by these.
