"""
Repro: SimEngine.robot_joint_names(wrong_name) silently returns [] instead of
raising, giving a user who follows docs/robots/unitree_g1.md verbatim a
zero-column state vector with no warning.

Pre-cond:
  - strands_robots editable install, MuJoCo backend.
  - MUJOCO_GL=egl (or any valid backend).

Expected: ValueError / NameError naming the registered robot(s), or a logger
warning, so the user discovers the mismatch before piping [] into a policy's
`set_robot_state_keys()` and getting silent no-ops.

Actual: `[]` with no stderr, no log, no raise.

Steps to see the footgun as a NEW user:

1. Open docs/robots/unitree_g1.md.
2. Line 10 lists aliases: g1, g1_wbc, unitree_g1_full_body, ...
3. User prefers the short alias and writes `Robot("g1")`.
4. Line 18 of the SAME doc says literally:
       print(robot.robot_joint_names("unitree_g1"))
5. User copies that line verbatim because they are reading the unitree_g1
   page. They now get `[]`, run their policy, every actuator silently no-ops,
   and the simulation renders the robot standing like a mannequin.

Root cause: strands_robots/simulation/mujoco/simulation.py:3611-3615
    def robot_joint_names(self, robot_name: str) -> list[str]:
        if self._world is None or not registered(self._world.robots, robot_name):
            return []                                                   # <<<<<<
        return list(self._world.robots[robot_name].joint_names)

Same pattern in newton/isaac/mjlab backends & robot_action_keys sibling.
"""
from strands_robots import Robot


def main() -> None:
    # User reads docs/robots/unitree_g1.md and uses the "g1" alias (line 10).
    robot = Robot("g1")

    # ----- PRE-FIX behavior (the footgun): -----
    # Line 18 verbatim: print(robot.robot_joint_names("unitree_g1"))
    # In v0.5.2 this returned [] with no stderr, no log, no raise.
    #
    # ----- POST-FIX behavior (this branch): -----
    # Raises ValueError naming the registered robot(s), with difflib hint.
    try:
        joints = robot.robot_joint_names("unitree_g1")
        print(f"[PRE-FIX FOOTGUN REPRODUCED] silent-empty joints={joints!r}")
        assert joints == [], "expected the silent-empty footgun on pre-fix tree"
    except ValueError as e:
        print(f"[POST-FIX OK] raised: {e}")
        assert "no robot named 'unitree_g1'" in str(e)
        assert "Registered: ['g1']" in str(e)

    # list_robots() always told the truth - but a reader of the unitree_g1
    # page has no reason to call it.
    assert robot.list_robots() == ["g1"]

    # Right call (name must EXACTLY match the string passed to Robot()):
    assert len(robot.robot_joint_names("g1")) == 30

    robot.cleanup()
    print("REPRO OK")


if __name__ == "__main__":
    main()
