"""Minimal repro for the README quickstart footgun.

The README's headline example at line 54 reads::

    from strands import Agent
    from strands_robots import Robot

    robot = Robot("so100")              # MuJoCo sim by default; mode="real" for hardware
    Agent(tools=[robot])("pick up the red cube")

A new user runs this verbatim and gets an agent that is asked to pick up a
"red cube" that **does not exist** in the default MuJoCo scene. The default
scene for ``Robot("so100")`` is a lone arm on the ground plane -- zero
manipulable objects, one external camera, no "red_cube" body in the world.

What the user actually gets:
    * ``world.objects == {}``
    * ``world.cameras == {'default': SimCamera(...)}`` (external view only,
      not the ``front``/``wrist`` cameras that language-conditioned policies
      want)
    * ``list_bodies`` returns 8 bodies, all of them robot links

The agent then flails -- it either tries to pick up something it has no
observation of, hallucinates coordinates, or (best case) emits the sim's
``add_object`` hint from ``simulation.py:1644,2237``. None of that is in the
README. The ``docs/start/first-robot.md`` page gets this right (move two
joints, save a camera frame -- no promise of manipulation on an empty table),
so the fix is to align the README snippet with reality.

The in-tree comment at ``strands_robots/simulation/mujoco/simulation.py:6591``
already acknowledges the problem::

    The README markets ``Robot("so100")`` as something you can drive with
    ``robot(action="...")``; without this method that contract raised
    ``TypeError: 'MuJoCoSimEngine' object is not callable``.

That TypeError was fixed. The empty-table footgun was not.

Run with the documented install (``uv pip install strands-robots[sim-mujoco]``)
on Python 3.12+::

    python readme_pick_red_cube_no_cube_in_scene_repro.py
"""
from __future__ import annotations

from strands_robots import Robot


def main() -> None:
    robot = Robot("so100")  # exact README line 53
    world = robot._world

    # Facts a user would need before "pick up the red cube" has any meaning:
    print(f"world.objects = {world.objects!r}")
    print(f"world.cameras = {sorted(world.cameras)!r}")

    bodies = robot(action="list_bodies")["content"][1]["json"]["bodies"]
    print(f"bodies ({len(bodies)}): {bodies}")

    assert world.objects == {}, "README footgun gone -- default scene now has objects"
    assert "front" not in world.cameras and "wrist" not in world.cameras, (
        "README footgun gone -- default scene now has a VLA-friendly camera"
    )
    assert not any("cube" in b.lower() for b in bodies), (
        "README footgun gone -- default scene now has a cube body"
    )

    print(
        "FOOTGUN CONFIRMED: README line 54 asks the agent to 'pick up the red cube' "
        "but the default scene contains zero objects, one external camera, and no "
        "cube body."
    )


if __name__ == "__main__":
    main()
