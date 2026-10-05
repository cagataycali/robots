"""Repro: add_camera and add_robot dup refusals drift from the fixed add_object sibling.

Post-#752 (commit 0f622f27, PR #4532) add_object dup refusal reads
    "add_object: object 'X' already exists. Remove it first (remove_object)."
on all three backends -- verb prefix + 'already' + 'Remove it first' + helper name.

The sibling scene mutators in the SAME class/file never followed:

mujoco (strands_robots/simulation/mujoco/simulation.py)
    add_camera (L5662): "add_camera: camera 'X' already exists. Remove it first."
                        -- no (remove_camera) helper hint
    add_robot  (L2518): "Robot 'X' already exists. Pick a different name, or
                         omit name= to auto-number. Existing: ..."
                        -- no add_robot: prefix; different remedy; no helper hint

newton (strands_robots/simulation/newton/simulation.py)
    add_camera (L1734): same as mujoco (no helper hint)
    add_robot  (L678):  "Robot 'X' already exists."            <- bare

isaac (strands_robots/simulation/isaac/simulation.py)
    add_camera (L8207): "Camera 'X' already exists."           <- bare
    add_robot  (L3613): "Robot 'X' already exists."            <- bare

Both remove_camera and remove_robot EXIST in every backend
(mujoco:4094 advertises remove_camera; mujoco:3455 defines remove_robot;
newton + isaac have matching remove_* siblings), so the helper-name hint
is a factual nudge, not a wish.

Surface:
A README-quickstart user who re-runs the 3-line snippet in the main README --
    robot.add_camera(name="front", ...)
    robot.add_camera(name="front", ...)   # typo second line, same name
    robot.add_robot(name="so100")
hits the shortest refusals of the three sibling add_* methods and has to grep
the source for the inverse call name on each backend.
"""

from strands_robots import Robot


def _tap(d: dict) -> str:
    return d["content"][0]["text"]


def main() -> None:
    r = Robot("so100")  # mujoco default

    r.add_object(name="red_cube", shape="box", size=[0.05] * 3,
                 position=[0.0, -0.2, 0.025], color=[1.0, 0.0, 0.0])
    r.add_camera(name="front", position=[0.3, -0.7, 0.45], target=[0.0, -0.2, 0.03])

    obj_dup = _tap(r.add_object(name="red_cube", shape="box", size=[0.05] * 3,
                                position=[0.0, -0.2, 0.025], color=[1.0, 0.0, 0.0]))
    cam_dup = _tap(r.add_camera(name="front", position=[0.3, -0.7, 0.45],
                                target=[0.0, -0.2, 0.03]))
    rob_dup = _tap(r.add_robot(name="so100"))

    print("add_object dup :", repr(obj_dup))
    print("add_camera dup :", repr(cam_dup))
    print("add_robot  dup :", repr(rob_dup))

    # Three asymmetries the fix pins:
    assert "(remove_object)" in obj_dup, "add_object regression"
    assert "(remove_camera)" in cam_dup, (
        f"add_camera dup should name remove_camera; got {cam_dup!r}"
    )
    assert rob_dup.startswith("add_robot:"), (
        f"add_robot dup should start with 'add_robot:' prefix; got {rob_dup!r}"
    )
    assert "(remove_robot)" in rob_dup, (
        f"add_robot dup should name remove_robot; got {rob_dup!r}"
    )
    print("\nPARITY OK")


if __name__ == "__main__":
    main()
