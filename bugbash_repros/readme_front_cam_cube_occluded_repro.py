"""Repro: README.md quickstart's 'front' camera cannot see the red_cube.

README.md (L48-58) tells a new user:

    robot = Robot("so100")
    robot(action="add_object", name="red_cube", shape="box",
          size=[0.025, 0.025, 0.025], position=[0.0, -0.20, 0.025],
          color=[1.0, 0.0, 0.0, 1.0])
    robot(action="add_camera", name="front", position=[0.0, -0.5, 0.3],
          target=[0.0, -0.2, 0.05])
    Agent(tools=[robot])("pick up the red cube")

and L63-65 prose explicitly promises:

    "the two action= calls above put a red cube in front of the gripper and
     add a 'front' camera so a language-conditioned policy has something to see."

Reality: at so100's zero pose the TCP sits at (0.001, -0.483, 0.096) --
closer to the 'front' camera (2 cm) than the cube (41 cm). The gripper body
occludes the cube. Rendering the camera the README told the user to add
produces 0 red pixels out of 307200 (640x480). The policy that was supposed
to "have something to see" sees a gripper and a floor plane.

Also broken in the DEFAULT external camera: a 2.5 cm cube at (0,-0.2,0.025)
projects to <36 px even uncovered; against a near-same-brightness floor
plane in the default lighting it emits 0 red pixels at (R>200,G<60,B<60).
Either the cube is 4x too small for the default view, or the floor plane
colour needs enough contrast for a 36-px target to register.

This is the second half of the fix that landed in commit 6329d9c
(doc: readme quickstart adds red cube + front camera before Agent call,
cagataycali/robots-harness#724). The scene has objects now, but the camera
geometry still makes the policy blind on the exact snippet in the README.

Run (no GPU, no network):

    cd <robots> && MUJOCO_GL=egl python bugbash_repros/readme_front_cam_cube_occluded_repro.py

Expected printout:

    README 'front' camera, 2.5cm red_cube: 0 red pixels (should be >= 30)
    DEFAULT external camera, 2.5cm red_cube:  0 red pixels (should be >= 30)
    FAIL: the camera the README told the user to add sees no cube.
"""
import os
import io
import sys
import base64

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot  # noqa: E402

try:
    from PIL import Image
    import numpy as np
except ImportError as exc:  # pragma: no cover
    sys.exit(
        "repro needs Pillow + numpy (both pulled by strands-robots[sim-mujoco]): "
        f"{exc}"
    )


def red_pixel_count(png_bytes: bytes) -> tuple[int, int]:
    im = np.array(Image.open(io.BytesIO(png_bytes)))
    red_mask = (im[:, :, 0] > 200) & (im[:, :, 1] < 60) & (im[:, :, 2] < 60)
    return int(red_mask.sum()), int(im.shape[0] * im.shape[1])


def render_bytes(robot: Robot, camera_name: str | None = None) -> bytes:
    kwargs: dict = {"action": "render"}
    if camera_name is not None:
        kwargs["camera_name"] = camera_name
    result = robot(**kwargs)
    assert result["status"] == "success", result
    for block in result["content"]:
        if "image" in block:
            data = block["image"]["source"]["bytes"]
            if isinstance(data, str):
                data = base64.b64decode(data)
            return data
    raise AssertionError("no image block in render response")


def main() -> int:
    # EXACT README snippet (minus the Agent() call, which is agent-SDK-driven)
    robot = Robot("so100")
    robot(action="add_object", name="red_cube", shape="box",
          size=[0.025, 0.025, 0.025], position=[0.0, -0.20, 0.025],
          color=[1.0, 0.0, 0.0, 1.0])
    robot(action="add_camera", name="front", position=[0.0, -0.5, 0.3],
          target=[0.0, -0.2, 0.05])

    front_red, total = red_pixel_count(render_bytes(robot, "front"))
    default_red, _ = red_pixel_count(render_bytes(robot))

    print(f"README 'front' camera, 2.5cm red_cube: {front_red}/{total} red pixels")
    print(f"DEFAULT external camera, 2.5cm red_cube: {default_red}/{total} red pixels")

    # README prose promises the policy has something to see. 30 pixels is a very
    # generous floor (a 2.5 cm cube at 0.4 m with 45 deg FOV projects to ~36 px).
    THRESHOLD = 30
    rc = 0
    if front_red < THRESHOLD:
        print(
            f"FAIL: README 'front' camera renders {front_red} red pixels (<{THRESHOLD}). "
            "At so100 zero pose the TCP (y=-0.483) is 2 cm in front of the camera "
            "(y=-0.5) and occludes the cube (y=-0.20)."
        )
        rc = 1
    if default_red < THRESHOLD:
        print(
            f"FAIL: default external camera renders {default_red} red pixels "
            f"(<{THRESHOLD}). The 2.5 cm cube is below the camera's useful "
            "angular size at that range + floor-plane contrast."
        )
        rc = 1

    robot(action="cleanup")
    return rc


if __name__ == "__main__":
    sys.exit(main())
