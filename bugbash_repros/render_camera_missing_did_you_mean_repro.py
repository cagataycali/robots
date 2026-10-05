"""
Repro: Robot("so100").render(camera_name=<typo>) refuses with "Available: [...]"
       but omits the "Did you mean" + "action='list_cameras'" hint that every
       sibling method on the same object (remove_camera / remove_object /
       move_object / remove_robot) emits.

The README quickstart (README.md L47-54) teaches a user to:

    robot = Robot("so100")
    robot.add_camera(name="front", position=[0.3, -0.7, 0.45], target=[0, -0.2, 0.03])
    Agent(tools=[robot])("pick up the red cube")

The agent then sees a tool whose state names camera 'front' and will almost
certainly try to render through it. A one-character typo on the LLM side
("fronf", "frontcam", "camera_front") hits a stone-wall message with no
suggestion, while the exact-matched helper (close_match_hint in
strands_robots/simulation/base.py:304) is wired into every sibling path.

Expected: parity with remove_camera's message
    Camera 'fronf' not found. Did you mean: front? Available: ['default', 'front']. Use action='list_cameras' to see all.

Actual (render):
    Camera 'fronf' not found. Available: ['default', 'front']

Upstream file:line (repro-pin):
    strands_robots/simulation/mujoco/rendering.py:1503  (robot.render)
    strands_robots/simulation/mujoco/rendering.py:1623  (render_depth)
    strands_robots/simulation/mujoco/rendering.py:1829  (image/numpy render)
    strands_robots/simulation/mujoco/rendering.py:2066  (sibling render helper)
    strands_robots/simulation/mujoco/rendering.py:2516/2765/3406  (batch forms,
        same pattern - "Camera(s) not found: {unresolved}. Available: {...}")

The reusable helper exists:
    strands_robots/simulation/base.py:304  close_match_hint(requested, known)
and is used by remove_camera / remove_object / move_object / remove_robot in
scene_ops.py.

Reproduces on main @ 7789f6b (2026-11-25) with env MUJOCO_GL=egl.
"""
from __future__ import annotations

from strands_robots import Robot


def main() -> None:
    robot = Robot("so100")
    robot.add_camera(
        name="front",
        position=[0.3, -0.7, 0.45],
        target=[0.0, -0.2, 0.03],
    )

    # Sibling - has "Did you mean" + "action='list_cameras'" hint
    sibling = robot.remove_camera(name="fronf")
    print("remove_camera (sibling, has hint):")
    print(" ", sibling["content"][0]["text"])

    # Re-add for the render test
    robot.add_camera(
        name="front",
        position=[0.3, -0.7, 0.45],
        target=[0.0, -0.2, 0.03],
    )

    # Render - no hint
    rendered = robot.render(camera_name="fronf")
    print("render (defect, no hint):")
    print(" ", rendered["content"][0]["text"])

    # Pin the shape of the defect so an automated run can grade it
    render_text = rendered["content"][0]["text"]
    assert "Did you mean" not in render_text, (
        "If this fires, the defect is fixed - update the repro."
    )
    assert "list_cameras" not in render_text, (
        "If this fires, the defect is partially fixed - update the repro."
    )


if __name__ == "__main__":
    main()
