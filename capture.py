"""Record the agent-visible tool surface and the cmd_vel wire for one session."""
from __future__ import annotations
import json, pathlib, sys
from typing import Any

import strands_robots.mesh.ros_bridge as ros_mod
import strands_robots.mesh.rosbridge_robot as rbr_mod
import strands_robots.mesh.rtps_robot as rtps_mod
from strands_robots.mesh import RosBridgedRobot, RosbridgeRobot, RtpsRobot

TREE = str(pathlib.Path(ros_mod.__file__).parents[2])
print("TREE:", TREE)


class Rec:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(kw)
        return {"status": "success", "content": [{"text": "ok"}]}


out: dict[str, Any] = {"tree": TREE, "session": [], "transports": {}}

# ---- one realistic agent session on the ROS 2 bridge ----
rec = Rec()
ros_mod.use_ros = rec  # type: ignore[assignment]
robot = RosBridgedRobot.from_ros(node_name="rover", cmd_vel_topic="/cmd_vel", odom_topic="/odom")
tools = {t.tool_name: t for t in robot.tools}
out["ros2_tools"] = sorted(tools)
out["has_stop_tool"] = "stop_rover" in tools

# step 1 - the agent drives forward (no duration -> latching)
rec.calls.clear()
drive: Any = tools["drive_rover"]
drive(linear=0.5)
out["session"].append({
    "step": "drive_rover(linear=0.5)", "tool": "drive_rover",
    "published": [{"linear_x": c["fields"]["linear"]["x"], "count": c["count"]} for c in rec.calls],
})

# step 2 - the agent tries to halt through its tool surface
rec.calls.clear()
if "stop_rover" in tools:
    stop: Any = tools["stop_rover"]
    stop()
    out["session"].append({
        "step": "stop_rover()", "tool": "stop_rover",
        "published": [{"linear_x": c["fields"]["linear"]["x"], "count": c["count"]} for c in rec.calls],
    })
    out["halt_tool"] = "stop_rover"
else:
    out["session"].append({"step": "halt: no stop tool in the surface", "tool": None, "published": []})
    out["halt_tool"] = None

# the public method exists on both trees either way
rec.calls.clear()
robot.stop()
out["stop_method_wire"] = [{"linear_x": c["fields"]["linear"]["x"], "count": c["count"]} for c in rec.calls]

# ---- cross-transport surface ----
rec2 = Rec()
rbr_mod.use_rosbridge = rec2  # type: ignore[assignment]
rb = RosbridgeRobot(node_name="rover", cmd_vel_topic="/cmd_vel", odom_topic="/odom")
rec3 = Rec()
rtps_mod.use_rtps = rec3  # type: ignore[assignment]
rt = RtpsRobot.from_rtps(node_name="rover", cmd_vel_topic="/cmd_vel")
for label, rb_obj in (
    ("ROS 2 (rclpy)", robot),
    ("rosbridge (websocket)", rb),
    ("RTPS (cyclonedds)", rt),
):
    names = sorted(t.tool_name for t in rb_obj.tools)
    out["transports"][label] = {
        "tools": names,
        "stop_tool": any(n.startswith("stop_") for n in names),
        "stop_method": callable(getattr(rb_obj, "stop", None)),
    }

dest = pathlib.Path(sys.argv[1])
dest.write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ("has_stop_tool", "halt_tool", "ros2_tools")}, indent=2))
