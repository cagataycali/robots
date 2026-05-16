---
description: HardwareRobot — async task execution, status reporting, the LeRobot bridge.
---

# Robot control (real hardware)

`Robot(name, mode="real", ...)` returns a `strands_robots.hardware_robot.Robot`. This
class wraps a LeRobot `Robot` instance with async task execution, status reporting,
and the same `Simulation`-style action surface so policies and agents work
identically across sim and real.

## TL;DR

```python
from strands_robots import Robot

robot = Robot(
    "so100",
    mode="real",
    cameras={"wrist": {"type": "opencv", "index_or_path": "/dev/video0"}},
    port="/dev/tty.usbserial-A50285BI",
    control_frequency=50.0,
)

# Same surface as a Simulation
robot.run_policy(
    instruction="pick up the cube",
    policy_provider="lerobot_local",
    pretrained_name_or_path="lerobot/pi0_so100",
    duration=15.0,
)

# Or async
robot.start_task(instruction="organize the table", policy_provider="mock")
status = robot.get_task_status()
robot.stop_task()
```

## Class layout

`HardwareRobot` is in `strands_robots/hardware_robot.py`:

```python
class Robot(AgentTool):
    def __init__(self,
                 tool_name: str,
                 robot: LeRobotRobot | RobotConfig | str,
                 cameras: dict[str, dict[str, Any]] | None = None,
                 action_horizon: int = 8,
                 data_config: str | Any | None = None,
                 control_frequency: float = 50.0,
                 **kwargs: Any) -> None: ...
```

| Param | What |
|-------|------|
| `tool_name` | Tool identifier the agent uses. |
| `robot` | Either a LeRobot `Robot` instance, a `RobotConfig`, or a string the LeRobot factory understands (`"so100"`, `"koch"`, etc.). |
| `cameras` | Mapping of camera name → config dict. |
| `action_horizon` | Number of actions per inference step (used by chunk-based policies). |
| `data_config` | GR00T `data_config` name. |
| `control_frequency` | Control loop frequency in Hz. Default 50. |

## Task lifecycle

`HardwareRobot` has explicit task state — important on real hardware where you don't
want to run two policies at once:

```python
class TaskStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"
```

Methods:

- `run_policy(...)` — synchronous; blocks until done. Sets status `RUNNING` →
  `COMPLETED`.
- `start_task(...)` — async; spawns the policy thread and returns immediately. Sets
  status `RUNNING`.
- `stop_task()` — halts the running policy. Sets status `STOPPED`.
- `get_task_status()` — current `TaskStatus`, step count, error message if any.

The task state is exposed over the mesh (`strands/{peer_id}/state`) so other peers can
monitor remotely. See [Tutorial 5 — Multi-robot](../tutorial/05-multi-robot.md).

## Camera ingestion

Cameras passed via the `cameras` kwarg flow through to LeRobot's camera abstractions.
Per-camera config:

```python
cameras = {
    "wrist": {
        "type": "opencv",                      # backend
        "index_or_path": "/dev/video0",
        "fps": 30,
        "width": 640,
        "height": 480,
    },
    "front": {
        "type": "realsense",                   # Intel RealSense
        "serial": "123456789",
        "fps": 30,
    },
}
```

The supported `type` values are whatever your installed LeRobot version supports.
Check with:

```python
from lerobot.cameras import CameraConfig
print(CameraConfig.__subclasses__())
```

## Cleanup

```python
robot.cleanup()       # stop tasks, shutdown executor, close cameras, stop mesh
```

`HardwareRobot.cleanup()` is also called from `__del__`, but explicit calls are
preferred so any errors are observable.

## Mesh attachment

Like `Simulation`, every `HardwareRobot` joins the Zenoh mesh by default:

```python
robot = Robot("so100", mode="real")
print(robot.mesh.peer_id)   # 'so100-...'
print(robot.mesh.alive)     # True
```

Disable per-robot with `Robot("so100", mode="real", mesh=False)` or process-wide with
`STRANDS_MESH=false`.

## Difference from Simulation

| Feature | Simulation | HardwareRobot |
|---------|------------|---------------|
| Joint control | MuJoCo `data.ctrl` | LeRobot servo writes |
| Cameras | `add_camera` post-construction | `cameras=` kwarg at construction |
| Recording | `start_recording` action | Same — DatasetRecorder works for both |
| Reset | `reset()` brings sim back to t=0 | Holds current pose; no rewind |
| Randomization | `randomize(...)` | N/A |
| Step | `step()` advances physics | N/A — real time |

The action vocabulary they share is the same — that's what makes `Agent(tools=[robot])`
agnostic to which one you constructed.

## See also

- [Tutorial 8 — Real hardware](../tutorial/08-real-hardware.md) — bring-up checklist.
- [Hardware tools](tools.md) — calibrate / camera / teleop / pose / serial helpers.
- [Robot factory](../getting-started/robot-factory.md) — every `Robot()` kwarg.
- [Policy providers](../policies/overview.md) — what to feed `run_policy`.
