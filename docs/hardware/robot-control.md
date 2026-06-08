---
description: HardwareRobot — async task execution, status reporting, the LeRobot bridge.
---

# Robot control (real hardware)

`Robot(name, mode="real", ...)` returns a `strands_robots.hardware_robot.Robot`. This
class wraps a LeRobot `Robot` instance with async task execution, status reporting,
and an `AgentTool` action surface so policies and agents can drive real hardware.

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

# Start a task - returns immediately (async)
robot.start_task(
    instruction="pick up the cube",
    policy_provider="groot",
    policy_port=5555,
    duration=30.0,
)

# Poll task state
status = robot.get_task_status()

# Stop early if needed
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
| `cameras` | Mapping of camera name -> config dict. |
| `action_horizon` | Number of actions per inference step (used by chunk-based policies). |
| `data_config` | GR00T `data_config` name. |
| `control_frequency` | Control loop frequency in Hz. Default 50. |
| `**kwargs` | Forwarded to the LeRobot backend (`port`, `robot_ip`, `kp`, `kd`, `controller`, `calibration_dir`, etc.). Unknown kwargs raise `ValueError`. |

Constructed via the top-level factory: `Robot("so100", mode="real", cameras={...}, port=...)`.

## Task lifecycle

`HardwareRobot` has explicit task state - important on real hardware where you don't
want to run two policies at once:

```python
class TaskStatus(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"
```

Methods:

- `start_task(instruction, policy_port=None, policy_host='localhost', policy_provider='groot', duration=30.0)` -
  async; spawns the policy thread and returns immediately. Sets status `CONNECTING` -> `RUNNING`.
- `stop_task()` - halts the running policy. Sets status `STOPPED`.
- `get_task_status()` - returns a `RobotTaskState` with current `TaskStatus`, step count,
  and error message if any.
- `cleanup()` - stop tasks, shutdown executor, close cameras, stop mesh.

## AgentTool actions

When used through a Strands `Agent`, the tool dispatcher exposes four actions:

| Action | Blocking? | What |
|--------|-----------|------|
| `execute` | Yes | Start and wait for completion. Needs `instruction` + `policy_port`. |
| `start` | No | Start in background. Needs `instruction` + `policy_port`. |
| `status` | - | Return current task status JSON. |
| `stop` | - | Stop the running task. |

## Camera ingestion

Cameras passed via the `cameras` kwarg flow through to LeRobot's camera abstractions.
Per-camera config:

```python
cameras = {
    "wrist": {
        "type": "opencv",
        "index_or_path": "/dev/video0",
        "fps": 30,
        "width": 640,
        "height": 480,
    },
    "front": {
        "type": "realsense",
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

## Mesh teleop

`HardwareRobot` supports cross-machine teleoperation over the Zenoh mesh:

```python
# On the publisher side (leader machine)
robot.start_teleop_publish(teleoperator, device_name="leader", method="joint", hz=50)

# On the receiver side (follower machine)
robot.start_teleop_receive(
    source_peer_id="leader-abc123",
    device_name="follower",
    apply_fn=my_apply_fn,
)

# Check status
robot.get_teleop_status()

# Stop a named session, or all sessions if device_name omitted
robot.stop_teleop(device_name="follower")
robot.stop_teleop()
```

See [Tutorial 5 - Multi-robot](../tutorial/05-multi-robot.md) for the full cross-machine
workflow using `InputPublisher` / `InputReceiver` from `strands_robots.mesh`.

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
| Recording | `start_recording` action | Same - DatasetRecorder works for both |
| Reset | `reset()` brings sim back to t=0 | Holds current pose; no rewind |
| Randomization | `randomize(...)` | N/A |
| Step | `step()` advances physics | N/A - real time |
| Policy execution | `run_policy()` / `start_policy()` | `start_task()` / `execute` action |

The action vocabulary they share is the same - that's what makes `Agent(tools=[robot])`
agnostic to which one you constructed.

## See also

- [Tutorial 8 - Real hardware](../tutorial/08-real-hardware.md) - bring-up checklist.
- [Hardware tools](tools.md) - calibrate / camera / teleop / pose / serial helpers.
- [Robot factory](../getting-started/robot-factory.md) - every `Robot()` kwarg.
- [Policy providers](../policies/overview.md) - available policy providers.
