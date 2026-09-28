# Franka Panda and FR3

At the end of this page a Franka Emika Panda or a Franka Research 3 accepts joint-space motion from `Robot("panda", mode="real")` through the Franka Control Interface, you know the joint names this package uses for each arm and why, and you know which things the driver leaves to libfranka on purpose.

This needs an arm with FCI enabled in Desk, a workstation on the arm's network, and `panda-py`:

```bash
pip install panda-py              # the panda-py binding over libfranka (MIT); not a strands-robots extra
```

```python title="sketch"
from strands_robots import Robot

arm = Robot("panda", mode="real", driver="strands", port="172.16.0.2")   # FrankaDriver; a bare call resolves to lerobot
print(arm.connect_eagerly())                                      # None, or why not
arm.send_action({f"joint{i}": 0.0 for i in range(1, 8)})         # all seven joints, radians
```

## Joint names

FCI carries an unnamed seven-element vector. The names this driver uses are the names in each arm's own MuJoCo asset, so an action dict authored against `Robot("panda")` in simulation commands the real arm unchanged:

| robot | joints | gripper key |
|---|---|---|
| `panda` | `joint1` ... `joint7` | `gripper_width` |
| `fr3` | `fr3_joint1` ... `fr3_joint7` | `gripper_width` |
| `fr3_v2` | `fr3v2_joint1` ... `fr3v2_joint7` | `gripper_width` |

A joint-space command must name all seven joints with finite values; a partial dict or a key from the wrong arm is refused before libfranka sees it.

## What the driver does

- `connect_eagerly()` resolves `panda_py`, opens the FCI link and the Franka Hand. Off hardware it returns a reason string and leaves the driver usable: reads return their cache, writes refuse "not connected".
- State: joint positions, velocities and link-side torques from one libfranka `RobotState` read, gripper width from the Hand. The arm sources state at 1 kHz (`FCI_RATE_HZ`); `downsample_stride` reports the ratio to the consumer's rate so a mesh publisher does not try to read every tick.
- `send_action` hands the target to `panda_py`'s guarded motion generator with `speed_factor` (default `0.2`, a fifth of the arm's maximum), which owns the realtime loop and enforces the arm's limits.
- `stop()` halts motion and reports.

## What it does not do, on purpose

| absent | why |
|---|---|
| a 1 kHz control loop, torque control | a Python thread that misses a tick triggers a reflex stop; libfranka's realtime context owns that loop |
| a joint-limit table | the Panda and the FR3 differ and this repo has neither to measure; libfranka refuses an out-of-envelope target and the refusal is reported verbatim |
| Cartesian control, kinematics | `O_T_EE` is on the state but not published as `_pose`, because the mesh's `_pose` is a base pose |
| `start_task`, `run_policy` | no policy provider emits Franka-shaped actions yet; both refuse and say so |

## Reading the arm from an agent

```python title="sketch"
from strands import Agent

agent = Agent(tools=[arm])
agent("Read the joint positions and the measured torques. Do not move.")
```

The tool's read verbs are ungated. A motion verb pauses for approval (see [agents](../agents.md)).

## Simulation

`Robot("panda", keyframe="home")`, `Robot("fr3")` and `Robot("fr3_v2")` build the MuJoCo twins from the menagerie assets. Because the joint names match, a policy or a recorded episode moves between the two modes without a remap.
