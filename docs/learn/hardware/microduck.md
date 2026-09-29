# Microduck

At the end of this page a Pollen Robotics Microduck walks on intents from `Robot("microduck", mode="real")`, you know why there is no per-joint write on this robot, and you know that the walking policy running on the duck is byte for byte the one you run in simulation.

This needs a duck on the network running `robotd`. No `port` is needed for the common case:

```python title="sketch"
from strands_robots import Robot

duck = Robot("microduck", mode="real")                        # finds robotd via $MICRODUCK_SOCKET, $MICRODUCK_HOST, /run/robotd.sock
duck = Robot("microduck", mode="real", port="ssh://radxa@duck.local")   # explicit ssh forward of the socket
print(duck.connect_eagerly())
duck.send_action({"vx": 0.1, "vyaw": 0.0})                   # a twist intent, m/s and rad/s
```

## How the socket is found

| source | meaning |
|---|---|
| `$MICRODUCK_SOCKET` | a local unix socket path, one you forwarded yourself |
| `$MICRODUCK_HOST` | `[user@]host`; the driver runs `ssh -N -L` and forwards `robotd`'s socket. User defaults to `radxa`; `$DUCK_BOARD_USER` is honoured as `duckctl` does |
| `/run/robotd.sock` | the answer when this code runs on the duck |
| `port="ssh://[user@]host"` | the same forward, named explicitly |

## Intents, not joints

`robotd` owns the 50 Hz control loop and runs the walking and skill policy on the board. Its JSON-RPC 2.0 surface (`duck-ipc-proto`, NDJSON over the socket) has no per-joint write. The whole `robot.*` surface is intent-level, so `mode="real"` is delegate-only by the robot's own design:

| intent | wire | kind |
|---|---|---|
| `robot.move` (twist), `robot.head`, `robot.pose`, `robot.mouth` | notification, no reply | continuous |
| `robot.do` (skills), `robot.enable`, `robot.relax`, `robot.init`, `robot.stop` | request, reply awaited | discrete |
| `robot.state` | request | read |

`send_action` accepts `vx`, `vy`, `vyaw` (twist), `neck_pitch`, `head_pitch`, `head_yaw`, `head_roll` (radians), `z`, `roll`, `pitch`, `active` (standing pose), `open` (mouth, 0 to 1) and `skill` (one of `SKILLS`, sent as `robot.do`). `run_policy` and `start_task` refuse and name the intent path; the driver does not pretend to stream 14 joint targets to a wire that has no method for them.

`read_state` publishes 14 joints, the locomotion set the policy speaks. `robotd`'s own `JOINT_NAMES` is 15 wide with `mouth` at index 9; the driver drops it, and the mouth travels through `robot.mouth`.

## Sim to real

The on-robot policy is the same `alpha_walking.onnx` the `[microduck]` extra runs in MuJoCo (byte-compatible, difference 0.0), so a sim rollout with equal observations predicts the hardware. Weights come from the Hub at `pollen-robotics/microduck-policies`:

```bash
pip install 'strands-robots[microduck,sim-mujoco]'     # onnxruntime + huggingface_hub for the policy, MuJoCo for the twin
```

```python title="sketch"
sim = Robot("microduck")
sim.run_policy(
    robot_name="microduck",
    policy_provider="microduck",
    policy_config={"onnx_path": "alpha_walking.onnx"},
    instruction="walk forward",
)   # the provider self-configures from the ONNX metadata
```

## Camera

The driver reads the duck's camera through `robotd` and returns one frame as an image the agent can see; `cv2` and `numpy` load on that call only.

## Safety

The duck is velocity-commanded: it keeps walking until the twist is zeroed or `robot.stop` is sent. `stop()` cancels any held move and sends `robot.stop`; `cleanup()` closes the socket and the ssh forward. Whether `robotd` times out a twist when intents stop arriving is the daemon's behaviour, not this driver's.

<robot-viewer name="microduck"></robot-viewer>
