# Design: whole-body teleoperation

A humanoid demonstration is not a leader arm. The operator wears a headset, the robot follows the head and both wrists while a whole-body controller fills in the legs, and the dataset that comes out carries a 64-D latent action rather than joint targets. After this page you know how that path is shaped in Strands Robots, which pieces exist today, which are optional and why, and what the recorded dataset looks like.

## What the reference stack does

The LeRobot humanoid post records Unitree G1 episodes at 50 fps: state is 29 joints plus two grippers (31 columns named `kLeftHipPitch.q` ... `right_gripper`), action is 64 SONIC motion tokens plus two grippers (`motion_token_0` ... `right_gripper`), and three 480x640 cameras. The operator's headset streams head and wrist poses (and, in one mode, tracked ankles); the streamer converts them to SMPL body poses and NVIDIA's SONIC encoder turns each frame into a token; the SONIC decoder on the robot turns tokens plus proprioception history into joint targets at 50 Hz.

Two facts shape the design. LeRobot itself ships only the decode half; the headset, SMPL and encoder path lives in NVIDIA's `gear_sonic` stack and speaks msgpack over ZMQ (protocol v1: 29 joint targets; v3: joints plus SMPL; v4: a finished token). And the SMPL body model files are licensed per user and cannot be redistributed, so an SMPL stage can only ever be optional.

## The seam

`strands_robots.teleop` is three small protocols and one class that composes them:

| stage | protocol | what it yields | shipped now |
|---|---|---|---|
| source | `PoseSource.latest()` | a `PoseFrame`: joints, gripper fractions, tracked poses, mode | `MockPoseSource` (scripted arcs) |
| retarget | `Retarget(frame)` | follower joint targets in radians, plus an encoder reference | `JointMapRetarget` (rename, scale, offset) |
| encoder | `Encoder.encode(out)` | 64 floats, or absent | none; the protocol and a token layout |
| layout | `teleop_layout(name)` | the state and action column names a recorder writes | four layouts |

`WholeBodyTeleoperator(source, retarget, encoder, layout)` duck-types a lerobot `Teleoperator` (`connect`, `disconnect`, `is_connected`, `get_action`, `action_features`), which is the surface `attach_teleop` already drives. The follower's existing 50 Hz `teleoperate()` loop therefore runs a headset the same way it runs a leader arm, with the same slew bound and the same status line.

```python title="a scripted operator drives the simulated G1"
from strands_robots import Robot
from strands_robots.teleop import G1_SIM_JOINTS, JointMapRetarget, MockPoseSource, WholeBodyTeleoperator

g1 = Robot("g1")
device = WholeBodyTeleoperator(
    MockPoseSource(period_s=2.0),
    JointMapRetarget({name: name for name in G1_SIM_JOINTS}),
    layout="g1_joint_29",
)
g1.attach_teleop(device, name="mock")
g1.teleoperate(hz=50, duration=2.0, block=True)
print(g1.get_teleoperate_status()["content"][0]["text"])
```

Units inside the package are radians, metres and seconds; grippers are open fractions in `[0, 1]`. A source converts at its own edge. The ZMQ wire lists joints in IsaacLab breadth-first order while the dataset, the hardware and our MuJoCo model use the depth-first order; `ISAACLAB_TO_HARDWARE` is that permutation, applied once by the source so nothing downstream sees two orders.

## Layouts

| layout | state | action | consumer |
|---|---|---|---|
| `g1_joint_29` | 29 MuJoCo actuator names | the same 29 | the simulated G1, the native driver |
| `blog_31_66` | `kLeftHipPitch.q` ... `right_gripper` | `motion_token_0` ... `right_gripper` | the post's dataset and checkpoint |
| `blog_31_31` | as above | 29 `k<Joint>.q` plus two grippers | a joint-target policy, no encoder |
| `lerobot_token_64` | `motion_token_state.{i}.pos` | `motion_token.{i}.pos` | lerobot's SONIC controller |

The dataset spells tokens with underscores and lerobot's controller with dots; a layout makes the choice a named table rather than something a recorder derives. `blog_31_66` is checked against the published dataset's feature names, column for column.

## What lands next

1. `ZmqPoseSource`: a subscriber for NVIDIA's own protocol, so their PICO streamer is a Strands input device with no hardware code of ours, and any mocap or simulator publisher is too.
2. `SonicEncoder`: `model_encoder.onnx` and its observation config from `nvidia/GEAR-SONIC`, downloaded behind the trust gate. The weights are under the NVIDIA Open Model License; the code we write from the config is ours. A token is only guaranteed to decode with the decoder from the same folder.
3. A recorder bridge: an observer the teleop loop calls after each accepted frame, writing state, action and cameras at the dataset rate.
4. `ThreePointRetarget` through the G1 IK already in the package, and `SmplRetarget` that refuses cleanly when no SMPL file is configured.

The simulated G1 is the follower for all of it: joint targets from the retarget stage drive MuJoCo while the token is recorded, so the whole path is exercised before a robot is switched on. On hardware every write still passes the G1 driver's motion gates.

## Verified today

On this Mac in MuJoCo: the scripted source runs at 50 Hz through `teleoperate()` with no errors and no slew rejections; a recording made through the seam verifies as one LeRobot v3 episode at 50 fps with a 29-column action; the `blog_31_66` layout equals the published dataset's 31 and 66 names. The tests live beside the code under `tests/teleop/`.
