# Design: driver composition

A humanoid with add-on hardware is several transports at once. The reference G1 speaks DDS for its 29 joints, its two OpenArm grippers speak CAN-FD from a backpack adapter, and its head camera is a video device. An agent, the mesh, the recorder and the operator gate all want one robot. After this page you know how several drivers present as one, which rules the composite enforces, and what was proved in simulation.

## Why the bimanual mechanism is not enough

Today two arms compose inside lerobot: a `bi_*` follower builds two arms from `left_arm_config` and `right_arm_config` and prefixes their features `left_` and `right_`. That works for two arms of one family behind one lerobot class. It cannot hold a native DDS driver next to a CAN gripper and a camera, and the registry has no way to say that three entries are one robot. The simulator can put several robots in one scene, but a scene is not a robot: each keeps its own name, features and stop.

## The registry shape

```json title="one robot, four parts"
"g1_openarm": {
  "category": "humanoid",
  "hardware": {"driver": "strands", "composite": true},
  "parts": {
    "body": "unitree_g1",
    "left_gripper": "openarm_gripper@can0",
    "right_gripper": "openarm_gripper@can1",
    "head": "camera@/dev/video0"
  },
  "primary": "body",
  "stop_order": ["body", "left_gripper", "right_gripper", "head"]
}
```

Each part is a registry robot or device with an optional `@port`, the same polymorphic `port=` every driver takes. `Robot("g1_openarm", mode="real")` builds each part through the factory it already has, with the mesh off, and wraps them in one `CompositeDriver`. Parts never become tools or peers themselves.

## The rules

**One key space.** The `primary` part's joints keep their names. Every other part's joints are prefixed `<part>_`. A one-joint part whose joint is `gripper` collapses to the part name, so `left_gripper` and `right_gripper` fall out of the registry names with no table, and the composite's 31 keys line up with the published dataset's 31 state columns.

**Units per part.** Each part declares a unit per joint (`rad`, `norm01`, `deg`, `m`). The composite reports values as the part gave them and never rescales; `features` answers key to unit so a consumer reads rather than guesses. An OpenArm gripper reads degrees in `(-65, 0)`; its part converts to an open fraction.

**One write.** A target for a key no part owns refuses the whole write and names the valid keys. Parts are written sequentially, primary first, so a gripper never closes on a pose the body did not reach. A part that refuses stops every part, and the refusal says so.

**One stop.** `stop()` runs `stop_order` (body first: it is the part that can hurt someone; grippers next so an object is released after the arm has stopped; sensors last), each part within its own timeout, and reports how many halted. A partial stop latches: motion is refused with the part's name until `reset_estop()`, which itself refuses while a part is disconnected.

**Reads never latch.** `get_observation()` returns every key it could read plus a `parts` section with each part's connection state, so a dashboard shows the dead part instead of a frozen number.

**One gate, one peer, one recorder.** The composite is the tool the operator gate keys on, the object the mesh wraps (it answers `get_observation` and `is_connected`, which is what the joint telemetry reader looks for), and the feature space the recorder declares.

```python title="a simulated body and two mock grippers, one robot"
from strands_robots import Robot
from strands_robots.drivers.composite import CompositeDriver, MockGripperPart, SimPart

g1 = Robot("g1")
robot = CompositeDriver(
    "g1_openarm",
    parts=[SimPart(g1, "g1", "body"), MockGripperPart("left_gripper"), MockGripperPart("right_gripper")],
    primary="body",
    stop_order=["body", "left_gripper", "right_gripper"],
)
print(len(robot.joint_names), robot.joint_names[-2:])          # 31 ('left_gripper', 'right_gripper')
print(robot.send_action({"left_shoulder_pitch_joint": 0.3, "left_gripper": 0.3})["content"][0]["text"])
print(robot.stop()["content"][0]["text"])                       # stopped 3/3 parts
```

## Recording the reference dataset

| column | width | parts |
|---|---|---|
| `observation.state` | 31 | body joints, then the two grippers |
| `action` | 66 with an encoder, 31 without | tokens or body joints, then the grippers |
| `observation.images.ego_view` | 480x640x3 | the head part |
| `observation.images.left_wrist`, `right_wrist` | 480x640x3 | the body's wrist cameras |

The column names come from a layout in `strands_robots.teleop`, so the composite's key space and the dataset's spelling stay two separate tables that are checked against each other.

## What lands next

1. `DriverPart` over any existing driver, and `CameraPart` over a lerobot camera, with the registry loader accepting `parts`.
2. An `openarm_gripper` entry: a one-motor lerobot config on the CAN bus, so no new transport code.
3. The token decoder slot on the body part, where lerobot's SONIC controller runs inside the G1 driver's control loop; until then a token action is refused with a sentence that says which part has no decoder.
4. Policy rollouts over a composite, gated once per rollout as today.

## Verified today

On this Mac in MuJoCo, a G1 body plus two mock grippers behind one `CompositeDriver`: 31 keys in the promised order; one `send_action` reaches all three parts and one read returns them; a mistyped key refuses with nothing written; a gripper that fails to halt yields `stopped 2/3 parts`, latches, and the next write is refused by name until `reset_estop()`; a disconnected gripper refuses motion while reads still answer with its status; the agent verbs dispatch and an unknown verb is refused. The tests live under `tests/drivers/`.
