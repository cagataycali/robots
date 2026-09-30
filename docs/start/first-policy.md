---
description: "Stage 3: one Hub checkpoint, one run_policy call, the simulated SO-101 here and the physical one with a two-line change."
---

# Same checkpoint: sim or real

By the end of this page SmolVLA, a vision-language-action model from the Hub, has driven the simulated SO-101 from two cameras and your sentence, you hold the same call for the physical arm, and you have watched the gate stop that call naming the checkpoint. The sim fence runs on a laptop with no GPU after an 865 MB download.

## One checkpoint, one call

`lerobot/smolvla_base` reads an instruction, `observation.state` and three camera images, and returns a chunk of `action` vectors. It ships no SO-101 statistics, so the embodiment is written inline in native units: the sim's joint names as `state_keys` and `action_keys`, `dim_policy` padding six joints to the width the model expects, `obs_rename` routing your camera names onto the three it declares (`default` fills the one you did not mount):

```python
import os
os.environ["STRANDS_TRUST_REMOTE_CODE"] = "1"
from strands_robots import Robot
from strands_robots.policies import create_policy

robot = Robot("so101")
robot.add_camera(name="front", position=[0.22, 0.025, 0.6], target=[0.22, 0.025, 0])
robot.add_camera(name="wrist", parent_body="so101/gripper", position=[0.058, 0.0, -0.029], target=[-0.024, 0.0, -0.297])
joints = robot.robot_joint_names("so101")
embodiment = {"state_keys": joints, "action_keys": joints, "dim_policy": "pad", "obs_rename": {
    "front": "observation.images.camera1", "wrist": "observation.images.camera2", "default": "observation.images.camera3"}}
policy = create_policy("lerobot_local", pretrained_name_or_path="lerobot/smolvla_base", embodiment=embodiment)
result = robot.run_policy(policy_object=policy, instruction="pick up the cube", duration=3.0, control_frequency=30.0)
print(result["status"])
print(result["content"][0]["text"])
report = result["content"][1]["json"]
print(report["actions_applied"], report["action_errors"], round(report["elapsed_s"], 1))
robot.cleanup()
```

You should see:

```text
success
Policy complete on 'so101'
LerobotLocalPolicy | pick up the cube
20.8s | 90 steps | sim_t=3.060s
90 0 20.8
```

{{sim:same-checkpoint-1|where SmolVLA left the simulated arm after ninety steps}}

Ninety of ninety actions applied; the 20.8 s is a laptop CPU inferring 50-action chunks while the simulator waits. SmolVLA reads the instruction, so the words shaped the actions. `STRANDS_TRUST_REMOTE_CODE=1` is the consent lerobot checkpoints need.

## The same call on the real arm

Change how the robot is built and the keys its state arrives under; the checkpoint and the call do not change:

```python title="sketch"
import os
os.environ["STRANDS_TRUST_REMOTE_CODE"] = "1"
from strands_robots import Robot
from strands_robots.policies import create_policy

robot = Robot("so101", mode="real", port="/dev/ttyACM0",
              cameras={"front": {"type": "opencv", "index_or_path": 0}, "wrist": {"type": "opencv", "index_or_path": 1}})
keys = [f"{m}.pos" for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]
embodiment = {"state_keys": keys, "action_keys": keys, "dim_policy": "pad", "obs_rename": {
    "front": "observation.images.camera1", "wrist": "observation.images.camera2", "default": "observation.images.camera3"}}
policy = create_policy("lerobot_local", pretrained_name_or_path="lerobot/smolvla_base", embodiment=embodiment)
result = robot.run_policy(policy, instruction="pick up the cube", duration=10.0)
print(result["status"])
print(result["content"][0]["text"])
robot.cleanup()
```

`mode="real"` picks the lerobot driver, `port=` is the arm's USB device ([Real arm](first-real-arm.md) finds it), `cameras=` is lerobot's camera dict under the sim cameras' names. The driver reports the arm as `shoulder_pan.pos` to `gripper.pos`, so those are the keys; the rest of the dict is unchanged. On hardware `run_policy` blocks until `duration` elapses, `n_steps` actions were applied or `stop_task()` is called; `start_task(...)` does not block.

## Why the same object works on both

{{drawing:d03_same_checkpoint}}

The checkpoint never sees a robot. It sees `observation.state`, the numbers under the keys the embodiment names, and three images under the names it declares; it returns `action` in the same layout. Everything between those tensors and a body is the embodiment map: in the simulator the state arrives as joints `1` to `6` in radians, on the lerobot driver as `shoulder_pan.pos` to `gripper.pos` in degrees, and each dict binds its own keys. A checkpoint that carries its statistics can name `embodiment="so101"` and have the map convert units both ways ([LeRobot local](../learn/policies/lerobot-local.md) shows an ACT checkpoint trained on a real SO-101 doing that). Policy, checkpoint and call are identical; only the backend changes.

## What it does not mean

`smolvla_base` is a base model, not fine-tuned on this arm or this table: the page shows the contract, one Policy object and one `run_policy` call on two backends, not task success. A checkpoint that performs a task on your arm comes from [Teach it](teach-it.md); measured rates live on the [robot page](../robots/so101.md), each with its source.

## Behind the gate

As a tool, the real arm's `execute` and `start` carry the checkpoint and the embodiment dict in `policy_config` and stop for a human first. The gate runs before the port opens or the config is read, so a laptop with no arm reaches it:

```python
import asyncio
from strands_robots import Robot

arm = Robot("so101", mode="real", port="/dev/null")

async def call(action, **fields):
    tool_use = {"toolUseId": "demo", "name": arm.tool_name, "input": {"action": action, **fields}}
    async for event in arm.stream(tool_use, {}):
        return event.tool_result

refused = asyncio.run(call("execute", instruction="pick up the cube", policy_provider="lerobot_local",
                           policy_config={"pretrained_name_or_path": "lerobot/smolvla_base"}, duration=10))
print(refused["status"])
print(refused["content"][0]["text"].split(" No tool_context")[0])
arm.cleanup()
```

```text
error
so101: 'execute' drives the real robot 'so101' for up to 10s with 'pick up the cube' (policy lerobot_local built in this process, no server, checkpoint pretrained_name_or_path lerobot/smolvla_base); it needs operator approval before it is dispatched.
```

With no operator to ask, the call fails closed. Inside an `Agent` that sentence is the question the operator answers with `y` ([Talk to it](first-agent.md)). Over the mesh `policy_config` travels but `embodiment` does not yet ([#4180](https://github.com/strands-labs/robots/issues/4180)), so run a Hub checkpoint on a real arm from the process that owns it.

## Where next

You now have one checkpoint and one call for both arms, and you saw the gate refuse it by name. Next rung: [Teach it](teach-it.md) records on your arm and trains the checkpoint you run back on it. [Policies](../learn/policies/index.md) lists every provider.
