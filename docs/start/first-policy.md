---
description: "Stage 3: one Hub checkpoint, one run_policy call, the simulated SO-101 here and the physical one with a two-line change."
---

# Same checkpoint: sim or real

By the end of this page a checkpoint trained on a physical SO-101 has driven the simulated one from its wrist camera, you hold the same call for the physical arm, and you have watched the gate stop that call naming the checkpoint. The sim fence runs on a laptop with no GPU and downloads about 200 MB once.

## One checkpoint, one call

`robotfuel/act_so101_t16b` is an ACT policy trained on a real SO-101: one `observation.images.wrist` camera and a six-joint state in, six joint targets out. Add a wrist camera to the sim arm, route it onto the name the checkpoint declares, and name the embodiment:

```python
import os
os.environ["STRANDS_TRUST_REMOTE_CODE"] = "1"
from strands_robots import Robot
from strands_robots.policies import create_policy

robot = Robot("so101")
robot.add_camera(name="wrist", parent_body="so101/gripper", position=[0.058, 0.0, -0.029], target=[-0.024, 0.0, -0.297])
policy = create_policy("lerobot_local", pretrained_name_or_path="robotfuel/act_so101_t16b", embodiment="so101",
                       obs_rename_override={"wrist": "observation.images.wrist", "default": None})
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
8.2s | 90 steps | sim_t=3.060s
Note: LerobotLocalPolicy does not read the instruction. Its actions - the act checkpoint's actions from the observation alone - were commanded to the robot whatever the task says; nothing above means the task was performed.
90 0 8.2
```

{{sim:same-checkpoint-1|where the ACT checkpoint left the simulated arm after ninety steps from its wrist camera}}

Ninety of ninety actions applied. The 8.2 s is a laptop CPU inferring a 30-action chunk each second; the note says ACT has no language input, so the instruction is a label, not a command. `STRANDS_TRUST_REMOTE_CODE=1` is the consent lerobot checkpoints need. `obs_rename_override` routes your camera names onto the checkpoint's feature names and drops the one it does not read; without it the call refuses before any download and names the override.

## The same call on the real arm

Change how the robot is built; the policy and the call do not change:

```python title="sketch"
import os
os.environ["STRANDS_TRUST_REMOTE_CODE"] = "1"
from strands_robots import Robot
from strands_robots.policies import create_policy

robot = Robot("so101", mode="real", port="/dev/ttyACM0",
              cameras={"wrist": {"type": "opencv", "index_or_path": 0, "fps": 30}})
policy = create_policy("lerobot_local", pretrained_name_or_path="robotfuel/act_so101_t16b", embodiment="so101",
                       obs_rename_override={"wrist": "observation.images.wrist", "default": None})
result = robot.run_policy(policy_object=policy, instruction="pick up the cube", duration=10.0)
print(result["status"])
print(result["content"][0]["text"])
robot.cleanup()
```

`mode="real"` picks the lerobot driver, `port=` is the arm's USB device ([Real arm](first-real-arm.md) finds it), `cameras=` is lerobot's camera dict under the key the sim camera had. On hardware `run_policy` blocks until `duration` elapses, `n_steps` actions were applied or `stop_task()` is called; `start_task(...)` is the non-blocking form.

## Why the same object works on both

{{drawing:d03_same_checkpoint}}

The checkpoint never sees a robot. It sees `observation.state`, six numbers in the units it was trained on, and returns `action`, six numbers in the same units. Everything between that tensor and a robot is the embodiment map `embodiment="so101"` selects. In the simulator the state arrives as joints `1` to `6` in radians, so the map converts to degrees on the way in and back on the way out. On the lerobot driver it arrives as `shoulder_pan.pos` to `gripper.pos`, already in degrees with the gripper on 0 to 100, so the map binds those keys and converts nothing. Cameras travel by name through `obs_rename_override`. Policy, checkpoint and call are identical; only the backend changes.

## What it does not mean

The arm moved; the cube was not picked up. A checkpoint trained on one physical arm and table meets other lighting, another camera pose and a rest pose at the edge of its training range, so ninety applied steps prove the plumbing, not the task. Measured success rates live on the [robot page](../robots/so101.md), each with its source; a number without one is not written down.

## Behind the gate

Mounted as a tool, the real arm's `execute` and `start` carry the checkpoint in `policy_config` and stop for a human first. The gate runs before the driver opens its port, so a laptop with no arm reaches it:

```python
import asyncio
from strands_robots import Robot

arm = Robot("so101", mode="real", port="/dev/null")

async def call(action, **fields):
    tool_use = {"toolUseId": "demo", "name": arm.tool_name, "input": {"action": action, **fields}}
    async for event in arm.stream(tool_use, {}):
        return event.tool_result

checkpoint = {"pretrained_name_or_path": "robotfuel/act_so101_t16b", "embodiment": "so101",
              "obs_rename_override": {"front": None, "wrist": "observation.images.wrist"}}
refused = asyncio.run(call("execute", instruction="pick up the cube", policy_provider="lerobot_local",
                           policy_config=checkpoint, duration=10))
print(refused["status"])
print(refused["content"][0]["text"].split(" No tool_context")[0])
arm.cleanup()
```

```text
error
so101: 'execute' drives the real robot 'so101' for up to 10s with 'pick up the cube' (policy lerobot_local built in this process, no server, checkpoint pretrained_name_or_path robotfuel/act_so101_t16b); it needs operator approval before it is dispatched.
```

With no operator to ask, the call fails closed. Inside an `Agent` the same sentence is the question the operator answers, and `y` dispatches ([Talk to it](first-agent.md)). Over the mesh `policy_config` travels but `embodiment` does not yet ([#4180](https://github.com/strands-labs/robots/issues/4180)), so run a Hub checkpoint on a real arm from the process that owns it.

## Where next

You now have one checkpoint and one call for both arms, and you saw the gate refuse it by name. Next rung: [Teach it](teach-it.md) records on your arm and trains the checkpoint you run back on it. [Policies](../learn/policies/index.md) lists every provider; [LeRobot local](../learn/policies/lerobot-local.md) covers camera routing, units and `processor_overrides`.
