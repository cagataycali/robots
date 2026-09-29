---
description: One run_policy call runs a learned policy from the Hugging Face Hub on the simulated SO-101, then the same call on the real one, behind the operator gate.
---

# First learned policy

By the end of this page a vision-language-action model from the Hub has driven the simulated SO-101 from its cameras, you have the same call for the physical arm, and you have watched an agent stop at the approval gate naming the checkpoint. The sim fences run on a laptop with no GPU; the first downloads about 865 MB once.

The page uses a vision-language-action checkpoint because the SO-101 has several; the same shape, one provider name and one `policy_config`, runs a world foundation model through [cosmos3](../learn/policies/cosmos3.md) or a whole-body controller through [wbc](../learn/policies/wbc.md).

## Run SmolVLA in the simulator

`lerobot/smolvla_base` reads three cameras and a six-joint state. The sim arm has one camera, `default`; add two more, name them the features the checkpoint declares, and tell the policy which joints it drives:

```python
import os
os.environ["STRANDS_TRUST_REMOTE_CODE"] = "1"
from strands_robots import Robot

robot = Robot("so101", mode="sim")
robot.add_camera(name="front", position=[0.22, 0.025, 0.6], target=[0.22, 0.025, 0])
robot.add_camera(name="wrist", parent_body="so101/gripper", position=[0.058, 0.0, -0.029], target=[-0.024, 0.0, -0.297])
joints = robot.robot_joint_names("so101")
embodiment = {"name": "so101_native", "state_keys": joints, "action_keys": joints, "dim_policy": "pad",
              "obs_rename": {"front": "observation.images.camera1", "wrist": "observation.images.camera2",
                             "default": "observation.images.camera3"}}
result = robot.run_policy(robot_name="so101", policy_provider="lerobot_local",
                          policy_config={"pretrained_name_or_path": "lerobot/smolvla_base", "embodiment": embodiment},
                          instruction="pick up the cube", n_steps=60, control_frequency=30.0)
print(result["status"])
print(result["content"][0]["text"])
report = result["content"][1]["json"]
print(report["actions_applied"], report["action_errors"], report["instruction_read"])
robot.cleanup()
```

You should see:

```text
Loading  HuggingFaceTB/SmolVLM2-500M-Video-Instruct weights ...
Reducing the number of VLM layers to 16 ...
success
Policy complete on 'so101'
LerobotLocalPolicy | pick up the cube
4.3s | 60 steps | sim_t=2.040s
60 0 True
```

Lerobot prints the first two lines while it loads; the timing line is yours; on the Apple laptop GPU that produced this output each 50-action chunk took about 1.2 s. `STRANDS_TRUST_REMOTE_CODE=1` is the consent lerobot checkpoints need before they build. `obs_rename` routes your camera names onto the feature names the checkpoint was trained with; without it the call refuses before any download and names the override to pass. The inline `embodiment` says which joints are the state and action vectors. It is inline because `smolvla_base` is a pretraining checkpoint with no SO-101 statistics, so the shipped `embodiment="so101"` (which converts servo degrees) is refused and the arm's radians reach the model raw. The motion you see is the plumbing working, not a task being done. A checkpoint fine-tuned on an SO-101 carries its stats, and `embodiment="so101"` then converts units both ways, as the ACT checkpoint on the [lerobot_local page](../learn/policies/lerobot-local.md) does.

## The same call on the real arm

Swap `mode`, the camera dict and the checkpoint: the real arm gets one fine-tuned on an SO-101, so the shipped `embodiment="so101"` binds its `shoulder_pan.pos` keys and converts the units:

```python title="sketch"
import os
os.environ["STRANDS_TRUST_REMOTE_CODE"] = "1"
from strands_robots import Robot
from strands_robots.policies import create_policy

robot = Robot("so101", mode="real", port="/dev/ttyACM0",
              cameras={"wrist": {"type": "opencv", "index_or_path": 0, "fps": 30}})
policy = create_policy("lerobot_local", pretrained_name_or_path="robotfuel/act_so101_t16b", embodiment="so101",
                       obs_rename_override={"front": None, "wrist": "observation.images.wrist"})
result = robot.run_policy(policy, instruction="pick up the cube", duration=10.0)
print(result["status"])
print(result["content"][0]["text"])
robot.cleanup()
```

`run_policy` on hardware takes a policy built with `create_policy` and blocks until `duration` elapses, `n_steps` actions were applied, or `stop_task()` is called; `start_task(instruction, policy_provider="lerobot_local", pretrained_name_or_path=..., embodiment=...)` is the non-blocking form. Both refuse while another rollout holds the bus, and after `cleanup()`. On a real arm use this checkpoint or [your own](../learn/training/lerobot.md).

## Behind the gate

Mounted as a tool, the real arm's `execute` and `start` actions carry the checkpoint in `policy_config` and stop for a human before anything is dispatched. `mock=True` gives the lerobot driver a mocked servo bus, so this runs on a laptop:

```python
import asyncio
from strands_robots import Robot

arm = Robot("so101", mode="real", port="/dev/null", mock=True)

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

With no operator to ask, the call fails closed. Inside an `Agent`, the same sentence is the question the operator answers and `y` dispatches the rollout; [First agent](first-agent.md) shows the interrupt and the resume line. Over the mesh, `policy_config` travels but `embodiment` does not yet ([#4180](https://github.com/strands-labs/robots/issues/4180)), so run a Hub checkpoint on a real arm from the process that owns it.

## Where next

[Policies](../learn/policies/index.md) lists every provider and which checkpoints ran where; [LeRobot local](../learn/policies/lerobot-local.md) covers camera routing, units and `processor_overrides`; [Training](../learn/training/lerobot.md) records on the arm and trains the checkpoint you run back on it.
