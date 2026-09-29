---
hide: [navigation, toc]
template_class: sr-home
---

# Strands Robots

<div class="sr-hero" markdown>
<div markdown>
<p class="sr-hero__title">Run a <em>VLA</em> from the Hub. On a real arm.</p>
<p class="sr-hero__lead">Name a checkpoint on the Hugging Face Hub, SmolVLA, ACT, Pi0, GR00T, and one <code>run_policy</code> call drives the robot: the same call in MuJoCo and on the physical arm. Hand the robot to a Strands Agent and it runs the policy as a tool. A real arm does not move until an operator says yes.</p>
<div class="sr-hero__actions">
<a class="sr-btn sr-btn--primary" href="start/first-policy/">Run your first policy</a>
<a class="sr-btn" href="robots/">Pick a robot</a>
<span class="sr-install">pip install "strands-robots[sim-mujoco]"<button class="sr-copy" data-clipboard-text='pip install "strands-robots[sim-mujoco]"'>copy</button></span>
</div>
</div>
<div class="sr-hero__stage" markdown>
<robot-viewer name="so101" autoload></robot-viewer>
<label class="sr-pick">Try another robot <select data-robot-pick><option value="so101">so101</option></select></label>
</div>
</div>

<div class="sr-proof" markdown>
<div><strong>{{n:policy_providers}}</strong><span>policy providers behind one <code>run_policy</code> call</span></div>
<div><strong>{{n:robots}}</strong><span>robots in the registry</span></div>
<div><strong>{{n:native_drivers}}</strong><span>native hardware drivers that need no lerobot install</span></div>
</div>

<div class="sr-grid" markdown>
<div class="sr-card" markdown>
### [Policies](learn/policies/index.md)
SmolVLA, ACT, Pi0, GR00T, Cosmos, whole-body control. Record on the arm, train in the cloud, run the checkpoint back on the arm.
</div>
<div class="sr-card" markdown>
### [Start](start/index.md)
Install, move a simulated SO-101, plug in the real one, run a Hub checkpoint on it, hand it to an agent.
</div>
<div class="sr-card" markdown>
### [Robots](robots/index.md)
The catalog: arms, hands, humanoids, quadrupeds, mobile bases. Which ones simulate, which have a driver, which checkpoints ran.
</div>
<div class="sr-card" markdown>
### [Agents](learn/agents.md)
`Agent(tools=[robot])`. What the model sees, what it can call, and the approval gate in front of real motion.
</div>
<div class="sr-card" markdown>
### [Simulation](learn/simulation/index.md)
MuJoCo on the CPU by default, Newton and Isaac on a GPU. Worlds, objects, cameras, predicates, recordings.
</div>
<div class="sr-card" markdown>
### [Hardware](learn/hardware/drivers.md)
Native and lerobot drivers, teleoperation, cameras, calibration. [Mesh](learn/mesh/fleet.md) for several robots and one e-stop.
</div>
</div>

## One checkpoint, sim or real

<div class="sr-pair" markdown>

```python
import os
os.environ["STRANDS_TRUST_REMOTE_CODE"] = "1"
from strands_robots import Robot

robot = Robot("so101", mode="sim")
robot.add_camera(name="wrist", parent_body="so101/gripper", position=[0.058, 0.0, -0.029], target=[-0.024, 0.0, -0.297])
checkpoint = {"pretrained_name_or_path": "robotfuel/act_so101_t16b", "embodiment": "so101",
              "obs_rename_override": {"front": None, "wrist": "observation.images.wrist"}}
result = robot.run_policy(robot_name="so101", policy_provider="lerobot_local", policy_config=checkpoint,
                          instruction="pick up the cube", n_steps=60, control_frequency=30.0)
print(result["status"])
print(result["content"][0]["text"])
robot.cleanup()
```

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
robot.cleanup()
```

</div>

The left fence runs on a laptop with no GPU: it downloads a 206 MB ACT checkpoint trained on a real SO-101, routes the wrist camera to the feature the checkpoint declares, converts between the checkpoint's degrees and the simulator's radians, and reports `success` with `60 steps`. The right one needs the arm on USB and a wrist camera; same checkpoint, same `embodiment`, same verb. Mounted as an agent tool, the real arm's `execute` stops for operator approval first. [Start here](start/first-policy.md).
