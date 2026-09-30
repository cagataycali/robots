---
hide: [navigation, toc]
template_class: sr-home
---

# Strands Robots

<div class="sr-hero" markdown>
<div markdown>
<p class="sr-hero__title">One <em>Robot</em> object. Any robot.</p>
<p class="sr-hero__lead">Simulated or physical, arm or humanoid, the same call. Hand it to a Strands Agent and it becomes a tool the model can use. Learned policies from the Hugging Face Hub run through the same call, in MuJoCo and on the physical robot. A policy rollout on a real arm waits for an operator's yes.</p>
<div class="sr-hero__actions">
<a class="sr-btn sr-btn--primary" href="start/">Start</a>
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
<div><strong>{{n:robots}}</strong><span>robots in the registry</span></div>
<div><strong>{{n:policy_providers}}</strong><span>policy providers behind one <code>run_policy</code> call</span></div>
<div><strong>{{n:native_drivers}}</strong><span>native hardware drivers that need no lerobot install</span></div>
</div>

<div class="sr-grid" markdown>
<div class="sr-card" markdown>
### [Start](start/index.md)
Install, move a simulated SO-101, plug in the real one, hand it to an agent, run the doctor.
</div>
<div class="sr-card" markdown>
### [Robots](robots/index.md)
The catalog: arms, hands, humanoids, quadrupeds, mobile bases. Which ones simulate, which ones have a driver.
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
Native and lerobot drivers, teleoperation, cameras, calibration.
</div>
<div class="sr-card" markdown>
### [Mesh](learn/mesh/fleet.md)
Several robots on several machines with one e-stop. Zenoh, mTLS, an access-control list.
</div>
<div class="sr-card" markdown>
### [Policies](learn/policies/index.md)
Learned policies behind one `run_policy` call: vision-language-action models, world foundation models, whole-body controllers, RL checkpoints, remote servers. Record on the robot, train, run the checkpoint back on it.
</div>
</div>

## One API, sim or real

<div class="sr-pair" markdown>

```python
from strands_robots import Robot

robot = Robot("so101", mode="sim")
robot.send_action({"1": 0.5}, n_substeps=200)
print(robot.get_robot_state()["content"][0]["text"])
robot.cleanup()
```

```python title="sketch"
from strands_robots import Robot

robot = Robot("so101", mode="real", driver="strands", port="/dev/ttyACM0")
robot.send_action({"shoulder_pan": 30.0})
print(robot.tool_spec["description"])
robot.cleanup()
```

</div>

The left fence runs on a laptop; the right needs an SO-101 on USB. Same tool, same verbs: the sim addresses joints by the model's names in radians, the native driver addresses servos by name in degrees. [Start here](start/index.md).

## The same checkpoint, sim or real

<div class="sr-pair" markdown>

```python
import os; os.environ["STRANDS_TRUST_REMOTE_CODE"] = "1"
from strands_robots import Robot
robot = Robot("so101", mode="sim")
robot.add_camera(name="front", position=[0.22, 0.025, 0.6], target=[0.22, 0.025, 0])
robot.add_camera(name="wrist", parent_body="so101/gripper", position=[0.058, 0.0, -0.029], target=[-0.024, 0.0, -0.297])
joints = robot.robot_joint_names("so101")
embodiment = {"state_keys": joints, "action_keys": joints, "dim_policy": "pad", "obs_rename": {
    "front": "observation.images.camera1", "wrist": "observation.images.camera2", "default": "observation.images.camera3"}}
print(robot.run_policy(robot_name="so101", policy_provider="lerobot_local", instruction="pick up the cube", n_steps=60,
                       policy_config={"pretrained_name_or_path": "lerobot/smolvla_base", "embodiment": embodiment})["status"])
robot.cleanup()
```

```python title="sketch"
import os; os.environ["STRANDS_TRUST_REMOTE_CODE"] = "1"
from strands_robots import Robot
from strands_robots.policies import create_policy
robot = Robot("so101", mode="real", port="/dev/ttyACM0", cameras={"front": {"type": "opencv", "index_or_path": 0},
              "wrist": {"type": "opencv", "index_or_path": 1}, "top": {"type": "opencv", "index_or_path": 2}})
embodiment = {"state_keys": list("123456"), "action_keys": list("123456"), "dim_policy": "pad", "obs_rename": {
    "front": "observation.images.camera1", "wrist": "observation.images.camera2", "top": "observation.images.camera3"}}
policy = create_policy("lerobot_local", pretrained_name_or_path="lerobot/smolvla_base", embodiment=embodiment)
print(robot.run_policy(policy, instruction="pick up the cube", duration=10.0)["status"])
robot.cleanup()
```

</div>

The left fence runs on a laptop with no GPU: SmolVLA, a vision-language-action model from the Hub, reads three cameras and the instruction and drives the simulated arm. The right one runs the same checkpoint on the physical arm; as an agent tool, its `execute` waits for operator approval. [First learned policy](start/first-policy.md) explains `obs_rename` and the embodiment.
