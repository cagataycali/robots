---
description: Five minutes from install to a robot picking up a cube.
---

# Quickstart

```bash
uv pip install "strands-robots[sim-mujoco]"
```

```python
from strands_robots import Robot
import imageio.v3 as iio

sim = Robot("so100")                              # sim by default; CPU-only, no GPU
sim.step()

frame = sim.get_observation("so100")["default"]   # uint8 HxWx3
iio.imwrite("first_frame.png", frame)
```

> Headless box? `export MUJOCO_GL=osmesa` before importing. See [Troubleshooting](../troubleshooting.md).

## Add an object and run a policy

```python
sim.add_object(
    name="cube", shape="box", size=[0.025]*3,
    position=[0.3, 0.0, 0.025], color=[1, 0, 0, 1],
)

sim.run_policy(
    robot_name="so100",
    instruction="pick up the red cube",
    policy_provider="mock",   # no GPU; swap "groot" or "lerobot_local" with a real model
    duration=10.0,
)
```

## Drive with an agent

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("so100")
agent = Agent(tools=[robot])
agent("Add a red cube and pick it up using the mock policy")
```

## The whole loop

Teleoperate a real arm to collect demos, post-tune a policy on them, run it in
sim and on hardware, hand work to a fleet peer, expose it on ROS 2 - one
library. Each step is a distinct capability; the pages linked cover the details.

```python
from strands import Agent
from strands_robots import Robot
from strands_robots.tools import train_policy

# 1. TELEOPERATE a real SO-101 with its leader arm and RECORD demos.
follower = Robot("so101", mode="real", port="/dev/ttyACM0",
                 cameras={"front": {"type": "opencv", "index_or_path": "/dev/video0"}},
                 mesh=True)
follower.attach_teleop("so101_leader", port="/dev/ttyACM1", id="leader")
Agent(tools=[follower])(
    "start_recording(repo_id='me/pick', root='/tmp/pick', fps=30, "
    "task='pick up the cube'); teleoperate for 60s; stop_recording"
)

# 2. POST-TUNE a policy on those demos (LoRA fine-tune; GPU box).
train_policy(action="train", provider="lerobot_local",
             dataset_root="/tmp/pick", base_model="lerobot/smolvla_base",
             output_dir="/tmp/pick_ckpt", method="lora", steps=20000)

# 3. RUN the tuned checkpoint - same policy on a MuJoCo twin AND the real arm.
twin = Robot("so101")
twin.run_policy(robot_name="so101", policy_provider="lerobot_local",
                policy_config={"pretrained_name_or_path": "/tmp/pick_ckpt"}, duration=10.0)
follower.start_task("pick up the cube", policy_provider="lerobot_local",
                    policy_port=None, duration=10.0)

# 4. COORDINATE a fleet - tell a mesh peer to assist, in natural language.
follower.mesh.tell(follower.mesh.peers[0]["peer_id"], "hold the tray steady")

# 5. EXPOSE the running sim on ROS 2 - rviz / nav2 / any ros2 node can subscribe.
from strands_robots.simulation import Simulation
sim = Simulation(ros2_bridge=True); sim.create_world(); sim.add_robot("so101")
sim.step(100)
```

1. Teleop + recording - [Teleoperation](../hardware/teleoperation.md), [Recording](../recording.md).
2. Post-tuning - [Training](../training/overview.md).
3. Sim and hardware rollout - [Policies](../policies/overview.md), [Robot control](../hardware/robot-control.md).
4. Fleet coordination - [Mesh](../mesh.md).
5. ROS 2 interop - [ROS 2](../ros2-integration.md).

Steps 1 and 3-real need hardware; step 2 needs a GPU. Everything else runs in sim.

## Next: the notebook series

For a guided, click-and-run path, work through the
[getting-started notebooks](../examples/overview.md), five notebooks that run
end-to-end in simulation with no hardware, no GPU, and no Hugging Face
credentials. They take you from `Robot("so100")` through recording a dataset,
training a policy, and the full streaming data loop, each building on the last.

## See also

- [Policy providers](../policies/overview.md) - GR00T, LeRobot Local, Cosmos 3.
- [Robot catalog](../robots/index.md) - all {{n:robots}} robots.
- [Real hardware](../hardware/robot-control.md) - same code, `mode="real"`.
