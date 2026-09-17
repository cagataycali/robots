---
description: Five minutes from install to an arm moving in MuJoCo, and the reference pick that lifts a cube.
---

# Quickstart

```bash
uv venv --python 3.12 && source .venv/bin/activate
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

`mock` exercises the loop, not the task. `MockPolicy` declares
`reads_instruction = False` and drives every joint through a test motion, so the
envelope it returns says nothing above means the task was performed - and after
ten seconds the cube has not moved. For a cube that really leaves the table on
this same install, run `examples/18_so101_pick_and_lift.py` (scripted, CPU-only,
lifts it about 150 mm). For a policy that acts on the words, pass
`lerobot_local` with a checkpoint.

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
from strands_robots import train_policy

# 1. TELEOPERATE a real SO-101 with its leader arm and RECORD demos.
#    Recording a real arm is lerobot-record; the lerobot_teleoperate tool runs
#    it as a session. Robot(mode="real") drives policies (execute, start,
#    status, stop) - it neither teleoperates nor records from an agent.
from strands_robots import lerobot_teleoperate
Agent(tools=[lerobot_teleoperate])(
    "start a recording session: follower so101_follower on /dev/ttyACM0 with "
    "camera front at /dev/video0, leader so101_leader on /dev/ttyACM1, dataset "
    "me/pick under /tmp/pick, one 60 s episode, task 'pick up the cube'"
)

# 2. POST-TUNE a policy on those demos (LoRA fine-tune; GPU box).
train_policy(action="train", provider="lerobot_local",
             dataset_root="/tmp/pick", base_model="lerobot/smolvla_base",
             output_dir="/tmp/pick_ckpt", method="lora", steps=20000)

# 3. RUN the tuned checkpoint - same policy on a MuJoCo twin AND the real arm.
follower = Robot("so101", mode="real", port="/dev/ttyACM0",
                 cameras={"front": {"type": "opencv", "index_or_path": "/dev/video0"}},
                 mesh=True)
twin = Robot("so101")
twin.run_policy(robot_name="so101", policy_provider="lerobot_local",
                policy_config={"pretrained_name_or_path": "/tmp/pick_ckpt"}, duration=10.0)
follower.start_task("pick up the cube", policy_provider="lerobot_local",
                    pretrained_name_or_path="/tmp/pick_ckpt", duration=10.0)

# 4. COORDINATE a fleet - hand a mesh peer a task: the policy it runs, and
#    the instruction it runs with. The wire boundary refuses an execute with
#    no policy_provider, and takes checkpoints as Hub ids (an org in
#    STRANDS_MESH_HF_REPO_ALLOW), not local paths. Presence arrives ~1 s
#    after the peer starts.
peer = follower.mesh.peers[0]["peer_id"]
follower.mesh.tell(peer, "hold the tray steady", policy_provider="lerobot_local",
                   pretrained_name_or_path="lerobot/smolvla_base", duration=10.0)

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

Steps 1 and 3-real need hardware; step 2 needs a GPU. Step 4 needs the `[mesh]`
extra and a mesh posture - `eclipse-zenoh` is not in the `[sim-mujoco]` install
above, and without it (or without `STRANDS_MESH_LOCAL_DEV=true` / an ACL file)
the mesh stays off, so `mesh.peers` is empty and `peers[0]` raises `IndexError`.
Step 5 needs a sourced ROS 2 distro - `rclpy` is not on PyPI, and
`Simulation(ros2_bridge=True)` raises an `ImportError` naming the
`source /opt/ros/<distro>/setup.bash` to run first. Step 3-twin runs in sim on
the install line at the top of this page.

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
