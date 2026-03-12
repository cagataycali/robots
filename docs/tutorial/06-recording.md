# Chapter 6: Recording Data

**Time:** 20 minutes · **Hardware:** Optional (can use simulation) · **Level:** Advanced

To train a robot policy, you need data. Data comes from **demonstrations** — recordings of a robot performing a task, either through teleoperation or scripted motions.

---

## Why Record?

Imitation learning works like this:

1. **Record** a human demonstrating the task (via teleoperation or kinesthetic teaching)
2. **Store** the observations and actions as a dataset
3. **Train** a policy to reproduce those behaviors
4. **Deploy** — the robot can now do the task autonomously

---

## Record from Simulation

```python
from strands_robots import Robot, create_policy

robot = Robot("so100")
policy = create_policy("mock")

# Start recording
robot.start_recording("my_dataset")

for episode in range(10):
    robot.reset()
    for step in range(200):
        obs = robot.get_observation()
        action = policy.get_actions(obs, "pick up cube")
        robot.apply_action(action)

# Stop and save
robot.stop_recording()
```

---

## Teleoperation

Record demonstrations by controlling a real robot:

```python
from strands import Agent
from strands_robots import Robot, teleoperator

agent = Agent(tools=[Robot("so100"), teleoperator])

# Start a teleoperation session
agent("Start teleoperation with the leader-follower setup")

# ... physically move the leader arm, follower records ...

# Stop and save
agent("Stop recording and save as 'pick_and_place_demo'")
```

The **teleoperator** tool manages leader-follower recording sessions.

---

## Dataset Format

Strands Robots uses the [LeRobot dataset format](https://github.com/huggingface/lerobot) — a standard format supported by the community:

```
dataset/
├── episode_000/
│   ├── observation.joint_positions.npy
│   ├── observation.cameras.front.mp4
│   └── action.npy
├── episode_001/
│   └── ...
└── metadata.json
```

---

## Manage Datasets

```python
from strands import Agent
from strands_robots import lerobot_dataset

agent = Agent(tools=[lerobot_dataset])

agent("List all datasets in /data/recordings")
agent("Upload 'pick_and_place_demo' to HuggingFace")
agent("Download dataset 'lerobot/so100_wipe'")
```

---

## What You Learned

- ✅ Demonstrations are the data for imitation learning
- ✅ Record from simulation or real hardware
- ✅ Teleoperation for human-guided recording
- ✅ LeRobot dataset format for compatibility
- ✅ Dataset management tools for organizing recordings

---

**Next:** [Chapter 7: Training →](07-training.md) — Train a policy on your recorded data.
