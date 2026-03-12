# Recording & Datasets

Record robot demonstrations and manage datasets for training.

---

## Overview

Training a robot policy requires data. Data comes from **demonstrations** — recordings of a robot performing a task. Strands Robots records directly to [LeRobotDataset](https://github.com/huggingface/lerobot) format: parquet episodes + encoded video, ready for training.

---

## Recording with HardwareRobot

The `HardwareRobot` tool has a built-in `record` action:

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("so100", mode="real", cameras={
    "wrist": {"type": "opencv", "index_or_path": "/dev/video0"}
})

agent = Agent(tools=[robot])
agent("Record a demonstration of picking up the red cube using the groot policy on port 5555")
```

This records observations + actions to a LeRobotDataset at each control step.

---

## DatasetRecorder (Programmatic)

```python
from strands_robots import DatasetRecorder

recorder = DatasetRecorder.create(
    repo_id="local/my_dataset",
    fps=30,
    robot_type="so100",
    joint_names=["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
    camera_keys=["wrist"],
    task="pick up the red cube",
    video_backend="auto",  # Auto-detect HW encoder
)

# Add frames during execution
recorder.add_frame(observation=obs, action=action, task="pick up the red cube", camera_keys=["wrist"])

# Save episode
recorder.save_episode()

# Push to HuggingFace Hub
recorder.push_to_hub(tags=["strands-robots", "real"])
recorder.finalize()
```

---

## RecordSession (Teleop + Policy)

```python
from strands_robots import RecordSession, RecordMode

session = RecordSession(
    robot=robot,
    mode=RecordMode.TELEOP,  # or RecordMode.POLICY
    repo_id="user/my_recordings",
    fps=30,
)
session.start()
# ... teleop or policy execution ...
session.stop()
```

---

## Replay on Hardware

Replay a recorded episode on the real robot:

```python
agent("Replay episode 0 from user/my_dataset at 0.5x speed")
```

Or programmatically:

```python
robot.replay_episode(
    repo_id="user/my_dataset",
    episode=0,
    speed=0.5,
)
```

---

## Dataset Management Tool

Use the `lerobot_dataset` tool for dataset operations:

```python
from strands import Agent
from strands_robots import lerobot_dataset

agent = Agent(tools=[lerobot_dataset])
agent("List all local LeRobot datasets")
agent("Show info about lerobot/so100_wipe")
```

---

## Camera Tool

Discover and manage cameras for recording:

```python
from strands import Agent
from strands_robots import lerobot_camera

agent = Agent(tools=[lerobot_camera])
agent("Discover all connected cameras")
agent("Capture a frame from camera 0")
```
