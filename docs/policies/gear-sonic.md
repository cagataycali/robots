# GEAR-SONIC Policy Provider

[GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) is NVIDIA's humanoid behavior foundation model — a 42M-parameter whole-body controller that gives robots natural movement from large-scale human motion data. It's the motor controller behind GR00T N1.5 and N1.6.

> **Paper:** [SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control](https://arxiv.org/abs/2511.07820)
> **Models:** [nvidia/GEAR-SONIC on HuggingFace](https://huggingface.co/nvidia/GEAR-SONIC)
> **Docs:** [GR00T-WholeBodyControl Documentation](https://nvlabs.github.io/GR00T-WholeBodyControl/)

---

## How It Works

```
Observation History → Encoder → 64-dim Motion Tokens → Decoder → 29-DOF Joint Actions
```

SONIC uses an encoder-decoder architecture:

| Component | Size | What it does |
|-----------|------|-------------|
| **Encoder** | 50 MB | Processes observation history → 64-dim motion tokens |
| **Decoder** | 40 MB | Current state + tokens → 29-DOF joint position targets |
| **Planner** | 774 MB | Optional kinematic planner for task-level control |

Runs at **>135Hz** on GPU, **>100Hz** on CPU via ONNX.

---

## Input Modes

SONIC supports multiple control interfaces:

| Mode | Input | Use Case |
|------|-------|----------|
| **Motion Tracking** | Reference joint trajectories | Retarget recorded motions |
| **VR Teleoperation** | PICO VR headset (3-point tracking) | Data collection, interactive control |
| **Video-Based** | SMPL pose estimation from video | Video-to-robot motion transfer |
| **VLA Integration** | GR00T N1.6 high-level plans | SONIC executes whole-body movement |

---

## Setup

### Option 1: Auto-download from HuggingFace

```python
from strands_robots import create_policy

# Downloads ONNX models automatically (~100MB)
policy = create_policy("gear_sonic")
```

### Option 2: Clone GR00T-WholeBodyControl

For the full framework including C++ deployment, VR teleoperation, and kinematic planner:

```bash
# Clone with Git LFS (required for model weights)
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git lfs pull
```

Then point to the local models:

```python
policy = create_policy("gear_sonic",
    model_dir="/path/to/GR00T-WholeBodyControl/models")
```

### Option 3: Install from PyPI (inference only)

```bash
pip install onnxruntime          # CPU inference
pip install onnxruntime-gpu      # GPU inference (CUDA)
```

---

## Usage

### Basic — Humanoid Control

```python
from strands_robots import Robot, create_policy

robot = Robot("unitree_g1")
policy = create_policy("gear_sonic")

for step in range(1000):
    obs = robot.get_observation()
    actions = policy.get_actions(obs, instruction="walk forward")
    robot.apply_action(actions[0])
```

### With Agent

```python
from strands import Agent
from strands_robots import Robot

robot = Robot("unitree_g1")
agent = Agent(tools=[robot])
agent("Stand up and walk forward using GEAR-SONIC")
```

### Explicit Configuration

```python
policy = create_policy("gear_sonic",
    model_dir="/path/to/models",
    mode="motion_tracking",   # or "teleop", "smpl"
    device="cuda",            # or "cpu"
    use_planner=True,         # Load 774MB kinematic planner
    history_len=10,           # Observation history frames
)
```

---

## Target Robot

GEAR-SONIC is designed for the **Unitree G1** (29 DOF):

| Body Part | DOF | Joints |
|-----------|-----|--------|
| Legs | 12 | hip pitch/roll/yaw, knee, ankle pitch/roll (×2) |
| Waist | 1 | waist yaw |
| Arms | 14 | shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw (×2) |
| Hands | 2 | left/right hand |
| **Total** | **29** | |

---

## GR00T-WholeBodyControl Repository

The [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) repository includes:

### What's Included

| Component | Description |
|-----------|-------------|
| **gear_sonic** | Python teleoperation stack for data collection |
| **gear_sonic_deploy** | C++ inference stack for real hardware deployment |
| **Model Checkpoints** | Pre-trained ONNX encoder, decoder, and planner |
| **VR Teleop** | PICO VR headset integration for whole-body teleoperation |
| **Kinematic Planner** | Keyboard/gamepad-driven locomotion with style selection |

### Full Setup (Training & Deployment)

```bash
# Clone
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git lfs pull

# Install Python dependencies
pip install -e .

# For IsaacLab integration (training)
# Requires IsaacLab 2.3.0 + Isaac Sim
```

### VR Teleoperation

SONIC supports real-time whole-body teleoperation via PICO VR headset:

- Walking, running, sideways movement
- Kneeling, getting up, jumping
- Bimanual manipulation, object hand-off

See: [VR Teleoperation Setup Guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html)

### Kinematic Planner

Interactive locomotion with style selection — choose movement style, steer with keyboard/gamepad:

- Walking styles: run, happy, stealth, injured
- Ground movements: kneeling, hand crawling, elbow crawling
- Combat: boxing stance

See: [Keyboard Control](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/keyboard.html) | [Gamepad Control](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/gamepad.html)

---

## Decoupled WBC (N1.5 / N1.6)

The earlier **Decoupled Whole-Body Control** approach (used in GR00T N1.5 and N1.6) separates:

- **Lower body**: RL-trained locomotion controller
- **Upper body**: Inverse kinematics solver

This is available in the same repository. See [Decoupled WBC documentation](https://nvlabs.github.io/GR00T-WholeBodyControl/).

---

## Resources

| Resource | Link |
|----------|------|
| GitHub | [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) |
| Paper | [SONIC (arXiv:2511.07820)](https://arxiv.org/abs/2511.07820) |
| Models | [nvidia/GEAR-SONIC on HuggingFace](https://huggingface.co/nvidia/GEAR-SONIC) |
| Docs | [Full Documentation](https://nvlabs.github.io/GR00T-WholeBodyControl/) |
| Website | [GEAR-SONIC Project Page](https://nvlabs.github.io/GEAR-SONIC/) |
| License | Apache 2.0 (code) + NVIDIA Open Model License (weights) |
