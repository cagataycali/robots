# Testing strands-robots on NVIDIA Thor (Jetson)

This guide covers setting up and running strands-robots with GR00T N1.6 inference on NVIDIA Jetson Thor.

## Hardware

| Component | Spec |
|-----------|------|
| **Platform** | NVIDIA Jetson Thor |
| **GPU** | NVIDIA Thor (131.9 GB unified VRAM) |
| **Architecture** | Linux aarch64 |
| **JetPack** | r36.5.0 |

## Quick Start

```bash
# 1. Pull the jetson-containers GR00T image
docker run -d --name gr00t-n1.6 \
  --runtime nvidia --gpus all \
  -v /data/models/huggingface:/data/models/huggingface \
  -e HF_HOME=/data/models/huggingface \
  dustynv/gr00t:r36.5.0 \
  sleep infinity

# 2. Patch Isaac-GR00T for Python 3.12 + newer deps (use our fork)
docker exec gr00t-n1.6 bash -c '
  cd /opt/Isaac-GR00T &&
  git remote add fork https://github.com/cagataycali/Isaac-GR00T.git &&
  git fetch fork main &&
  git checkout fork/main -- pyproject.toml gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/configuration_eagle3_vl.py &&
  pip3 install -e .
'

# 3. Install strands-robots
docker exec gr00t-n1.6 pip3 install strands-robots

# 4. Download the N1.6 model (first run only, ~6 GB)
docker exec gr00t-n1.6 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/GR00T-N1.6-3B')
"

# 5. Run inference test
docker exec gr00t-n1.6 python3 /opt/Isaac-GR00T/tests/test_n16_inference.py
```

## Why the Fork?

The upstream `NVIDIA/Isaac-GR00T` has two issues when running inside `dustynv/gr00t:r36.5.0`:

| Issue | Upstream | Our Fork (`cagataycali/Isaac-GR00T`) |
|-------|----------|--------------------------------------|
| Python version | `requires-python = "==3.10.*"` | `requires-python = ">=3.10"` |
| Dependency pins | `torch==2.7.1`, `numpy==1.26.4`, etc. | `torch>=2.7.1`, `numpy>=1.26.4`, etc. |
| Eagle VL config | Crashes on `transformers>=5.0` (`_attn_implementation_autoset` removed) | Guarded with `hasattr()` check |

The container ships Python 3.12 with pre-built CUDA wheels — the strict pins prevent installation.

## Model Versions

> **Important**: N1.6 code is **not** compatible with N1.5 weights.

| Model | Version | Parameters | Use With |
|-------|---------|------------|----------|
| `nvidia/GR00T-N1-2B` | N1.5 | 2B | Isaac-GR00T ≤ v0.1 (`gr00t.model.policy`) |
| `nvidia/GR00T-N1.6-3B` | **N1.6** | 3B | Isaac-GR00T v1.6 (`gr00t.policy.gr00t_policy`) |
| `nvidia/GR00T-N1.5-3B` | N1.5 | 3B | Isaac-GR00T N1.5 branch |

The architecture difference: N1.6 `CategorySpecificLinear` expects `[32, 1536, 1536]` vs N1.5 `[32, 1024, 1024]`. Loading the wrong weights causes a `state_dict` size mismatch.

## Available Embodiment Tags

The **base** `GR00T-N1.6-3B` model ships with 3 embodiments:

| Tag | DOF | Layout |
|-----|-----|--------|
| `gr1` | 29 | `left_arm(7) + right_arm(7) + left_hand(6) + right_hand(6) + waist(3)` |
| `behavior_r1_pro` | varies | Galaxea R1 Pro bimanual |
| `robocasa_panda_omron` | varies | Single Panda + mobile base |

The `new_embodiment` tag is only available in **fine-tuned** checkpoints (e.g. `nvidia/GR00T-N1.6-fractal`, `nvidia/GR00T-N1.6-DROID`).

Fine-tuned models on HuggingFace:
- `nvidia/GR00T-N1.6-fractal`
- `nvidia/GR00T-N1.6-bridge`
- `nvidia/GR00T-N1.6-DROID`
- `nvidia/GR00T-N1.6-BEHAVIOR1k`
- `nvidia/GR00T-N1.6-G1-PnPAppleToPlate`

## Using strands-robots Gr00tPolicy

### Native Isaac-GR00T API (low-level)

```python
import gr00t.model  # registers model types
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np

policy = Gr00tPolicy(
    embodiment_tag=EmbodimentTag.GR1,
    model_path="nvidia/GR00T-N1.6-3B",
    device="cuda",
)

# Modality config tells you the exact input format
mc = policy.get_modality_config()
sap = policy.processor.state_action_processor
print(f"State dim: {sap.get_state_dim('gr1')}")  # 29
print(f"Action dim: {sap.get_action_dim('gr1')}")  # 29

# Build observation (B=1, T=1, ...)
obs = {"video": {}, "state": {}, "language": {}}

for vk in mc["video"].modality_keys:
    obs["video"][vk] = np.random.randint(0, 255, (1, 1, 256, 256, 3), dtype=np.uint8)

gr1_dims = {"left_arm": 7, "right_arm": 7, "left_hand": 6, "right_hand": 6, "waist": 3}
for sk in mc["state"].modality_keys:
    obs["state"][sk] = np.zeros((1, 1, gr1_dims[sk]), dtype=np.float32)

for lk in mc["language"].modality_keys:
    obs["language"][lk] = [["pick up the red cube"]]

# Inference → 16-step action horizon
actions, info = policy.get_action(obs)
# actions = {"left_arm": (1,16,7), "right_arm": (1,16,7), "left_hand": (1,16,6), ...}
```

### strands-robots Wrapper (high-level)

```python
from strands_robots.policies.groot import Gr00tPolicy

# For SO-100 (6 DOF) with a fine-tuned checkpoint
policy = Gr00tPolicy(
    data_config="so100_dualcam",
    model_path="nvidia/GR00T-N1.6-fractal",  # must have new_embodiment tag
    embodiment_tag="new_embodiment",
    denoising_steps=4,
    device="cuda",
)

robot_keys = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_angle", "wrist_rotate", "gripper"]
policy.set_robot_state_keys(robot_keys)

obs = {
    "webcam": camera_frame,        # (H, W, 3) uint8
    "front": front_camera_frame,   # (H, W, 3) uint8
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.5,
    "elbow": 0.3,
    "wrist_angle": 0.1,
    "wrist_rotate": 0.0,
    "gripper": 0.5,
}

actions = policy.get_actions_sync(obs, "pick up the red cube")
# actions = [{"shoulder_pan": 0.01, "shoulder_lift": -0.02, ...}, ...]  # 16 timesteps
```

### GR1 via strands-robots

```python
from strands_robots.policies.groot import Gr00tPolicy

# GR1 (29 DOF) with base model
policy = Gr00tPolicy(
    data_config="fourier_gr1_arms_waist",
    model_path="nvidia/GR00T-N1.6-3B",
    embodiment_tag="gr1",
    device="cuda",
)
```

## Available Data Configs

strands-robots ships configs for all major embodiments:

| Config Name | Robot | DOF | Cameras |
|------------|-------|-----|---------|
| `so100` | SO-100 | 6 | 1 (webcam) |
| `so100_dualcam` | SO-100 | 6 | 2 (front + wrist) |
| `fourier_gr1_arms_only` | Fourier GR-1 | 28 | 1 (ego) |
| `fourier_gr1_arms_waist` | Fourier GR-1 | 31 | 1 (ego) |
| `unitree_g1` | Unitree G1 | 28 | 1 (ego) |
| `unitree_g1_full_body` | Unitree G1 | 43 | 1 (ego) |
| `bimanual_panda_gripper` | Franka Panda | 14 | 3 |
| `oxe_droid` | DROID | 8 | 2 |
| `agibot_dual_arm` | AgiBot | 16 | 3 |

Full list: `from strands_robots.policies.groot.data_config import DATA_CONFIG_MAP; print(list(DATA_CONFIG_MAP.keys()))`

## Benchmark Results

Tested on NVIDIA Thor (131.9 GB VRAM), GR00T N1.6-3B, GR1 embodiment (29 DOF):

| Metric | Value |
|--------|-------|
| Cold inference | 0.421s (2.4 Hz) |
| Warm inference | 0.126s (7.9 Hz) |
| **Average (10 runs)** | **0.125s (8.0 Hz)** |
| Action horizon | 16 timesteps |
| Denoising steps | 4 (default) |

## Running the Thor Test Suite

The `thor_tests/` directory contains end-to-end tests designed to run on Thor:

```bash
# Inside the Docker container:

# Test 1: MuJoCo simulation basics
python3 thor_tests/test_01_mujoco_sim.py

# Test 3: GR00T N1.6 local inference
python3 thor_tests/test_03_groot_n16_local.py

# Test 5: Policy resolver (auto-selects policy by name)
python3 thor_tests/test_05_policy_resolver.py

# Test 7: Newton GPU-accelerated backend
python3 thor_tests/test_07_newton_gpu.py

# Run all tests
for t in thor_tests/test_*.py; do echo "=== $t ===" && python3 "$t"; done
```

## Troubleshooting

### `KeyError: 'new_embodiment'`

The base `GR00T-N1.6-3B` model doesn't include the `new_embodiment` tag. Use one of:
- A fine-tuned model (`nvidia/GR00T-N1.6-fractal`, etc.)
- A tag that exists in the base model (`gr1`, `behavior_r1_pro`, `robocasa_panda_omron`)

### `size mismatch for CategorySpecificLinear`

You're loading N1.5 weights (`GR00T-N1-2B`) with N1.6 code. Use `nvidia/GR00T-N1.6-3B` instead.

### `State dim X != action dim Y`

The per-key state dimensions must match the model's expected layout. For GR1:
```
left_arm: 7    right_arm: 7    left_hand: 6    right_hand: 6    waist: 3
```
Total = 29. Do **not** guess — check with:
```python
sap = policy.processor.state_action_processor
print(sap.get_state_dim('gr1'))  # 29
```

### `requires-python == 3.10.*`

Use our fork: `pip install git+https://github.com/cagataycali/Isaac-GR00T.git`

### `_attn_implementation_autoset` AttributeError

Eagle VL config incompatibility with `transformers >= 5.0`. Our fork patches this.

### HuggingFace cache location

The container expects models in `/data/models/huggingface`. Mount it:
```bash
-v /data/models/huggingface:/data/models/huggingface -e HF_HOME=/data/models/huggingface
```

## Docker Container Details

| Property | Value |
|----------|-------|
| Image | `dustynv/gr00t:r36.5.0` |
| Python | 3.12 |
| PyTorch | 2.7+ (pre-built CUDA aarch64 wheel) |
| Transformers | 4.51+ |
| Isaac-GR00T | Mounted at `/opt/Isaac-GR00T` |
| GPU Runtime | `--runtime nvidia --gpus all` |
