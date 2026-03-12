# Troubleshooting

Common issues and how to fix them.

---

## Installation

### `pip install strands-robots` fails

```bash
# Make sure you have Python 3.10+
python --version

# Upgrade pip
pip install --upgrade pip

# Try again
pip install strands-robots
```

### MuJoCo won't install

```bash
# MuJoCo requires specific system libraries
# On Ubuntu:
sudo apt-get install libgl1-mesa-glx libosmesa6

# On Mac (usually works out of the box):
pip install "strands-robots[mujoco]"
```

### MuJoCo interactive viewer fails on macOS

If you get `launch_passive requires that the Python script be run under mjpython on macOS`:

```bash
# Use mjpython instead of python for interactive viewer
mjpython my_script.py
```

`mjpython` is installed with the `mujoco` package. This is only needed for the interactive 3D viewer window — headless rendering (`render()`, `sim.render("frame.png")`) works fine with standard `python` on all platforms.

> **Agent workflows don't need mjpython** — agents use headless `render()` which returns PNG bytes directly.

---

## Robot Creation

### `Robot("name")` not found

```python
# Check available robots
from strands_robots.factory import list_robots
for r in list_robots():
    print(r['name'])
```

Make sure the robot name matches exactly (case-sensitive).

### Wrong mode (sim vs real)

`Robot()` auto-detects hardware. If it's picking the wrong mode:

```python
# Force simulation
robot = Robot("so100", force_sim=True)

# Force real hardware
robot = Robot("so100", port="/dev/ttyACM0")
```

---

## Cameras

### Camera not found

```python
from strands import Agent
from strands_robots import lerobot_camera

agent = Agent(tools=[lerobot_camera])
agent("Discover all cameras")
```

Common issues:

- Camera in use by another application
- Wrong device path (try `/dev/video0`, `/dev/video2`, etc.)
- Permissions: `sudo chmod 666 /dev/video0`

---

## GR00T Inference

### Connection refused

The GR00T inference server must be running before you connect:

```bash
# On the Jetson
jetson-containers run $(autotag isaac-gr00t) &

# Wait for it to start, then try connecting
```

### Slow inference

Enable TensorRT acceleration:

```python
policy = create_policy(
    provider="groot",
    use_tensorrt=True,
    vit_dtype="fp8",
    llm_dtype="nvfp4",
)
```

---

## GEAR-SONIC

### Models not found

GEAR-SONIC needs ONNX model files. Either auto-download from HuggingFace or clone the repo:

```bash
# Option 1: Auto-download (happens on first use)
policy = create_policy("gear_sonic")

# Option 2: Clone with Git LFS
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git lfs pull

# Then point to local models
policy = create_policy("gear_sonic",
    model_dir="/path/to/GR00T-WholeBodyControl/models")
```

### ONNX runtime not found

```bash
pip install onnxruntime          # CPU
pip install onnxruntime-gpu      # GPU (CUDA)
```

### Git LFS files not downloaded

If model files are tiny placeholders instead of actual weights:

```bash
cd GR00T-WholeBodyControl
git lfs install
git lfs pull
```

---

## Agent Issues

### "No AWS credentials"

The default Strands model provider is Bedrock. Either:

1. Configure AWS credentials: `aws configure`
2. Or use a different provider:

```python
from strands.models.ollama import OllamaModel
agent = Agent(model=OllamaModel(model_id="llama3"), tools=[robot])
```

---

## Still Stuck?

- [GitHub Issues](https://github.com/strands-labs/robots/issues) — search existing issues or open a new one
- [Strands Docs](https://strandsagents.com/) — for agent and model provider issues
