# GR00T Policy Provider

[NVIDIA GR00T](https://github.com/NVIDIA/Isaac-GR00T) is a vision-language-action (VLA) foundation model. It sees camera images, understands natural language instructions, and outputs robot actions.

---

## How It Works

```
Camera Image + "Pick up the red cube" → GR00T → Joint Actions
```

GR00T processes visual observations and language together to produce action sequences (action chunks).

---

## Setup

GR00T inference runs in a Docker container, typically on an NVIDIA Jetson or GPU server:

```bash
# On the Jetson / GPU server
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers
jetson-containers run $(autotag isaac-gr00t) &
```

The container exposes inference on port 5555 (ZMQ) or 8000 (HTTP).

---

## Usage

```python
from strands_robots import create_policy

# Connect to GR00T server
policy = create_policy("zmq://jetson:5555")

# Or explicitly
policy = create_policy(
    provider="groot",
    host="192.168.1.100",
    port=5555,
    data_config="so100_dualcam",
)
```

---

## With Agent

```python
from strands import Agent
from strands_robots import Robot, gr00t_inference

robot = Robot("so100")
agent = Agent(tools=[robot, gr00t_inference])

# Start inference service
agent("Start GR00T inference on port 8000 with checkpoint /data/model")

# Control robot
agent("Pick up the red block using GR00T")
```

---

## TensorRT Acceleration

For maximum performance:

```python
policy = create_policy(
    provider="groot",
    host="localhost",
    port=8000,
    use_tensorrt=True,
    vit_dtype="fp8",
    llm_dtype="nvfp4",
    dit_dtype="fp8",
)
```

---

## Supported Models

- GR00T N1.5
- GR00T N1.6

## Supported Platforms

- NVIDIA Thor Dev Kit (Jetpack 7.0)
- NVIDIA Jetson AGX Orin (Jetpack 6.x)
- Any NVIDIA GPU with sufficient VRAM
