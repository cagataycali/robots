# Chapter 9: Advanced

**Time:** 30 minutes · **Hardware:** Optional · **Level:** Advanced

You've mastered the basics. This chapter covers the powerful features that make Strands Robots production-ready.

---

## Custom Policies

Build your own policy provider:

```python
from strands_robots.policies import Policy, register_policy

class MyPolicy(Policy):
    def get_actions(self, observation, instruction):
        # Your logic here — classical control, neural network, anything
        return compute_action(observation)

    @property
    def provider_name(self):
        return "my_policy"

# Register it
register_policy("my_policy", lambda: MyPolicy, aliases=["custom"])

# Now use it like any other
policy = create_policy("my_policy")
```

Your policy plugs into the same ecosystem — same interface, same tools, same agent integration.

---

## DreamGen Pipeline

Record one demonstration. Generate 50 training variations:

```python
from strands_robots import DreamGenPipeline

pipeline = DreamGenPipeline(
    video_model="wan2.1",
    idm_checkpoint="nvidia/gr00t-idm-so100",
    embodiment_tag="so100",
)

results = pipeline.run_full_pipeline(
    robot_dataset_path="/data/pick_and_place",
    instructions=["pour water", "fold towel", "stack blocks"],
    num_per_prompt=50,
)
```

How it works:

1. **Video generation** — Wan 2.1 generates new task videos from your demonstration + text instructions
2. **Inverse dynamics** — GR00T IDM extracts actions from the generated videos
3. **Dataset** — synthetic demonstrations ready for training

One real demo → hundreds of diverse training episodes.

---

## GR00T Foundation Models

NVIDIA GR00T is a vision-language-action (VLA) foundation model. It sees, understands language, and acts:

```python
from strands_robots import create_policy

# GR00T on a remote Jetson
policy = create_policy("zmq://jetson:5555")

# GR00T with TensorRT acceleration
policy = create_policy(
    provider="groot",
    host="localhost",
    port=8000,
    use_tensorrt=True,
)
```

GR00T supports N1.5 and N1.6 checkpoints, with fp16/fp8/nvfp4 quantization for deployment.

---

---

## Zenoh Robot Mesh

Connect robots across machines with zero configuration:

```python
from strands_robots import robot_mesh

# Machine A
robot_mesh.start()
robot_mesh.publish("arm_1", {"joint_positions": [...]})

# Machine B (auto-discovers Machine A)
robot_mesh.start()
data = robot_mesh.subscribe("arm_1")
```

Zenoh uses multicast scouting for auto-discovery on local networks. No broker, no server, no configuration.

---

## Stereo Depth

Estimate depth from stereo cameras:

```python
from strands import Agent
from strands_robots import stereo_depth

agent = Agent(tools=[stereo_depth])
agent("Estimate depth from the stereo camera pair")
```

---

## Telemetry Streaming

Monitor robot state in real-time:

```python
from strands import Agent
from strands_robots import stream

agent = Agent(tools=[stream])
agent("Start streaming telemetry from the robot")
```

---

## What You Learned

- ✅ Custom policy providers with `register_policy()`
- ✅ DreamGen: one demo → many training episodes
- ✅ GR00T VLA foundation models
- ✅ Zenoh mesh for cross-machine robot networking
- ✅ Stereo depth and telemetry tools

---

**Congratulations!** 🎉 You've completed the full tutorial. You can now:

- Create and control 38 different robots
- Run in simulation or on real hardware
- Use 8 policy providers
- Control robots with natural language
- Record, train, and deploy
- Build production robot systems

Go build something amazing. 🤖
