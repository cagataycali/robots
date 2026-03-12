# Chapter 3: Policies

**Time:** 15 minutes · **Hardware:** None needed · **Level:** Intermediate

A robot without a policy is like a car without a driver. The **policy** is the brain — it takes observations and decides what actions to take.

---

## What Is a Policy?

A policy is a function:

```
observation → action
```

Given what the robot sees (camera images, joint positions), the policy outputs what the robot should do (target joint positions).

Policies can be:

- **Neural networks** trained on demonstrations (imitation learning)
- **Foundation models** like NVIDIA GR00T (vision-language-action)
- **Classical controllers** (PID, trajectory planning)
- **Mock generators** for testing

---

## Your First Policy

Start with the mock policy — it generates smooth sinusoidal motions, perfect for testing:

```python
from strands_robots import Robot, create_policy

robot = Robot("so100")
policy = create_policy("mock")

obs = robot.get_observation()
action = policy.get_actions(obs, instruction="wave hello")
robot.apply_action(action)
```

The mock policy ignores the instruction and the observation — it just makes the robot move smoothly. But the **interface** is identical to real policies.

---

## Auto-Resolution

`create_policy()` is smart about what you pass it:

```python
# HuggingFace model ID → LeRobot local inference
policy = create_policy("lerobot/act_aloha_sim_transfer_cube_human")

# ZMQ address → GR00T server
policy = create_policy("zmq://jetson:5555")

# gRPC address → LeRobot PolicyServer
policy = create_policy("localhost:8080")

# Just a name → mock for testing
policy = create_policy("mock")
```

You don't need to know which provider to use. Pass a string, get a policy.

---

## The Control Loop

Here's a complete policy-controlled robot:

```python
from strands_robots import Robot, create_policy

robot = Robot("so100")
policy = create_policy("mock")

for step in range(200):
    obs = robot.get_observation()
    action = policy.get_actions(obs, instruction="pick up the cube")
    robot.apply_action(action)
```

Every 20ms (50Hz), the robot:

1. Observes the world
2. Asks the policy what to do
3. Executes the action

---

## Available Providers

| Provider | What it does | When to use |
|---|---|---|
| `mock` | Smooth sinusoidal motions | Testing and development |
| `groot` | NVIDIA GR00T N1.5/N1.6 | Production VLA inference |
| `lerobot_local` | HuggingFace local inference | ACT, Pi0, SmolVLA, Diffusion |
| `lerobot_async` | gRPC to PolicyServer | Remote inference |
| `cosmos_predict` | NVIDIA Cosmos world model | Predictive control |
| `gear_sonic` | NVIDIA humanoid control | Whole-body @ 135Hz |
| `dreamgen` | GR00T-Dreams IDM + VLA | Augmented training |
| `dreamzero` | Zero-shot world model | No demonstrations needed |

---

## What You Learned

- ✅ A policy maps observations to actions
- ✅ `create_policy()` auto-resolves the right provider
- ✅ The control loop: observe → policy → act → repeat
- ✅ 8 providers from mock testing to production VLA

---

**Next:** [Chapter 4: AI Agents →](04-agents.md) — Control robots with natural language.
