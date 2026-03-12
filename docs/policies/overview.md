# Policy Providers Overview

Policies are the brains of your robots. They take observations and produce actions.

---

## The Interface

Every policy implements the same abstract class:

```python
from strands_robots.policies import Policy

class Policy(ABC):
    async def get_actions(self, observation_dict, instruction, **kwargs):
        """Given what the robot sees, decide what to do."""
        ...

    def get_actions_sync(self, observation_dict, instruction, **kwargs):
        """Synchronous convenience wrapper."""
        ...

    def set_robot_state_keys(self, robot_state_keys):
        """Configure the policy with robot state keys."""
        ...

    @property
    def provider_name(self) -> str:
        """Provider identifier."""
        ...
```

This means you can swap providers without changing your robot code.

---

## 8 Providers

All provider definitions live in `registry/policies.json`. No hardcoded if/elif chains.

| Provider | What it does | Best for |
|---|---|---|
| [**GR00T**](groot.md) | NVIDIA GR00T N1.5/N1.6 via ZMQ | Production deployment |
| [**LeRobot Local**](lerobot-local.md) | HuggingFace local inference | ACT, Pi0, Pi0-FAST, SmolVLA, Wall-X, X-VLA, SARM, Diffusion |
| **LeRobot Async** | gRPC to LeRobot PolicyServer | Remote inference |
| **Cosmos Predict** | NVIDIA Cosmos world model | Predictive control |
| [**GEAR-SONIC**](gear-sonic.md) | NVIDIA humanoid control | Whole-body @ 135Hz |
| **DreamGen** | GR00T-Dreams IDM + VLA | Augmented training |
| **DreamZero** | Zero-shot world action model | No demonstrations needed |
| **Mock** | Sinusoidal test motions | Development and testing |

---

## Auto-Resolution

You don't need to specify the provider. `create_policy()` figures it out from what you pass. Resolution rules are defined in `registry/policies.json` — URL patterns, HF orgs, shorthands, and aliases.

```python
from strands_robots import create_policy

# HuggingFace model ID → matches "lerobot" org → lerobot_local
policy = create_policy("lerobot/act_aloha_sim_transfer_cube_human")

# ZMQ address → matches "^zmq://" URL pattern → groot
policy = create_policy("zmq://jetson:5555")

# host:port → matches "^[\w.\-]+:\d+$" URL pattern → lerobot_async
policy = create_policy("localhost:8080")

# WebSocket → matches "^wss?://" URL pattern → dreamzero
policy = create_policy("ws://gpu:9000")

# Shorthand → direct lookup
policy = create_policy("mock")
```

---

## Provider Aliases

Many providers have aliases for convenience:

| Alias | Resolves to |
|---|---|
| `lerobot` | `lerobot_local` |
| `cosmos`, `cosmos_policy`, `nvidia_cosmos` | `cosmos_predict` |
| `sonic`, `gear`, `humanoid_sonic` | `gear_sonic` |
| `dream_zero`, `world_action_model` | `dreamzero` |
| `random`, `test` | `mock` |

---

## Register Your Own

```python
from strands_robots.policies import Policy, register_policy

class MyPolicy(Policy):
    async def get_actions(self, observation_dict, instruction, **kwargs):
        return [{"joint_0": 0.1, "joint_1": -0.2}]

    def set_robot_state_keys(self, keys):
        self.keys = keys

    @property
    def provider_name(self):
        return "my_policy"

register_policy("my_policy", lambda: MyPolicy, aliases=["custom"])

# Now use it
policy = create_policy("my_policy")
```

---

## Learn More

- [GR00T](groot.md) — NVIDIA foundation model
- [GEAR-SONIC](gear-sonic.md) — Humanoid whole-body control
- [LeRobot Local](lerobot-local.md) — HuggingFace models
- [Custom Policies](custom-policies.md) — Build your own
