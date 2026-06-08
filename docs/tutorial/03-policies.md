---
description: Mock, GR00T, LeRobot, Cosmos3 — four drop-in policies behind one ABC. Pick one, plug it into run_policy.
---

# 3 — Policies

```python
from strands_robots import Robot
from strands_robots.policies import create_policy

sim = Robot("so100")

mock    = create_policy("mock")
groot   = create_policy("groot", port=5555, data_config="so100_dualcam")       # requires GPU
lerobot = create_policy("lerobot_local",
                        pretrained_name_or_path="lerobot/pi0_so100")           # requires GPU
cosmos3 = create_policy("cosmos3", embodiment="droid", port=8000)              # requires GPU

# Provider name: kwargs go in policy_config={}
sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_provider="mock", duration=10.0)

# Prebuilt instance: pass via policy_object=
sim.run_policy(robot_name="so100", instruction="pick up the cube",
               policy_object=groot, duration=10.0)                             # requires GPU
```

## Providers

| Provider | Extra | Notes |
|----------|-------|-------|
| `mock` | — | Sinusoidal joints. No GPU, no network. `requires_images=False`. |
| `groot` | `groot-service` | ZMQ to GR00T inference container. 27 `data_config` values. N1.5–N1.7. |
| `lerobot_local` | `lerobot` | In-process inference. ACT, Pi0, SmolVLA, Diffusion, etc. Needs `STRANDS_TRUST_REMOTE_CODE=1`. |
| `cosmos3` | `cosmos3-service` | WebSocket to Cosmos 3 server. Embodiments: `droid`, `umi`, `av`, `bridge`. |

`list_providers()` returns every name `create_policy` accepts, including smart URIs (`zmq://localhost:5555`, `cosmos3://host:8000`).

## Policy ABC

```python
class Policy(ABC):
    @abstractmethod
    async def get_actions(self, observation_dict, instruction, **kwargs) -> list[dict]: ...
    @abstractmethod
    def set_robot_state_keys(self, keys: list[str]) -> None: ...
    @property
    @abstractmethod
    def provider_name(self) -> str: ...
    @property
    def requires_images(self) -> bool: return True   # override to False if unneeded
    def get_actions_sync(self, *args, **kwargs): ...  # sync convenience wrapper
```

## Async vs. sync control

`run_policy` blocks until `duration` elapses. For non-blocking control:

```python
sim.start_policy(robot_name="so100", instruction="organize the table",
                 policy_provider="mock")
# ... do other work ...
sim.stop_policy("so100")
```

## Eval

```python
result = sim.eval_policy(
    robot_name="so100",
    instruction="pick up the red cube",
    policy_provider="mock",
    n_episodes=10,   # NOT num_episodes
    max_steps=300,   # NOT duration
)
print(result["success_rate"], result["mean_reward"])
```

## See also

- [Tutorial 4 — AI agents](04-agents.md) — let the agent pick the policy.
- [Policy providers overview](../policies/overview.md) — full provider matrix.
- [GR00T](../policies/groot.md) — container setup, 27 embodiments, ZMQ wire format.
- [LeRobot Local](../policies/lerobot-local.md) — supported models, RTC.
- [Cosmos3](../policies/cosmos3.md) — embodiment mapping, container setup.
- [Custom policies](../policies/custom-policies.md) — write and register your own.
