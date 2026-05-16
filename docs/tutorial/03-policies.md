---
description: Mock, GR00T, LeRobot — three drop-in policies behind one ABC. Pick one, plug it into run_policy.
---

# 3 — Policies

A `Policy` decides what joint values to send. The `Simulation` you built in chapter 2
runs whatever `Policy` you hand it. This chapter shows the three that ship with
`strands-robots` and how to choose between them.

## TL;DR

```python
from strands_robots import Robot
from strands_robots.policies import create_policy

sim = Robot("so100")

# Three providers, same .get_actions() interface
mock   = create_policy("mock")                                      # always works
groot  = create_policy("groot", port=5555)                          # GR00T server on :5555
lerobot = create_policy("lerobot_local",
                        pretrained_name_or_path="lerobot/pi0_so100") # local checkpoint

# Plug any of them into the sim
sim.run_policy(instruction="pick up the cube", policy_provider="mock", duration=10.0)
```

## The Policy ABC

```python
# strands_robots/policies/base.py
class Policy(ABC):
    @abstractmethod
    async def get_actions(
        self,
        observation_dict: dict[str, Any],
        instruction: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None: ...

    @property
    def requires_images(self) -> bool: ...
```

Three abstract methods, one property. Every shipping policy implements them. There's
also a synchronous `get_actions_sync()` convenience wrapper for non-async callers.

## The three providers

### MockPolicy — for tests

```python
from strands_robots.policies import MockPolicy

policy = MockPolicy()

# Or via the factory
from strands_robots.policies import create_policy
policy = create_policy("mock")
```

`MockPolicy` returns sinusoidal joint traces. No model, no GPU, no network. It's how
the test suite verifies the rest of the pipeline without needing inference
infrastructure. Use it for:

- Unit tests
- Pipeline smoke checks
- "Does my recording loop produce the right schema?" experiments
- Demos where you just need *something* to move

### Gr00tPolicy — NVIDIA GR00T (N1.5 / N1.6 / N1.7)

```python
policy = create_policy(
    "groot",
    server_address="localhost:5555",   # GR00T inference server
    data_config="so100_dualcam",        # which embodiment config
)
```

`Gr00tPolicy` talks to a GR00T inference container over ZMQ (or HTTP for newer servers).
The container does the heavy lifting; this policy is a thin client. See
[GR00T policy](../policies/groot.md) for:

- The 25 supported embodiment data_configs
- ZMQ vs HTTP transport
- Container lifecycle (`build_image` / `download_checkpoint` / `start_container`
  helpers in `strands_robots.tools.gr00t_inference`)
- N1.7 wire-format specifics (batched-time observations, float32 state)

Requires `pip install "strands-robots[groot-service]"` plus the GR00T container running.

### LerobotLocalPolicy — direct HuggingFace LeRobot inference

```python
policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="lerobot/pi0_so100",   # any HF model_id
    device="cuda",
)
```

`LerobotLocalPolicy` runs the HF LeRobot inference loop in-process — no separate server.
Supports the LeRobot model zoo (ACT, Pi0, SmolVLA, Diffusion Policy, etc.).

Includes Real-Time Chunk (RTC) support and processor-bridge pre/post-processing that
matches LeRobot 0.4 and 0.5 conventions.

Requires `pip install "strands-robots[lerobot]"`. Because it loads HF models with
`trust_remote_code=True`, you must opt in:

```bash
export STRANDS_TRUST_REMOTE_CODE=1
```

Without that env var, `create_policy("lerobot_local", ...)` raises
`UntrustedRemoteCodeError` — see the security gate in
`strands_robots/policies/factory.py`.

## The factory

`create_policy(provider, **kwargs)` accepts:

- A provider name from `registry/policies.json`: `"mock"`, `"groot"`, `"lerobot_local"`.
- A smart URI: `"zmq://localhost:5555"` resolves to `"groot"`, etc.
- A runtime-registered name (see [Custom policies](../policies/custom-policies.md)).

The `**kwargs` are forwarded to the provider's constructor. See each provider's docs
for the exact parameters.

`list_providers()` returns every name `create_policy` will accept.

## Plugging a policy into a Simulation

`Simulation.run_policy(...)` and `Simulation.start_policy(...)` both accept either a
provider name or a Policy instance:

```python
# By name (the simulation calls create_policy internally)
sim.run_policy(
    instruction="pick up the cube",
    policy_provider="lerobot_local",
    pretrained_name_or_path="lerobot/pi0_so100",
    duration=15.0,
)

# Or pass an instance you constructed yourself
policy = create_policy("groot", server_address="localhost:5555")
sim.run_policy(instruction="pick up the cube", policy=policy, duration=15.0)
```

`run_policy` is synchronous — it blocks until the duration elapses or the policy
returns done. Use `start_policy` / `stop_policy` for async control:

```python
sim.start_policy(instruction="organize the table", policy_provider="mock")
# ... do other things, e.g. record metrics
sim.stop_policy()
```

## Eval mode

For benchmarking:

```python
result = sim.eval_policy(
    instruction="pick up the red cube",
    policy_provider="mock",
    num_episodes=10,
    randomize=True,
)
print(result["success_rate"], result["mean_reward"])
```

`eval_policy` randomises the world between episodes (if `randomize=True`), records
success/reward, and returns aggregate stats.

## Recap

- Three policies ship: `MockPolicy`, `Gr00tPolicy`, `LerobotLocalPolicy`.
- One `Policy` ABC. New providers are 1 file + 1 registry entry — see
  [Custom policies](../policies/custom-policies.md).
- `create_policy("name", ...)` is the only entry point you need.
- `Simulation.run_policy(...)` accepts both a provider name and an instance.

## See also

- [Tutorial 4 — AI agents](04-agents.md) — let the agent pick the policy and the
  instruction.
- [Policy providers overview](../policies/overview.md) — full provider matrix.
- [GR00T](../policies/groot.md) — server setup, embodiments, RTC.
- [LeRobot Local](../policies/lerobot-local.md) — supported models, RTC, TRC gate.
- [Custom policies](../policies/custom-policies.md) — write and register your own.
