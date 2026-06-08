---
description: Implement the Policy ABC, register the provider, plug it into Robot.run_policy. Full walkthrough.
---

# Custom policies

Three steps: subclass `Policy`, register it, use it. Total of about 80 lines.

## TL;DR

```python
# my_policy.py
from typing import Any
from strands_robots.policies import Policy, register_policy

class MyPolicy(Policy):
    async def get_actions(self, observation_dict, instruction, **kwargs):
        return [{"motor.0": 0.5, "motor.1": -0.2}]

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._keys = robot_state_keys

    @property
    def provider_name(self) -> str:
        return "my_provider"

    @property
    def requires_images(self) -> bool:
        return False

register_policy("my_provider", lambda: MyPolicy, aliases=["mine"])
```

```python
# user_code.py
import my_policy            # triggers the registration
from strands_robots.policies import create_policy
policy = create_policy("my_provider")
```

## Step 1 — subclass `Policy`

```python
from typing import Any
from strands_robots.policies import Policy

class GreedyPolicy(Policy):
    """Always return zero actions — the dumbest possible baseline."""

    def __init__(self, **kwargs: Any) -> None:
        self._keys: list[str] = []

    async def get_actions(
        self,
        observation_dict: dict[str, Any],
        instruction: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        # Return a single-action chunk where every joint sits at zero.
        return [{key: 0.0 for key in self._keys}]

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._keys = robot_state_keys

    @property
    def provider_name(self) -> str:
        return "greedy"

    @property
    def requires_images(self) -> bool:
        return False
```

Three abstract methods (`get_actions`, `set_robot_state_keys`, `provider_name`) plus
one overrideable property (`requires_images`, default `True`). `reset(seed=None)` is
provided by the base class as a no-op — override it if your model carries episode
state. The synchronous wrapper `get_actions_sync` is also provided by the base class.

## Step 2 — register

There are two ways to register a provider:

### Runtime registration (good for prototypes)

```python
from strands_robots.policies import register_policy

register_policy("greedy", lambda: GreedyPolicy, aliases=["zero"])
```

The lambda defers the class import — useful when your policy has heavy dependencies
you don't want to load until `create_policy` is called.

### JSON registration (for permanent providers)

Add an entry to `strands_robots/registry/policies.json`:

```json
{
  "greedy": {
    "module": "my_pkg.my_policy",
    "class": "GreedyPolicy",
    "shorthands": ["zero"],
    "description": "Zero-action baseline."
  }
}
```

The factory imports the class from `module` lazily on first use. Use separate `module`
and `class` keys — not a single dotted path.

## Step 3 — use it

```python
from strands_robots import Robot
from strands_robots.policies import create_policy

policy = create_policy("greedy")        # or "zero"
sim = Robot("so100")
sim.run_policy(robot_name="so100",
               instruction="do nothing",
               policy_object=policy,
               duration=5.0)
```

## What the simulation does for you

When you pass a `Policy` (or provider name) to `run_policy`, the simulation:

1. Calls `policy.set_robot_state_keys(...)` with the joint names from the registered
   robots.
2. On each control tick:
   - Builds an `observation_dict` with cameras + state.
   - Awaits `policy.get_actions(observation_dict, instruction, ...)`.
   - Applies the first action in the returned chunk.
   - Repeats every `1/control_frequency` seconds.
3. After `duration` seconds (or when the policy raises a "done" signal), stops.

Your `get_actions` implementation can:

- Return a single-element list (one action) — re-invoked every tick.
- Return a multi-element list (action chunk) — the sim consumes the chunk over
  `len(chunk) * action_dt` seconds before re-invoking.
- Use `**kwargs` to receive policy-specific overrides from the sim/agent.

## Tips

- **Async-first.** Even for synchronous models, declare `async def get_actions`. The
  base class provides `get_actions_sync` for non-async callers; the sim and agent
  use the async path.
- **Don't import heavy modules at module top-level.** Defer them to your `__init__`
  or `get_actions` so non-users of your policy don't pay the import cost.
- **Test with a `MockPolicy`-shaped fake.** Look at `strands_robots/policies/mock.py`
  — it's the smallest possible reference implementation.
- **Document the wire format.** If your policy talks to a server, document the
  request/response schema so future maintainers can swap servers.

## Common patterns

### Wrapping an existing PyTorch model

```python
import torch
from strands_robots.policies import Policy

class MyTorchPolicy(Policy):
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self._model = torch.load(checkpoint_path).to(device)
        self._model.eval()
        self._device = device
        self._keys: list[str] = []

    async def get_actions(self, observation_dict, instruction, **kwargs):
        with torch.no_grad():
            obs_tensor = self._build_obs(observation_dict).to(self._device)
            action_tensor = self._model(obs_tensor)
        return [self._tensor_to_action(action_tensor)]

    def set_robot_state_keys(self, robot_state_keys):
        self._keys = robot_state_keys

    @property
    def provider_name(self) -> str:
        return "my_torch"

    @property
    def requires_images(self) -> bool:
        return True
```

### Talking to a custom inference server

Look at `strands_robots/policies/groot/client.py` — it's a clean reference for ZMQ
wire-protocol handling, msgpack serialisation, and request correlation.

## See also

- [Policy overview](overview.md) — the ABC contract.
- [Tutorial 9 — Advanced](../tutorial/09-advanced.md) — registry mechanics.
- [Architecture](../architecture.md) — where policies sit in the module map.
- `strands_robots/policies/mock.py` — minimal reference impl.
