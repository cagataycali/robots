---
description: The Policy ABC and the three providers that ship — MockPolicy, Gr00tPolicy, LerobotLocalPolicy.
---

# Policy providers

A `Policy` decides what action to send to a robot. `strands-robots` ships three
implementations of the same ABC, plus a factory that resolves them by name.

## TL;DR

```python
from strands_robots.policies import create_policy, list_providers

print(list_providers())
# ['groot', 'lerobot_local', 'mock', ...]

policy = create_policy("mock")                                  # always works
policy = create_policy("groot", server_address="localhost:5555")
policy = create_policy("lerobot_local",
                        pretrained_name_or_path="lerobot/pi0_so100")
```

## The ABC

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

    # Convenience: synchronous wrapper around get_actions
    def get_actions_sync(self, ...) -> list[dict[str, Any]]: ...
```

Two abstract methods, one property. Async-first; the synchronous wrapper handles
running event loops cleanly so it's safe in notebooks and sync callers.

## The shipping providers

| Provider | Class | Module | When to use |
|----------|-------|--------|-------------|
| `mock` | `MockPolicy` | `policies.mock` | Tests, smoke checks, demos with no model. |
| `groot` | `Gr00tPolicy` | `policies.groot.policy` | NVIDIA GR00T (N1.5 / N1.6 / N1.7). |
| `lerobot_local` | `LerobotLocalPolicy` | `policies.lerobot_local.policy` | HuggingFace LeRobot (ACT, Pi0, SmolVLA, Diffusion). |

Each has its own page:

- [GR00T](groot.md)
- [LeRobot Local](lerobot-local.md)

`MockPolicy` is documented inline below since it's tiny.

## MockPolicy

```python
from strands_robots.policies import MockPolicy
policy = MockPolicy()
```

Returns sinusoidal joint traces. No model load, no GPU, no network. Great for:

- Unit tests of the recording/eval/agent loop.
- Demos where you want *something* to move.
- Pipeline checks before plugging in a heavy provider.

The full source is `strands_robots/policies/mock.py` — about 50 lines. Use it as a
template when writing your own policy.

## Factory

```python
from strands_robots.policies import create_policy, list_providers, register_policy
```

### `create_policy(provider, **kwargs) -> Policy`

Accepts:

- A provider name from `registry/policies.json`: `"mock"`, `"groot"`, `"lerobot_local"`.
- A smart URI shortcut: `"zmq://localhost:5555"` resolves to `groot`.
- A runtime-registered name (see `register_policy`).

`**kwargs` flow into the provider's constructor. See each provider's page for the
exact parameters.

### `list_providers() -> list[str]`

Every name `create_policy` will accept (JSON registry + runtime aliases).

### `register_policy(name, loader, aliases=None)`

Add a custom provider at runtime without editing the JSON. See
[Custom policies](custom-policies.md).

## Plugging into a Simulation or HardwareRobot

`Simulation.run_policy(...)` and `HardwareRobot.run_policy(...)` accept either a
provider name or an instance:

```python
# By name (the simulation calls create_policy internally)
sim.run_policy(instruction="pick up the cube",
               policy_provider="mock", duration=10.0)

# Or pass an instance
policy = create_policy("groot", server_address="localhost:5555")
sim.run_policy(instruction="pick up the cube", policy=policy, duration=10.0)
```

Same goes for `start_policy` / `eval_policy` / `run_policy` on real hardware — the
interface is consistent.

## The trust_remote_code gate

`LerobotLocalPolicy` loads HuggingFace models with `trust_remote_code=True`, which
allows arbitrary code execution from the model repository. The factory enforces an
opt-in:

```bash
export STRANDS_TRUST_REMOTE_CODE=1
```

Without that env var, `create_policy("lerobot_local", ...)` raises
`UntrustedRemoteCodeError`. This is most important on real hardware.

The full list of remote-code providers is in `_HF_REMOTE_CODE_PROVIDERS` inside
`strands_robots/policies/factory.py`.

## See also

- [GR00T](groot.md) — server setup, embodiments, RTC, container lifecycle.
- [LeRobot Local](lerobot-local.md) — supported models, RTC, processor bridge.
- [GEAR-SONIC](gear-sonic.md) — third-party VLA (status: external).
- [Custom policies](custom-policies.md) — write your own.
- [Tutorial 3 — Policies](../tutorial/03-policies.md) — guided walkthrough.
