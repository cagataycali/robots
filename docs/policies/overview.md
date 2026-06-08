---
description: The Policy ABC and the four providers that ship — MockPolicy, Gr00tPolicy, LerobotLocalPolicy, Cosmos3Policy.
---

# Policy providers

A `Policy` decides what action to send to a robot. `strands-robots` ships four
implementations of the same ABC, plus a factory that resolves them by name.

## TL;DR

```python
from strands_robots.policies import create_policy, list_providers

print(list_providers())
# ['cosmos3', 'groot', 'lerobot_local', 'mock', ...]

policy = create_policy("mock")                                  # always works
policy = create_policy("groot", port=5555,
                        data_config="so100_dualcam")
policy = create_policy("lerobot_local",
                        pretrained_name_or_path="lerobot/pi0_so100")
policy = create_policy("cosmos3", embodiment="droid", port=8000)
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
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    def requires_images(self) -> bool:
        return True  # default; override to False for state-only policies

    def reset(self, seed: int | None = None) -> None:
        pass  # default no-op; override to clear episode state

    # Convenience: synchronous wrapper around get_actions
    def get_actions_sync(self, ...) -> list[dict[str, Any]]: ...
```

Three abstract methods (`get_actions`, `set_robot_state_keys`, `provider_name`).
`requires_images` defaults to `True` — override to `False` for policies that only
need joint state. `reset` is a non-abstract no-op — override it to clear model state
between episodes. Async-first; `get_actions_sync` is a convenience wrapper for
synchronous callers and notebooks.

## The shipping providers

| Provider | Class | Module | When to use |
|----------|-------|--------|-------------|
| `mock` | `MockPolicy` | `policies.mock` | Tests, smoke checks, demos with no model. |
| `groot` | `Gr00tPolicy` | `policies.groot.policy` | NVIDIA GR00T (N1.5 / N1.6 / N1.7). |
| `lerobot_local` | `LerobotLocalPolicy` | `policies.lerobot_local.policy` | HuggingFace LeRobot (ACT, Pi0, SmolVLA, Diffusion, MolmoAct2). |
| `cosmos3` | `Cosmos3Policy` | `policies.cosmos3.policy` | NVIDIA Cosmos 3 omnimodal VLA. |

Each has its own page:

- [GR00T](groot.md)
- [LeRobot Local](lerobot-local.md)
- [Cosmos 3](cosmos3.md)

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

- A provider name from `registry/policies.json`: `"mock"`, `"groot"`,
  `"lerobot_local"`, `"cosmos3"`.
- A smart URI shortcut: `"zmq://localhost:5555"` resolves to `groot`;
  `"cosmos3://host:port"` resolves to `cosmos3`.
- A runtime-registered name (see `register_policy`).

`**kwargs` flow into the provider's constructor. See each provider's page for the
exact parameters.

### `list_providers() -> list[str]`

Every name `create_policy` will accept (JSON registry + runtime aliases).

### `register_policy(name, loader, aliases=None)`

Add a custom provider at runtime without editing the JSON. See
[Custom policies](custom-policies.md).

## Plugging into a Simulation

`Simulation.run_policy(...)` requires `robot_name`. Pass provider kwargs via
`policy_config={}` or pass a pre-built instance via `policy_object=`.

```python
# By provider name — provider kwargs go in policy_config={}
sim.run_policy(robot_name="so100",
               instruction="pick up the cube",
               policy_provider="mock",
               duration=10.0)

# GR00T service — kwargs in policy_config={}
sim.run_policy(robot_name="so100",
               instruction="pick up the cube",
               policy_provider="groot",
               policy_config={"port": 5555, "data_config": "so100_dualcam"},
               duration=10.0)

# Pre-built instance — pass via policy_object=
policy = create_policy("groot", port=5555, data_config="so100_dualcam")
sim.run_policy(robot_name="so100",
               instruction="pick up the cube",
               policy_object=policy,
               duration=10.0)
```

Same goes for `start_policy` / `eval_policy` / `run_multi_policy` — the interface is
consistent.

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

- [GR00T](groot.md) — server setup, embodiments, container lifecycle.
- [LeRobot Local](lerobot-local.md) — supported models, RTC, processor bridge.
- [Cosmos 3](cosmos3.md) — NVIDIA Cosmos 3 omnimodal VLA.
- [GEAR-SONIC](gear-sonic.md) — third-party VLA (status: external).
- [Custom policies](custom-policies.md) — write your own.
- [Tutorial 3 — Policies](../tutorial/03-policies.md) — guided walkthrough.
