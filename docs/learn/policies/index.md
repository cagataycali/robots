---
description: The Policy contract, the provider matrix generated from the registry, create_policy, and swapping a policy without touching the robot.
---

# Policies

By the end of this page you can name every provider the package ships, build one with `create_policy`, write your own in twenty lines, and swap providers by changing one string.

## Which policies run where

One `run_policy` call takes a provider name and a `policy_config`.

| family | providers | ran first-hand |
|---|---|---|
| vision-language-action models: SmolVLA, ACT, Pi0, MolmoAct2, GR00T N1.7, FLUX 3 Action | [lerobot_local](lerobot-local.md), [remote](remote.md), [flux3_action](flux3-action.md) | sim on a laptop ([First learned policy](../../start/first-policy.md), [so101](../../robots/so101.md)); the same call on the arm |
| world foundation models: Cosmos 3 | [cosmos3](cosmos3.md) | as a service, sim and hardware behind one endpoint |
| whole-body controllers | [wbc](wbc.md), [holosoma](holosoma.md), [protomotions](protomotions.md), [kimodo](kimodo.md) | sim ([unitree_g1](../../robots/unitree_g1.md)); the [G1 driver](../hardware/unitree.md) on hardware |
| reinforcement learning | [rl](rl.md) | sim ([unitree_go2](../../robots/unitree_go2.md)) |
| remote and planning | [remote](remote.md), [curobo](curobo.md), [moveit2](moveit2.md) | wherever the server runs |

Three gaps are open: the [lerobot_local](lerobot-local.md) example used to refuse ([#4157](https://github.com/strands-labs/robots/issues/4157); it runs now); over the mesh `policy_config` travels but `embodiment` does not ([#4180](https://github.com/strands-labs/robots/issues/4180)); a policy with no language input still reports `instruction_read=True` ([#4159](https://github.com/strands-labs/robots/issues/4159)).

## The contract

```python title="strands_robots/policies/base.py (abridged)"
class Policy(ABC):
    control_frequency: float | None = None          # set by the runtime
    rtc_observed_delay_steps: int | None = None      # set by the runtime
    reads_instruction: bool = True                   # False: the words never shape actions
    instruction_free_actions: str | None = None             # what a non-reader does
    requires_action_controller: ClassVar[str | None] = None # the engine installs it or refuses

    @abstractmethod
    async def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any) -> list[dict[str, Any]]: ...

    def get_actions_sync(self, observation_dict, instruction, **kwargs) -> list[dict[str, Any]]: ...

    @abstractmethod
    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None: ...

    def reset(self, seed: int | None = None) -> None: ...

    @classmethod
    def preflight(cls, observation_keys: set[str], **policy_config: Any) -> None: ...

    @property
    def requires_images(self) -> bool: ...            # planners return False
    @property
    def required_bodies(self) -> tuple[str, ...]: ... # a tracker names its anchor
    @property
    def children(self) -> tuple[Policy, ...]: ...     # a wrapper lists what it drives

    @property
    @abstractmethod
    def provider_name(self) -> str: ...
```

Abridged ([API reference](../../reference/api/policies.md)). `get_actions` returns the chunk: one dict per control tick, joint name to a python `float`. Planners read `target_pose`, `target_joints` or `target_velocity` instead of the instruction.

## Providers

Generated from `strands_robots/registry/policies.json` and `pyproject.toml` at build time; "Also spelled" lists the shorthands `create_policy` accepts.

{{providers:table}}

`composite` and `persistent` (two joint groups; a warm worker) resolve by module name, not the registry.

## Build one

```python
from strands_robots.policies import create_policy, list_providers

print(list_providers())
policy = create_policy("mock")
policy.set_robot_state_keys(["shoulder_pan", "elbow_flex"])
actions = policy.get_actions_sync({"shoulder_pan": 0.0, "elbow_flex": 0.0}, "wave")
print(len(actions), actions[0])
```

Smart strings work too: a HuggingFace id resolves to `lerobot_local` (`nvidia/cosmos3*` to `cosmos3`), `ws://` to [`remote`](remote.md); `"groot"` is refused, naming `lerobot_local(policy_type="groot")`. A misspelled keyword is a `TypeError` before any download; `lerobot_local` and `kimodo` need `STRANDS_TRUST_REMOTE_CODE=1`.

## Run one, then swap it

`run_policy` builds the policy from the provider name, runs `preflight` before any download, then drives the loop. A registered class is one more string; the robot and the scene do not change.

```python
from typing import Any

from strands_robots.policies import Policy, register_policy
from strands_robots.simulation import create_simulation


class HoldPolicy(Policy):
    reads_instruction = False
    instruction_free_actions = "a fixed pose on every joint"

    def __init__(self, angle: float = 0.3, **kwargs: Any) -> None:
        self.angle = angle
        self.keys: list[str] = []

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.keys = list(robot_state_keys)

    async def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [{k: self.angle for k in self.keys}]

    @property
    def requires_images(self) -> bool:
        return False

    @property
    def provider_name(self) -> str:
        return "hold"


register_policy("hold", lambda: HoldPolicy, aliases=["freeze"])

sim = create_simulation("mujoco")
sim.create_world()
sim.add_robot("so101")
for provider, config in (("mock", None), ("freeze", {"angle": 0.5})):
    result = sim.run_policy(robot_name="so101", policy_provider=provider, policy_config=config, instruction="wave",
                            n_steps=50, control_frequency=50.0)
    print(provider, result["status"], round(sim.get_observation("so101", skip_images=True)["1"], 3))
    print(result["content"][0]["text"].splitlines()[-1])
sim.cleanup()
```

You should see:

```text
mock success 0.493
Note: MockPolicy does not read the instruction. Its actions - a test motion on every joint - were commanded to the robot whatever the task says; nothing above means the task was performed.
freeze success 0.501
Note: HoldPolicy does not read the instruction. Its actions - a fixed pose on every joint - were commanded to the robot whatever the task says; nothing above means the task was performed.
```

The notes come from `reads_instruction = False`: a policy that never reads the words says so in every report, so an agent cannot relay a test motion as done.

On hardware, `start_task(instruction, policy_provider=..., **policy_config)` takes the provider string and `run_policy(create_policy(...))` a built object; [the operator gate](../agents.md#the-operator-gate) sits in front of both.
