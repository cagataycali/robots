---
description: The Policy contract, the provider matrix generated from the registry, create_policy, and how a policy is swapped without touching the robot.
---

# Policies

By the end of this page you can pick a provider for what you have, build one with `create_policy`, write your own in twenty lines, and swap providers by changing one string; one `run_policy` call takes the provider name and a `policy_config`.

## Which provider

| you have | start with | because |
|---|---|---|
| nothing yet, a laptop | `mock` | no download; it proves the loop and its report says it did not read the instruction |
| a Hub checkpoint for your arm (ACT, SmolVLA, Pi0, MolmoAct2) | [`lerobot_local`](lerobot-local.md) with `embodiment=` | runs in this process; the [embodiment map](../../concepts/embodiments.md) speaks sim and real |
| a GPU on another machine | [`remote`](remote.md), `create_policy("ws://gpu:8765")` | the robot host keeps the loop and the gate, the GPU host runs the model |
| a G1 or another humanoid | [`wbc`](wbc.md) | velocity commands in, whole-body joint targets out |
| a target pose, no model | [`curobo`](curobo.md) | a planner reads `target_pose`, not words |
| a Cosmos endpoint | [`cosmos3`](cosmos3.md) | a world model behind one URL |
| a task the arm has never seen | [Teach it](../../start/teach-it.md) | no checkpoint learns your task from a page; record and train first |

One gap is open: over the mesh `policy_config` travels but `embodiment` does not ([#4180](https://github.com/strands-labs/robots/issues/4180)), so run a Hub checkpoint on a real arm from the process that owns it. Deprecated providers (`moveit2`, `kimodo`, `protomotions`) stay in the table until 0.7; their pages name the replacement.

## The contract

```python title="strands_robots/policies/base.py (abridged)"
class Policy(ABC):
    control_frequency: float | None = None          # set by the runtime
    rtc_observed_delay_steps: int | None = None
    reads_instruction: bool = True                   # False: the words never shape actions
    instruction_free_actions: str | None = None             # what a non-reader does
    requires_action_controller: ClassVar[str | None] = None # the engine installs it or refuses

    @abstractmethod
    async def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any) -> list[dict[str, Any]]: ...

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

Abridged ([API reference](../../reference/api/policies.md)). `get_actions` returns the chunk: one dict per control tick, joint name to a `float`. Planners read `target_pose` or `target_joints`, not the instruction.

## Providers

From the registry at build time; "Also spelled" lists the shorthands `create_policy` accepts.

{{providers:table}}

`composite` and `persistent` resolve by module name, outside the registry.

## Build one

```python
from strands_robots.policies import create_policy, list_providers

print(list_providers())
policy = create_policy("mock")
policy.set_robot_state_keys(["shoulder_pan", "elbow_flex"])
actions = policy.get_actions_sync({"shoulder_pan": 0.0, "elbow_flex": 0.0}, "wave")
print(len(actions), actions[0])
```

Smart strings: a Hub id resolves to `lerobot_local`, `ws://` to [`remote`](remote.md). A misspelled keyword is a `TypeError` before any download; `lerobot_local` needs `STRANDS_TRUST_REMOTE_CODE=1`.

## Run one, then swap it

`run_policy` builds the policy from the provider name, runs `preflight` before any download, then drives the loop; a registered class is one more string:

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

sim = create_simulation("mujoco", mesh=False)
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

The notes come from `reads_instruction = False`: a policy that never reads the words says so in every report, so an agent cannot relay a test motion as done. On hardware `start_task(instruction, policy_provider=..., **policy_config)` takes the provider string and `run_policy(policy_object=...)` a built object, behind the [gate](../agents.md).
