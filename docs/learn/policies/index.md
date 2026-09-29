---
description: Which VLAs run where, the three known gaps, the Policy contract, the generated provider matrix, and how a policy is swapped.
---

# Policies

`run_policy` takes a provider name and a `policy_config`, in the simulator and on a real robot alike. Run first-hand at this commit:

| checkpoint | sim | real robot, in process |
|---|---|---|
| SmolVLA `lerobot/smolvla_base` | laptop, inline native embodiment ([First policy](../../start/first-policy.md)) | same call; fine-tune first, the base has no SO-101 stats |
| ACT `robotfuel/act_so101_t16b` | laptop, `embodiment="so101"` ([Home](../../index.md)) | same `policy_config`, through `run_policy` or the tool's `execute` |
| Pi0 `lerobot/pi0_base`, Pi0.5 `lerobot/pi05_base` | GPU; Pi0 via `camera_key_map` ([#4193](https://github.com/strands-labs/robots/issues/4193)); Pi0.5 crashes on main, [#4203](https://github.com/strands-labs/robots/pull/4203) fixes it | untested |
| MolmoAct2 `allenai/MolmoAct2-SO100_101` | GPU, `embodiment="so101"` | untested |
| GR00T N1.x, Cosmos 3 | as services: [groot](groot.md), [cosmos3](cosmos3.md) | [G1 driver](../hardware/unitree.md) |
| WBC, G1 | 0.665 m in 2 s at 50 Hz from a local directory; a Hub id is refused ([#4161](https://github.com/strands-labs/robots/issues/4161)) | [G1 driver](../hardware/unitree.md) |

Numbers and their sources: [so101](../../robots/so101.md), [unitree_g1](../../robots/unitree_g1.md), [unitree_go2](../../robots/unitree_go2.md). Three gaps are open: the [lerobot_local](lerobot-local.md) example used to refuse ([#4157](https://github.com/strands-labs/robots/issues/4157); its fence now runs); over the mesh `policy_config` travels but `embodiment` does not ([#4180](https://github.com/strands-labs/robots/issues/4180)), so run a Hub checkpoint on a real arm from the process that owns it; a policy with no language input still reports `instruction_read=True` ([#4159](https://github.com/strands-labs/robots/issues/4159)).

## The contract

```python title="strands_robots/policies/base.py (abridged)"
class Policy(ABC):
    control_frequency: float | None = None          # set by the runtime
    rtc_observed_delay_steps: int | None = None      # set by the runtime
    reads_instruction: ClassVar[bool] = True         # False: the words never shape actions
    instruction_free_actions: ClassVar[str | None] = None   # what a non-reader does
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

Generated from `strands_robots/registry/policies.json` and `pyproject.toml` at build time.

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

Smart strings work too: a HuggingFace id resolves to `lerobot_local` (`groot` / `cosmos3` for the `nvidia` org), `zmq://host:port` to `groot`, `ws://host:port` to [`remote`](remote.md). A misspelled keyword is a `TypeError` before any download; `lerobot_local` and `kimodo` refuse to build until `STRANDS_TRUST_REMOTE_CODE=1` is set.

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

The notes come from `reads_instruction = False`: a policy that never reads the words says so in every report, so an agent cannot relay a test motion as done.

On hardware, `start_task(instruction, policy_provider=..., **policy_config)` takes the provider string and `run_policy(create_policy(...))` a built object; [Agents](../agents.md) covers the approval gate.
