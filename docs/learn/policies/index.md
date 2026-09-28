---
description: The Policy contract, the provider matrix generated from the registry, create_policy, and how a policy is swapped without touching the robot.
---

# Policies

By the end of this page you can name every provider the package ships, build one with `create_policy`, write your own in twenty lines, and swap providers by changing one string. A policy turns an observation into joint targets; the robot never knows which provider produced them.

## The contract

```python title="strands_robots/policies/base.py (abridged)"
class Policy(ABC):
    control_frequency: float | None = None          # runtime sets it before the loop
    rtc_observed_delay_steps: int | None = None      # runtime sets it before each call
    reads_instruction: ClassVar[bool] = True         # False: the words never shape the actions

    @abstractmethod
    async def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any) -> list[dict[str, Any]]: ...

    def get_actions_sync(self, observation_dict, instruction, **kwargs) -> list[dict[str, Any]]: ...

    @abstractmethod
    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None: ...

    def reset(self, seed: int | None = None) -> None: ...

    @classmethod
    def preflight(cls, observation_keys: set[str], **policy_config: Any) -> None: ...

    @property
    def requires_images(self) -> bool: ...            # default True; planners return False

    @property
    def required_bodies(self) -> tuple[str, ...]: ... # default (); a whole-body tracker names its anchor

    @property
    def children(self) -> tuple[Policy, ...]: ...     # default (); a wrapper lists what it drives

    @property
    @abstractmethod
    def provider_name(self) -> str: ...
```

`get_actions` returns one action dict per control tick: joint name to a python `float` (or `list[float]` for a grouped actuator), never an array. The list is the action chunk; the runtime plays it at `control_frequency` and asks again.

Non-VLA providers read their goal from well-known keywords instead of the instruction: `target_pose` (`[x, y, z, qw, qx, qy, qz]`), `target_joints` (`{name: radians}`), `target_velocity` (`[vx, vy, omega]`), and `world_update` for collision-aware planners. Every provider ignores keywords it does not know, so one `policy_kwargs` dict travels across providers.

## Providers

Generated from `strands_robots/registry/policies.json` and `pyproject.toml` at build time. "Also spelled" lists the shorthands `create_policy` accepts.

{{providers:table}}

`composite` and `persistent` are not in the registry but resolve by module name: `CompositePolicy` merges two policies over disjoint joint groups (legs from `wbc`, arms from a manipulation policy); `PersistentPolicy` keeps a provider warm in a worker.

## Build one

```python
from strands_robots.policies import create_policy, list_providers

print(list_providers())
policy = create_policy("mock")
policy.set_robot_state_keys(["shoulder_pan", "elbow_flex"])
actions = policy.get_actions_sync({"shoulder_pan": 0.0, "elbow_flex": 0.0}, "wave")
print(len(actions), actions[0])
```

You should see:

```text
['cosmos3', 'curobo', 'groot', 'kimodo', 'lerobot_local', 'microduck', 'mock', 'moveit2', 'protomotions', 'remote', 'rl', 'wbc', 'wbc_gait']
8 {'shoulder_pan': 0.0, 'elbow_flex': 0.4330127018922193}
```

`create_policy` also takes smart strings: a HuggingFace model id resolves to `lerobot_local` (or `groot` / `cosmos3` for the `nvidia` org), `zmq://host:port` to `groot`, `ws://host:port` to [`remote`](remote.md). A misspelled keyword is a `TypeError` before anything downloads. `lerobot_local` and `kimodo` load models with `trust_remote_code=True` and refuse to build until `STRANDS_TRUST_REMOTE_CODE=1` is set (refusal code `TRUST_REMOTE_CODE_REQUIRED`).

## Run one in the simulator

`run_policy` builds the policy from the provider name, runs `preflight` against the observation keys before any weights download, then drives the loop.

```python
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco", mesh=False)
sim.create_world()
sim.add_robot("so101")
result = sim.run_policy(robot_name="so101", policy_provider="mock", instruction="wave", n_steps=20, control_frequency=50.0)
print(result["status"])
print(result["content"][0]["text"])
sim.cleanup()
```

You should see:

```text
success
Policy complete on 'so101'
MockPolicy | wave
0.4s | 20 steps | sim_t=0.400s
Note: MockPolicy does not read the instruction. Its actions - a test motion on every joint - were commanded to the robot whatever the task says; nothing above means the task was performed.
```

The last line comes from `reads_instruction = False`: a policy that never reads the words says so in every task report, so an agent cannot relay a test motion as a done task.

## Swap the policy, keep the robot

Nothing in the robot or scene changes between providers. Register your class once; it is one more string.

```python
from typing import Any

from strands_robots.policies import Policy, register_policy
from strands_robots.simulation import create_simulation


class HoldPolicy(Policy):
    """Command every joint to one fixed angle. No cameras, no instruction."""

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
    result = sim.run_policy(robot_name="so101", policy_provider=provider, policy_config=config, n_steps=50, control_frequency=50.0)
    print(provider, result["status"], round(sim.get_observation("so101", skip_images=True)["1"], 3))
sim.cleanup()
```

You should see:

```text
mock success 0.493
freeze success 0.501
```

The same swap works on hardware: `Robot("so101", mode="real").start_task(instruction, policy_provider=...)` takes the same provider string, and `run_policy(create_policy(...))` a built object. See [Agents](../agents.md) for the approval gate a real robot adds.

## Pick a provider

| you have | use |
|---|---|
| a LeRobot checkpoint (ACT, diffusion, pi0, SmolVLA, GR00T N1.7, MolmoAct2) | [lerobot-local](lerobot-local.md) |
| a GR00T inference server | [groot](groot.md) |
| a Cosmos 3 policy server | [cosmos3](cosmos3.md) |
| a Cartesian or joint goal and a GPU | [curobo](curobo.md) |
| a Cartesian or joint goal and ROS 2 | [moveit2](moveit2.md) |
| a Unitree G1 to walk | [wbc](wbc.md) |
| a Unitree G1 and a motion prompt | [kimodo](kimodo.md) then [protomotions](protomotions.md) |
| a Microduck biped | [microduck](microduck.md) |
| an actor from `create_trainer` | [rl](rl.md) |
