---
description: Factory internals, custom backends, custom data_configs, lazy-import discipline.
---

# 9 — Advanced

- `Robot()` is a function — returns `Simulation` or `HardwareRobot` based on `mode=`.
- New backends: subclass `SimEngine`, call `register_backend(name, cls)`.
- New robots: JSON entry in `registry/robots.json` — usually no code change.
- New policies: `Policy` subclass + entry in `registry/policies.json`.
- Lazy imports are mandatory; CI fails on eager heavy-module loads.

## Factory internals

```python
# strands_robots/robot.py
def Robot(name, mode="sim", backend="mujoco", urdf_path=None,
          cameras=None, position=None, data_config=None,
          mesh=True, peer_id=None, **kwargs) -> Simulation | HardwareRobot: ...
```

Call path: `resolve_name(name)` (alias map) → validate → branch on `mode` → optional `init_mesh(...)`.
`STRANDS_ROBOT_MODE` env var overrides `mode=` (useful for forcing sim in CI).

## Custom simulation backend

```python
# 1. Subclass SimEngine (strands_robots/simulation/base.py)
class MyBackend(SimEngine):
    def create_world(self, ...): ...
    def add_robot(self, robot_name, ...): ...
    # implement 30+ abstract methods

# 2. Register and use
from strands_robots.simulation import register_backend
register_backend("mybk", MyBackend)
Robot("so100", backend="mybk")
```

## Adding a robot (registry entry)

```json
{
  "robots": {
    "my_arm": {
      "description": "My custom 7-DOF arm",
      "category": "arm",
      "joints": 7,
      "asset": { "robot_descriptions_module": "my_arm_mj_description" },
      "aliases": ["myarm"]
    }
  }
}
```

Asset options: `robot_descriptions_module` (auto-install from robot_descriptions.py) · `source: {type: "github"}` · `auto_download: false` (caller provides `urdf_path`).

## Adding a policy

```python
from strands_robots.policies import Policy, register_policy

class MyPolicy(Policy):
    provider_name = "my_provider"

    async def get_actions(self, observation_dict, instruction, **kwargs):
        return [{"joint_0": 0.5}]

    def set_robot_state_keys(self, keys): self._keys = keys

register_policy("my_provider", lambda: MyPolicy, aliases=["mine"])
# create_policy("my_provider") now works
```

For permanent providers add `{"module": "...", "class": "..."}` to `registry/policies.json`.

## Custom GR00T data_config

```json
// strands_robots/policies/groot/data_configs.json
{
  "my_robot_dualcam": {
    "video_keys": ["video.front", "video.wrist"],
    "state_keys": ["state.joint_pos", "state.gripper"],
    "action_keys": ["action.joint_pos", "action.gripper"]
  }
}
```

```python
create_policy("groot", data_config="my_robot_dualcam", port=5555)
```

## Lazy imports

Heavy modules (`mujoco`, `lerobot`, `torch`, `zenoh`) must not load at `import strands_robots`. The library uses PEP 562 `__getattr__` in `__init__.py`. CI guards this in `tests/test_init.py` — an eager heavy import fails the build.

## Authoring a tool

```python
from strands import tool

@tool
def my_tool(param: str) -> dict:
    """Short description the agent reads."""
    return {"status": "success", "content": [{"text": f"got {param}"}]}
```

Register in `strands_robots/tools/__init__.py`: `_LAZY_IMPORTS["my_tool"] = (".my_tool", "my_tool")`.

## New capabilities

| Feature | API |
|---------|-----|
| Cosmos 3 VLA | `create_policy("cosmos3", embodiment="droid", port=8000)` — see [Cosmos3Policy](../policies/cosmos3.md) |
| Synchronized multi-robot | `sim.run_multi_policy(policies={"left": p1, "right": p2}, duration=15.0)` |
| Benchmark eval | `sim.evaluate_benchmark("libero_spatial", robot_name="panda", n_episodes=20)` · `list_benchmarks()` |
| Resume dataset | `DatasetRecorder.resume(repo_id="user/my_dataset")` (requires lerobot ≥ 0.5.2) |

## See also

- [Architecture](../architecture.md) — module map and ABC contracts.
- [Custom policies](../policies/custom-policies.md) — full policy authoring walkthrough.
- [Cosmos3Policy](../policies/cosmos3.md) — NVIDIA Cosmos 3 omnimodal VLA.
- [Robot factory reference](../getting-started/robot-factory.md) — every kwarg.
- [Contributing](../contributing.md) — PR conventions, lint rules, hatch envs.
