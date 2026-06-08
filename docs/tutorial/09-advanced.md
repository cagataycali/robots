---
description: Factory internals, custom backends, custom data_configs, lazy-import discipline.
---

# 9 — Advanced

You've shipped something with the library. Now you want to extend it. This chapter
walks the internals.

## TL;DR

- `Robot()` is a function, not a class — it returns a `Simulation` or `HardwareRobot`
  depending on `mode=`.
- New simulation backends implement the `SimEngine` ABC.
- New robots are registry entries in `registry/robots.json` — usually no code change.
- New policies are `Policy` subclasses + a registry entry in `registry/policies.json`.
- Lazy imports are mandatory; CI fails if a heavy module loads at top-level.

## The factory in detail

```python
# strands_robots/robot.py
def Robot(
    name: str,
    mode: str = "sim",                # "sim" | "real" | "auto"
    backend: str = "mujoco",          # only "mujoco" today
    urdf_path: str | None = None,     # explicit MJCF/URDF override
    cameras: dict | None = None,      # real-hardware only (sim cameras: add_camera)
    position: list[float] | None = None,
    data_config: str | None = None,
    mesh: bool = True,                # auto-join the Zenoh mesh
    peer_id: str | None = None,
    **kwargs: Any,                    # forwarded to the backend ctor
) -> Simulation | HardwareRobot: ...
```

What happens when you call `Robot("so100")`:

1. `_normalize_mode(mode)` — case-insensitive, whitespace-stripped, raises on bad input.
2. `resolve_name("so100")` — alias map → canonical name from `registry/robots.json`.
3. `_validate_known_robot(...)` — refuses unknown names unless `urdf_path` is provided.
4. If `mode == "auto"`, `_auto_detect_mode(...)` probes USB for servo controllers.
5. Branch on mode:
   - `"sim"` → `Simulation(tool_name=f"{canonical}_sim", **kwargs)` →
     `_dispatch_action("create_world")` → `_dispatch_action("add_robot", {...})`.
   - `"real"` → `HardwareRobot(tool_name=canonical, robot=real_type, ...)`.
6. Optional mesh: `init_mesh(instance, mesh=mesh, peer_id=peer_id, ...)`.

`STRANDS_ROBOT_MODE` env var overrides the `mode` kwarg if set (case-insensitive).
This is convenient for CI ("force everything sim").

## Custom simulation backends

The simulation layer is backend-agnostic. `SimEngine` is the ABC:

```python
# strands_robots/simulation/base.py
class SimEngine(ABC):
    @abstractmethod
    def create_world(self, ...) -> SimWorld: ...
    @abstractmethod
    def add_robot(self, robot_name, ...) -> SimRobot: ...
    # ... 30+ more abstract actions
```

To add an Isaac Sim or Newton backend:

1. Create `strands_robots/simulation/{backend}/{backend}.py` implementing `SimEngine`.
2. Register it via `register_backend("isaac", IsaacSimulation)` from your module's
   `__init__.py`.
3. Inspect registered backends with `list_backends()`.
4. Now `Robot("so100", backend="isaac")` works.

Today only `mujoco` is implemented — see `strands_robots/simulation/mujoco/simulation.py`
for the reference implementation. Future backends will land here.

## Adding a robot

Robots live entirely in `strands_robots/registry/robots.json`. To add `my_arm`:

```json
{
  "robots": {
    "my_arm": {
      "description": "My custom 7-DOF arm",
      "category": "arm",
      "joints": 7,
      "asset": {
        "dir": "my_arm_assets",
        "model_xml": "my_arm.xml",
        "scene_xml": "scene.xml",
        "robot_descriptions_module": "my_arm_mj_description"
      },
      "aliases": ["myarm", "MA-7"]
    }
  }
}
```

The asset block tells the factory how to resolve the MJCF. Three options:

1. **`robot_descriptions_module`** (preferred): a Python module from
   [robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py)
   that the factory `pip install`s on demand.
2. **`source: {type: "github"}`**: a GitHub repository to clone.
3. **`auto_download: false`**: explicitly opt out — caller must provide `urdf_path`.

A test in `tests/test_registry_integrity.py` enforces that every registry entry has
exactly one of these three.

## Adding a policy

Subclass `Policy` and register it. Three abstract methods are required:
`get_actions`, `set_robot_state_keys`, and `provider_name`:

```python
# my_policies.py
from strands_robots.policies import Policy, register_policy

class MyPolicy(Policy):
    provider_name = "my_provider"   # required: identifies this policy in logs

    async def get_actions(self, observation_dict, instruction, **kwargs):
        return [{"joint_0": 0.5}]   # whatever your model returns

    def set_robot_state_keys(self, robot_state_keys):
        self._keys = robot_state_keys

register_policy("my_provider", lambda: MyPolicy, aliases=["mine"])

# Now create_policy("my_provider") works
```

For permanent providers, add an entry to `strands_robots/registry/policies.json`
instead of using `register_policy`. The JSON schema requires `module` and `class`
(plus optional `aliases`):

```json
{
  "my_provider": {
    "module": "my_policies",
    "class": "MyPolicy",
    "aliases": ["mine"]
  }
}
```

See [Custom policies](../policies/custom-policies.md) for the full authoring walkthrough.

## Custom data_configs (GR00T)

GR00T's data_configs are 27 embodiment definitions in
`strands_robots/policies/groot/data_configs.json`. Each entry maps a config name to
the keys / shapes that GR00T expects. Adding a new one is a JSON edit:

```json
{
  "my_robot_dualcam": {
    "video_keys": ["video.front", "video.wrist"],
    "state_keys": ["state.joint_pos", "state.gripper"],
    "action_keys": ["action.joint_pos", "action.gripper"]
  }
}
```

Then on the policy: `create_policy("groot", data_config="my_robot_dualcam", ...)`.

## Lazy imports

Heavy modules (`mujoco`, `lerobot`, `torch`, `zenoh`) must NOT load at
`import strands_robots` time. The library uses PEP 562 `__getattr__` in `__init__.py`
to defer them:

```python
# strands_robots/__init__.py
_LAZY = {
    "Robot": ("strands_robots.robot", "Robot"),
    "Simulation": ("strands_robots.simulation", "Simulation"),
    # ...
}

def __getattr__(name):
    if name in _LAZY:
        module, attr = _LAZY[name]
        return getattr(import_module(module), attr)
    raise AttributeError(name)
```

This means:

- `import strands_robots` is fast (~50ms even with all extras installed).
- `from strands_robots import MockPolicy` is also fast (light symbol).
- `from strands_robots import Simulation` triggers the MuJoCo import only at that line.

CI guards this with `tests/test_init.py` — adding a heavy eager import will fail.

## Authoring a tool

Tools (functions in `strands_robots/tools/`) are `@tool`-decorated callables a Strands
Agent can use directly:

```python
# strands_robots/tools/my_tool.py
from strands import tool

@tool
def my_tool(param: str) -> dict:
    """Short description the agent reads.

    Args:
        param: what it does.

    Returns:
        A dict with at least 'status' and 'content'.
    """
    return {"status": "success", "content": [{"text": f"got {param}"}]}
```

Register in `strands_robots/tools/__init__.py`:

```python
_LAZY_IMPORTS["my_tool"] = (".my_tool", "my_tool")
```

The tool is now `from strands_robots.tools import my_tool` and can be passed straight
to `Agent(tools=[my_tool, ...])`.

## New capabilities (highlights)

### Cosmos 3 policy provider  # requires GPU

`Cosmos3Policy` is a new first-class provider (PR #317). It connects to an
NVIDIA Cosmos 3 action-policy server over WebSocket:

```python
from strands_robots.policies import create_policy
from strands_robots import Robot

# pip install "strands-robots[cosmos3-service]"  # requires GPU
policy = create_policy("cosmos3", embodiment="droid", port=8000)

sim = Robot("panda")
sim.run_policy(
    robot_name="panda",
    policy_object=policy,
    instruction="grasp the mug",
    duration=20.0,
)
```

Supported embodiments: `droid`, `umi`, `av`, `bridge` (call `list_embodiments()`).
The `droid` embodiment maps to the Franka/DROID joint layout; use the `panda` sim
asset. See [Cosmos3Policy](../policies/cosmos3.md) and
`examples/cosmos3_sim_rollout.py`.

### Synchronized multi-robot control

`run_multi_policy` drives multiple robots in a single merged control loop, recording
one frame per step:

```python
from strands_robots.policies import create_policy
from strands_robots import Robot

sim = Robot("aloha")   # bimanual
policy_left  = create_policy("mock")
policy_right = create_policy("mock")

sim.run_multi_policy(
    policies={"left": policy_left, "right": policy_right},
    instructions="coordinate bimanual pick",
    duration=15.0,
    control_frequency=50.0,
)
```

### Benchmark evaluation

`evaluate_benchmark` runs a full benchmark protocol against a named benchmark spec:

```python
results = sim.evaluate_benchmark(
    "libero_spatial",
    robot_name="panda",
    policy_provider="mock",
    n_episodes=20,
    seed=42,
)
print(results)   # success_rate, per_episode details, etc.
```

Use `list_benchmarks()` to see registered benchmarks and
`register_benchmark_from_file(name, spec_path)` to add your own.
The `[benchmark-libero]` extra is required for LIBERO benchmarks.

### Resuming an existing dataset

`DatasetRecorder.resume` appends episodes to an existing LeRobot v3 dataset
(requires `lerobot >= 0.5.2`):

```python
from strands_robots.dataset_recorder import DatasetRecorder

# Resume appends to the existing dataset rather than creating a new one
recorder = DatasetRecorder.resume(
    repo_id="user/my_dataset",
    task="extended recording session",
)

for _ in range(10):
    recorder.add_frame(observation, action)
recorder.save_episode()
recorder.finalize()
```

## Recap

- `Robot()` factory: a function, branch on `mode`, attach optional mesh.
- New backends: subclass `SimEngine`, call `register_backend(...)`, inspect with
  `list_backends()`.
- New robots: JSON entry, asset block specifies fetch strategy.
- New policies: subclass `Policy` (implement `get_actions` + `set_robot_state_keys` +
  `provider_name`) + `policies.json` entry (`module` + `class`) or `register_policy(...)`.
- GR00T ships 27 embodiment data_configs.
- Lazy imports: required everywhere heavy.
- New tools: `@tool` decorator + entry in `tools/__init__.py`.
- NEW capabilities: `Cosmos3Policy`, `run_multi_policy`, `evaluate_benchmark`,
  `DatasetRecorder.resume`.

## See also

- [Architecture](../architecture.md) — module map and ABC contracts.
- [Custom policies](../policies/custom-policies.md) — full policy authoring walkthrough.
- [Cosmos3Policy](../policies/cosmos3.md) — NVIDIA Cosmos 3 omnimodal VLA.
- [Robot factory reference](../getting-started/robot-factory.md) — every kwarg.
- [Contributing](../contributing.md) — PR conventions, lint rules, hatch envs.
