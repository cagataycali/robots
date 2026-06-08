---
description: Every public symbol grouped by module — Robot, registry, simulation, policies, tools, dataset_recorder, mesh.
---

# API reference

Every public symbol exported by `strands-robots`. For deep walkthroughs of how to use
them, see the linked tutorials.

## `strands_robots`

Top-level package. Lazy-loaded.

```python
import strands_robots
```

| Symbol | What | More |
|--------|------|------|
| `Robot(name, mode='sim', ...)` | Factory; returns a `Simulation` or `HardwareRobot`. | [Robot factory](getting-started/robot-factory.md) |
| `list_robots(category='all')` | Catalog query. | [Robot catalog](robots/index.md) |
| `Policy` | Policy ABC. | [Policies](policies/overview.md) |
| `MockPolicy` | Sinusoidal-trace mock. | [Policies](policies/overview.md) |
| `create_policy(provider, **kw)` | Policy factory. | [Policies](policies/overview.md) |
| `register_policy(name, loader, aliases=None)` | Runtime policy registration. | [Custom policies](policies/custom-policies.md) |
| `list_providers()` | Known policy providers. | [Policies](policies/overview.md) |
| `Simulation` (lazy) | The MuJoCo-backed AgentTool. | [Simulation overview](simulation/overview.md) |
| `Gr00tPolicy` (lazy) | NVIDIA GR00T client. | [GR00T](policies/groot.md) |

## `strands_robots.registry`

Robot and policy catalog.

```python
from strands_robots.registry import (
    list_robots,
    resolve_name,
    get_robot,
    has_sim,
    has_hardware,
    get_hardware_type,
    list_robots_by_category,
    list_aliases,
    format_robot_table,
    register_robot,
    unregister_robot,
    list_user_robots,
    list_policy_providers,
    resolve_policy,
    import_policy_class,
    build_policy_kwargs,
)
```

| Symbol | What |
|--------|------|
| `list_robots(category)` | All robots in a category, or `"all"`. |
| `resolve_name(name)` | Alias -> canonical name. |
| `get_robot(name)` | Full registry entry dict. |
| `has_sim(name)` / `has_hardware(name)` | Whether a robot is supported in sim / real. |
| `get_hardware_type(name)` | LeRobot type string for `mode="real"`. |
| `list_robots_by_category()` | Dict of category -> list of robot names. |
| `list_aliases()` | All 106 registered name aliases. |
| `format_robot_table()` | Pretty-printed table of robots for display. |
| `register_robot(name, entry)` | Add a user-defined robot at runtime. |
| `unregister_robot(name)` | Remove a previously registered robot. |
| `list_user_robots()` | Names registered via `register_robot`. |
| `list_policy_providers()` | Names of providers in `policies.json`. |
| `resolve_policy(uri)` | URI -> provider name. |
| `import_policy_class(provider)` | Lazy import of the provider's class. |
| `build_policy_kwargs(provider, **kw)` | Normalise and validate kwargs for a provider. |

## `strands_robots.simulation`

```python
from strands_robots.simulation import (
    Simulation,
    SimWorld,
    SimRobot,
    SimObject,
    SimCamera,
    create_simulation,
    list_backends,
    register_backend,
)
from strands_robots.simulation.base import SimEngine   # ABC
```

| Symbol | What |
|--------|------|
| `Simulation` | The `mujoco` backend's concrete class. Exposes 60+ agent actions. |
| `SimWorld`, `SimRobot`, `SimObject`, `SimCamera` | Shared dataclasses. |
| `create_simulation(backend='mujoco')` | Factory for non-Robot() construction. |
| `list_backends()` | Registered sim backends. |
| `register_backend(name, cls)` | Register a custom backend. |
| `SimEngine` | The ABC custom backends implement. |

Selected simulation actions beyond the basics:

| Action | What |
|--------|------|
| `run_policy(robot_name, ...)` | Blocking policy rollout. |
| `start_policy(robot_name, ...)` | Async policy rollout (background thread). |
| `stop_policy(robot_name)` | Stop a running policy. |
| `run_multi_policy(policies, ...)` | Synchronized multi-robot rollout, one merged frame per step. |
| `eval_policy(robot_name, n_episodes, ...)` | Multi-episode evaluation. |
| `evaluate_benchmark(benchmark_name, ...)` | Run a registered benchmark protocol. |
| `list_benchmarks()` | Names of registered benchmarks. |
| `register_benchmark_from_file(benchmark_name, spec_path)` | Load a benchmark spec from a YAML/JSON file. |
| `replay_episode(repo_id, robot_name, ...)` | Replay a recorded episode. |

## `strands_robots.hardware_robot`

```python
from strands_robots.hardware_robot import Robot, TaskStatus, RobotTaskState
```

| Symbol | What |
|--------|------|
| `Robot` (the class, not the factory) | Real-hardware AgentTool. |
| `TaskStatus` | Enum: `IDLE` / `CONNECTING` / `RUNNING` / `COMPLETED` / `STOPPED` / `ERROR`. |
| `RobotTaskState` | Dataclass with current task info (status, step count, error). |

Key methods on the hardware `Robot`:

| Method | What |
|--------|------|
| `start_task(instruction, policy_port, ...)` | Async task start; returns immediately. |
| `stop_task()` | Halt the running policy. |
| `get_task_status()` | Return `RobotTaskState`. |
| `cleanup()` | Stop tasks, shut down executor, close cameras, stop mesh. |

Construction and lifecycle: see [Hardware Robot Control](hardware/robot-control.md).

## `strands_robots.policies`

```python
from strands_robots.policies import (
    Policy,                  # ABC
    MockPolicy,
    create_policy,
    register_policy,
    list_providers,
    UntrustedRemoteCodeError,
)
from strands_robots.policies.groot import Gr00tPolicy
from strands_robots.policies.lerobot_local import LerobotLocalPolicy
from strands_robots.policies.cosmos3 import Cosmos3Policy
```

| Symbol | What |
|--------|------|
| `Policy` | ABC - `get_actions`, `set_robot_state_keys`, `requires_images`, `provider_name`. |
| `MockPolicy` | Sinusoidal-trace mock. `requires_images=False`. |
| `create_policy(provider, **kw)` | Resolve provider + construct. Accepts smart strings (`zmq://`, `cosmos3://`, HF `org/model`). |
| `register_policy(name, loader, aliases)` | Runtime registration. |
| `list_providers()` | All known provider names: `cosmos3`, `groot`, `lerobot_local`, `mock`, plus aliases. |
| `UntrustedRemoteCodeError` | Raised when `STRANDS_TRUST_REMOTE_CODE` is required but not set. |
| `Gr00tPolicy` | NVIDIA GR00T client (ZMQ service or local in-process). |
| `LerobotLocalPolicy` | HuggingFace LeRobot inference (ACT, Pi0, Pi0.5, SmolVLA, etc.). Requires `STRANDS_TRUST_REMOTE_CODE=1`. |
| `Cosmos3Policy` | NVIDIA Cosmos 3 omnimodal VLA over WebSocket. | [Cosmos3](policies/cosmos3.md) |

## `strands_robots.tools`

```python
from strands_robots.tools import (
    download_assets,
    gr00t_inference,
    lerobot_calibrate,
    lerobot_camera,
    lerobot_teleoperate,
    pose_tool,
    serial_tool,
    robot_mesh,
)
```

All are `@tool`-decorated callables. Each returns
`{"status": "...", "content": [{"text": "..."}]}`. See [Hardware tools](hardware/tools.md).

## `strands_robots.dataset_recorder`

```python
from strands_robots.dataset_recorder import DatasetRecorder, has_lerobot_dataset
```

| Symbol | What |
|--------|------|
| `DatasetRecorder.create(repo_id, fps, robot_features, action_features, task, ...)` | Factory — creates a new dataset. |
| `DatasetRecorder.resume(repo_id, root, task, ...)` | Classmethod — append episodes to an existing dataset. Requires `lerobot>=0.5.2`. |
| `recorder.add_frame(observation, action, task=...)` | Append one frame. |
| `recorder.save_episode()` | Finalise the current episode. |
| `recorder.clear_episode_buffer()` | Discard the current episode buffer without saving. |
| `recorder.finalize()` | Flush and close the dataset writer. |
| `recorder.push_to_hub(tags=None, private=False)` | Upload to HuggingFace. |
| `has_lerobot_dataset()` | Cached check whether `lerobot` is importable. |

See [Recording](recording.md).

## `strands_robots.mesh`

```python
from strands_robots.mesh import (
    init_mesh,
    Mesh,
    InputPublisher,
    InputReceiver,
)
```

| Symbol | What |
|--------|------|
| `init_mesh(robot, peer_id=None, ...)` | Attach a mesh to a `Robot()` instance. |
| `Mesh` | The mesh client class. |
| `InputPublisher` | Stream a teleoperator's actions over the mesh. |
| `InputReceiver` | Receive + apply remote teleoperator actions. |

`Mesh` exposes `peer_id`, `peers`, `alive`, `send`, `broadcast`, `tell`,
`emergency_stop`. See [Tutorial 5 - Multi-robot](tutorial/05-multi-robot.md).

## `strands_robots.benchmarks.libero`

```python
from strands_robots.benchmarks.libero import LiberoSuite
```

LIBERO benchmark adapter - task suites, BDDL parser, eval helper. Pulls heavy deps
only on first use; install with `pip install "strands-robots[benchmark-libero]"`.

## Environment variables

Reference of every env var the library reads:

| Variable | Purpose | Default |
|----------|---------|---------|
| `STRANDS_ASSETS_DIR` | Robot model asset cache. | `~/.strands_robots/assets/` |
| `STRANDS_ROBOT_MODE` | Force `Robot()` mode. | (unset -> kwarg honoured) |
| `STRANDS_TRUST_REMOTE_CODE` | Allow HF `trust_remote_code=True`. | unset -> blocked |
| `STRANDS_MESH` | Disable mesh globally when `false`. | `true` |
| `STRANDS_MESH_AUDIT_DIR` | Safety event audit log. | `~/.strands_robots/` |
| `MUJOCO_GL` | GL backend for MuJoCo renderer. | auto |
| `GROOT_API_TOKEN` | API token for GR00T cloud inference. Falls back to `Gr00tPolicy(api_token=...)`. | (unset) |
| `STRANDS_GROOT_WIRE_LOG` | Log raw ZMQ frames for GR00T debugging when `1`. | (unset) |

## See also

- [Architecture](architecture.md) - module map and ABC contracts.
- [Tutorial](tutorial/index.md) - concept walkthroughs.
- [Robot factory](getting-started/robot-factory.md) - full factory signature.
