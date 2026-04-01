# AGENTS.md — strands-labs/robots

## Overview

`strands-robots` is a robot control and simulation library for [Strands Agents](https://strandsagents.com). It provides policy inference, simulation (MuJoCo), dataset recording (LeRobot format), teleoperation, and calibration tools for 38+ robots.

## Project Dashboard

**Board**: https://github.com/orgs/strands-labs/projects/2
**Project ID**: `PVT_kwDOD151Fs4BSRJP`

> **RULE**: ALWAYS use the project board to track work. When creating follow-up items,
> create GitHub issues and add them to this board with Status + Priority set.
> Never track work only in local markdown — the board is the source of truth.

## Repository Structure

```
strands_robots/
├── __init__.py            # Public API: Robot, list_robots, resolve_name
├── factory.py             # Robot("so100") → sim or hardware dispatch
├── robot.py               # HardwareRobot class (real robot control)
├── _async_utils.py        # Coroutine resolution helpers
├── utils.py               # require_optional(), shared utilities
│
├── policies/              # Policy providers (pluggable via registry)
│   ├── base.py            # Abstract Policy base class
│   ├── factory.py         # create_policy() factory + registry
│   ├── mock.py            # MockPolicy for testing (random actions)
│   ├── groot/             # NVIDIA GR00T N1.5/N1.6/N1.7 inference
│   │   ├── policy.py      # Gr00tPolicy (ZMQ + HTTP modes)
│   │   ├── client.py      # Gr00tInferenceClient
│   │   ├── data_config.py # Gr00tDataConfig + ModalityConfig
│   │   └── data_configs.json  # 25 robot embodiment configs
│   └── lerobot_local/     # HuggingFace LeRobot direct inference
│       ├── policy.py      # LerobotLocalPolicy (RTC support)
│       ├── processor.py   # ProcessorBridge (pre/post pipelines)
│       └── resolution.py  # Policy class resolution (v0.4/v0.5)
│
├── simulation/            # Simulation backends
│   ├── base.py            # SimulationBackend ABC
│   ├── factory.py         # create_simulation() dispatch
│   ├── models.py          # SimWorld, SimRobot, SimObject, SimCamera dataclasses
│   ├── model_registry.py  # URDF/MJCF path resolution
│   └── mujoco/            # MuJoCo backend (primary)
│       ├── simulation.py  # Simulation(AgentTool) — 35 actions via NL
│       ├── backend.py     # _ensure_mujoco() lazy loader
│       ├── mjcf_builder.py# Procedural MJCF XML generation
│       ├── policy_runner.py # run_policy, eval_policy, replay_episode
│       ├── recording.py   # start/stop_recording → LeRobot dataset
│       ├── rendering.py   # RGB + depth offscreen rendering
│       ├── randomization.py # Domain randomization (colors, physics, lighting)
│       ├── scene_ops.py   # Inject/eject objects & cameras into live scenes
│       └── tool_spec.json # AgentTool JSON schema (35 actions)
│
├── assets/                # Robot asset manager
│   ├── __init__.py        # resolve_model_path(), list_available_robots()
│   └── download.py        # Auto-download from MuJoCo Menagerie
│
├── dataset_recorder.py    # DatasetRecorder → LeRobot v3 parquet + video
│
├── registry/              # JSON registry for robots + policies
│   ├── robots.json        # 38 robots, 120+ aliases, asset paths
│   ├── robots.py          # get_robot(), list_robots(), resolve_name()
│   ├── policies.json      # Policy provider registry
│   ├── policies.py        # get_policy_info(), list_policies()
│   └── loader.py          # JSON loader utilities
│
└── tools/                 # Strands @tool functions (for Agent use)
    ├── download_assets.py # Download robot meshes from Menagerie/GitHub
    ├── gr00t_inference.py # GR00T inference tool
    ├── lerobot_calibrate.py
    ├── lerobot_camera.py
    ├── lerobot_teleoperate.py
    ├── pose_tool.py
    └── serial_tool.py

tests/                     # Unit tests
tests_integ/               # Integration tests (GPU + model weights)
```

## Setup & Development

### Using uv (recommended)

```bash
# Clone and enter
git clone git@github.com:strands-labs/robots.git && cd robots

# Create env + install dev deps (uses .python-version=3.12 + uv.lock)
uv sync --extra dev

# Install with simulation support
uv sync --extra sim --extra dev

# Install with everything (sim + lerobot + groot)
uv sync --extra all --extra dev

# Or one-shot editable install
uv pip install -e ".[all,dev]"
```

> **Note**: Python >=3.12 is required (enforced by `requires-python` and `.python-version`).
> `uv.lock` is committed — all contributors get identical dependency versions.

### Optional extras

| Extra | What it installs | When you need it |
|-------|-----------------|------------------|
| `sim` | `mujoco`, `robot_descriptions`, `opencv`, `Pillow` | Simulation (MuJoCo) |
| `lerobot` | `lerobot>=0.5` | LeRobot policy inference + dataset recording |
| `groot-service` | `pyzmq`, `msgpack` | NVIDIA GR00T inference |
| `all` | All of the above | Full development |
| `dev` | `sim` + `pytest`, `ruff`, `mypy` | Running tests + linting |

## Testing

### Run unit tests

```bash
# All unit tests (34 tests, ~1s)
uv run pytest tests/ -v

# Specific test files
uv run pytest tests/test_factory.py -v      # 22 tests — Robot factory, registry, mode detection
uv run pytest tests/test_mujoco_e2e.py -v   # 12 tests — MuJoCo physics, rendering, policy loop
uv run pytest tests/test_registry.py -v     # Registry resolution, aliases
uv run pytest tests/test_policies.py -v     # Policy creation, mock policy
uv run pytest tests/test_utils.py -v        # Utility functions
```

### Run integration tests (needs GPU + model weights)

```bash
uv run pytest tests_integ/ -v --timeout=300
```

### What the tests cover

**`test_factory.py`** (22 tests):
- Name resolution: canonical, alias, case-insensitive, hyphen-to-underscore
- `list_robots(mode=)`: all, sim, real, both — verifies registry filtering
- Robot registry: so100 exists, all aliases valid, robot count, descriptions
- Auto-detect mode: defaults to sim, env override (`STRANDS_ROBOT_MODE`), case-insensitive
- Robot factory: `Robot()` is callable (AgentTool), unknown backend raises, newton raises NotImplementedError
- URDF path passthrough: `Robot("so100", urdf_path="/custom/path.xml")`
- Top-level import: `from strands_robots import Robot`

**`test_mujoco_e2e.py`** (12 tests):
- Simulation ABC: all required methods exist on base class
- Shared dataclasses: SimWorld, SimRobot, SimObject, SimCamera, TrajectoryStep
- Physics: step advances time, position actuators move joints, contacts detected, reset zeros time
- Rendering: RGB frames (H×W×3 uint8), depth frames (H×W float32)
- Mock policy loop: generates actions, full observe→act loop, loop with rendering
- Domain randomization: color randomization changes model properties

### Manual E2E validation

```bash
# Quick smoke test — Robot → MuJoCo → physics
uv run python3 -c "
from strands_robots import Robot
sim = Robot('unitree_g1')
print(sim.get_state()['content'][0]['text'])
sim.step(n_steps=100)
sim.render(width=320, height=240)
sim.destroy()
print('✅ MuJoCo E2E works')
"

# Full Agent integration — natural language → simulation
uv run python3 -c "
from strands_robots import Robot
from strands import Agent
robot = Robot('so100')
agent = Agent(tools=[robot])
result = agent('Get the simulation state and run mock policy for 1 second in fast mode on so100')
print(result)
robot.destroy()
"

# Policy + video recording
uv run python3 -c "
from strands_robots import Robot
sim = Robot('so100')
result = sim.run_policy(
    robot_name='so100',
    policy_provider='mock',
    instruction='pick up the red cube',
    duration=2.0,
    fast_mode=True,
    record_video='/tmp/so100_demo.mp4',
    video_fps=30,
)
print(result['content'][0]['text'])
sim.destroy()
"

# Dataset recording (LeRobot v3 format)
uv run python3 -c "
from strands_robots import Robot
sim = Robot('so100')
sim.start_recording(repo_id='local/demo', task='pick cube', root='/tmp/demo_dataset')
sim.run_policy(robot_name='so100', policy_provider='mock', instruction='pick cube', duration=2.0, fast_mode=True)
sim.stop_recording()
sim.destroy()
# Verify: /tmp/demo_dataset/meta/info.json + data/chunk-000/file-000.parquet
import json
info = json.load(open('/tmp/demo_dataset/meta/info.json'))
print(f'✅ Dataset: {info[\"total_frames\"]} frames, {info[\"total_episodes\"]} episodes')
"
```

### Supported robots for simulation

Any robot in `registry/robots.json` with an `asset` field works. Assets are auto-downloaded from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) on first use via `robot_descriptions`.

```bash
# List all robots
uv run python3 -c "from strands_robots import list_robots; [print(r['name']) for r in list_robots(mode='sim')]"
```

Key robots tested: `so100`, `unitree_g1` (30 joints), `panda` (Franka), `unitree_h1`, `aloha`.

## The 5-Line Promise

```python
from strands_robots import Robot
from strands import Agent

robot = Robot("so100")            # → MuJoCo sim, auto-downloads assets
agent = Agent(tools=[robot])      # → 35 simulation actions as AgentTool
agent("pick up the red cube")     # → agent orchestrates sim via natural language
```

> **Note**: Hatch uses `uv` as installer (`installer = "uv"` in pyproject.toml) for faster
> environment creation. No manual uv install needed — hatch handles it.

## Key Conventions

1. **Python 3.12+** — `requires-python = ">=3.12"` (LeRobot >=0.5.0 requires 3.12, pinned in `.python-version`)
2. **Dependency bounds** — `>=1.0`: cap major. `<1.0`: cap minor. E.g. `lerobot>=0.5.0,<0.6.0`
3. **`__init__.py` must be thin** — exports only, no logic
4. **Imports at file top** — unless lazy-loading heavy deps with documented reason
5. **Raise on fatal errors** — never warn-and-continue if behavior will be wrong
6. **No silent defaults on error** — returning zero-valued actions on failure is forbidden
7. **Use `require_optional()`** — from `strands_robots/utils.py` for all optional deps
8. **Integration tests required** — each policy needs `tests_integ/` tests with real inference
9. **Test behavior, not implementation** — assert on outputs, not internal state
10. **No dead code** — if it's not called and not part of base class, delete it

## PR Workflow

1. Create feature branch from `main`
2. Make changes, run `uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -v`
3. All tests must pass, lint must be clean
4. Open PR from your fork, address all review comments
5. Track follow-up items as issues on the [project board](https://github.com/orgs/strands-labs/projects/2)
6. Squash merge into `main`


## Registry conventions (strands_robots/registry/robots.json)

- **Flat asset paths** (e.g. `"model_xml": "scene.xml"`) are the common case.
- **Nested asset paths** (e.g. `"model_xml": "xmls/asimov.xml"`) are allowed when
  the upstream source repo uses a subdir layout. Example: `asimov_v0` maps to
  `asimovinc/asimov-v0` which has `sim-model/xmls/asimov.xml` +
  `sim-model/assets/`. The `_safe_join` helper in `strands_robots/utils.py`
  guards against traversal (`..`).
- **Auto-download strategy** — every robot with an `asset` block must declare
  exactly one of:
    1. `asset.robot_descriptions_module` (preferred)
    2. `asset.source` with `type: "github"`
    3. `asset.auto_download: false` (explicit opt-out)
  Enforced by `tests/test_registry_integrity.py`.
