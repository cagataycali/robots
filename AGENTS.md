# AGENTS.md - strands-labs/robots

## Overview

`strands-robots` is a robot control library for [Strands Agents](https://strandsagents.com). It provides policy inference, teleoperation, calibration, and simulation tools for physical robots.

## Project Dashboard

**Board**: https://github.com/orgs/strands-labs/projects/2
**Project ID**: `PVT_kwDOD151Fs4BSRJP`

> **RULE**: ALWAYS use the project board to track work. When creating follow-up items,
> create GitHub issues and add them to this board with Status + Priority set.
> Never track work only in local markdown - the board is the source of truth.

## Repository Structure

```
strands_robots/
├── policies/              # Policy providers (pluggable via registry)
│   ├── base.py            # Abstract Policy base class
│   ├── factory.py         # create_policy() factory + registry
│   ├── mock.py            # MockPolicy for testing
│   ├── groot/             # NVIDIA GR00T N1.5/N1.6/N1.7 inference
│   │   ├── policy.py      # Gr00tPolicy (ZMQ + HTTP modes)
│   │   ├── client.py      # Gr00tInferenceClient
│   │   ├── data_config.py # Gr00tDataConfig + ModalityConfig
│   │   └── data_configs.json  # 25 robot embodiment configs
│   └── lerobot_local/     # HuggingFace LeRobot direct inference
│       ├── policy.py      # LerobotLocalPolicy (RTC support)
│       ├── processor.py   # ProcessorBridge (pre/post pipelines)
│       └── resolution.py  # Policy class resolution (v0.4/v0.5)
├── registry/              # JSON registry for policy discovery
├── tools/                 # Strands @tool functions
│   ├── gr00t_inference.py # GR00T inference tool
│   ├── lerobot_calibrate.py
│   ├── lerobot_camera.py
│   ├── lerobot_teleoperate.py
│   ├── pose_tool.py
│   └── serial_tool.py
├── robot.py               # Core Robot class
└── utils.py               # Shared utilities (require_optional, etc.)

tests/                     # Unit tests (run with: hatch run test)
tests_integ/               # Integration tests (run with: hatch run test-integ)
```

## Development

```bash
# Install with all optional deps
pip install -e ".[all,dev]"

# Run tests
hatch run test              # unit tests
hatch run test-integ        # integration tests (needs GPU + model weights)

# Lint & format
hatch run lint              # ruff check, ruff format --check, mypy
hatch run format            # ruff check --fix, ruff format
```

> **Note**: Hatch uses `uv` as installer (`installer = "uv"` in pyproject.toml) for faster
> environment creation. No manual uv install needed - hatch handles it.

## Key Conventions

1. **Python 3.12+** - `requires-python = ">=3.12"` (LeRobot >=0.5.0 requires 3.12)
2. **Dependency bounds** - `>=1.0` deps: cap major. `<1.0` deps: cap minor. E.g. `lerobot>=0.5.0,<0.6.0`
3. **`__init__.py` must be thin** - exports only, no logic
4. **Imports at file top** - unless lazy-loading heavy deps with documented reason
5. **Raise on fatal errors** - never warn-and-continue if the system will behave unexpectedly
6. **No silent defaults on error** - returning zero-valued actions on failure is forbidden
7. **Use `require_optional()`** - from `strands_robots/utils.py` for all optional deps
8. **Integration tests required** - each policy needs `tests_integ/` tests with real inference
9. **Test behavior, not implementation** - assert on outputs, not internal state
10. **No dead code** - if it's not called and not part of base class, delete it

## PR Workflow

1. Create feature branch from `main`
2. Make changes, run `hatch run format && hatch run lint && hatch run test`
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
- **Auto-download strategy** - every robot with an `asset` block must declare
  exactly one of:
    1. `asset.robot_descriptions_module` (preferred)
    2. `asset.source` with `type: "github"`
    3. `asset.auto_download: false` (explicit opt-out)
  Enforced by `tests/test_registry_integrity.py`.


## Review Learnings (PR #85 - MuJoCo Backend)

Corrections from code review that apply to all future contributions:

### Thread Safety
- **Lock ALL model/data mutations** - MuJoCo `model`/`data` are not thread-safe. Any method that writes `qpos`, `qvel`, `ctrl`, `qfrc_applied`, `body_mass`, `geom_friction`, or calls `mj_step`/`mj_forward`/`mj_resetData` MUST hold `self._lock`.
- **Guard scene mutations during policy** - Use `_require_no_running_policy()` before any action that recompiles or replaces the model/data objects.
- **Document the concurrency contract** - If a method is safe to call concurrently, say so. If not, say so.

### Error Handling Contracts
- **Return error dicts, never raise** - All `AgentTool` action handlers must return `{"status": "error", "content": [...]}` on failure. Never raise exceptions that bypass the structured response.
- **Clean up on failure** - If you register state (e.g., add to `self._world.objects`) before an operation that can fail, pop/undo it in the except path.
- **Fail-fast with `strict=True`** - Silent frame dropping or catch-all `except Exception` with logging is forbidden unless gated behind a `strict=False` parameter.

### API Consistency
- **Don't export private functions** - `_`-prefixed names must never appear in `__all__`.
- **Match docstrings to semantics** - If the docstring says "single-shot" but the code is "latched", one of them must change. Always verify by reading the underlying library docs.
- **Forward all advertised kwargs** - If `tool_spec.json` exposes a parameter, the dispatch chain must forward it all the way through. Silent drops are bugs.
- **Centralize import checks at init** - Prefer checking optional deps once in `__init__` over scattered `_ensure_X()` guards. Consumers catch issues at init time.

### Data Integrity
- **Per-name state copy, not flat index** - When recompiling MuJoCo models (inject/eject), copy qpos/qvel per-joint by name. Flat-index slicing breaks when body-tree order shifts.
- **Sanitize user inputs into XML** - Validate names against `^[a-zA-Z0-9_-]+$` before interpolating into MJCF. LLM-provided strings are untrusted.
- **Match schema and data keys** - If a feature is declared with sanitized names (e.g., `__`), the data producer must emit the same sanitized keys.

### Testing
- **Test import paths must match production** - If `src/` imports `from lerobot.datasets.X`, tests must use the same path. Mismatched paths cause silent skips via `except ImportError`.
- **Round-trip tests for recording** - Any recording feature needs: start -> write -> stop -> reopen -> assert non-empty. Schema-only tests miss silent data loss.
- **Pin regression tests for reviewed fixes** - Every review fix gets a test that fails on pre-fix code. Otherwise the next refactor silently reintroduces the bug.
- **No host paths in test files** - Never commit `/Users/<name>/` or `/home/<name>/` paths. CI test `test_no_host_paths.py` enforces this.

### Performance
- **Don't create executors in hot loops** - Reuse a single `ThreadPoolExecutor` instance instead of creating one per call at 50Hz.
- **Cache expensive JSON parsing** - If a `@property` re-parses a JSON file on every access, cache the result at module load or first access.
