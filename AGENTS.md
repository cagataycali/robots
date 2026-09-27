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
hatch run whole-tree-check  # the graders whose input is the rest of the repo

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
7. **Use `require_optional()`** - from `strands_robots/utils.py` for all optional deps.
   It reports the absent module in `ImportError.name`, so a caller can tell an absent
   extra from a broken package path without parsing the message. A hand-rolled
   `raise ImportError(...)` that reports an absent dependency must leave that module
   recoverable from the exception for the same reason, and
   `tests/test_absent_dependency_reports_name_the_module.py` refuses one that leaves it
   readable only in prose. Three shapes satisfy it and the test grades them one by one:
   `name=`, `raise ... from exc`, or raising lexically inside the `except` handler, which
   sets `__context__` - `from None` suppresses the *rendering* of that chain but not the
   attribute. `name=` is the only shape left once the raise moves out of the handler,
   which is why `require_optionals` passes it explicitly. So this is not "every site
   spells `name=`". Measured over the package with that test's own AST grader: 28
   constructed sites, 7 spelling `name=`, 19 carrying the module on the chain instead
   (18 `from exc`, one raising inside the handler), 2 exempt because they report a
   rejected argument rather than an absent install, 0 blind. Adding the keyword to the
   19 is not a compliance fix. Reach for the grader and not `grep -v name=` when
   auditing this: `name=` on a continuation line of a multi-line `raise` is invisible to
   a line-oriented grep, which reports those 7 as 5 and the 19 as 23
8. **Integration tests required** - each policy needs `tests_integ/` tests with real inference
9. **Test behavior, not implementation** - assert on outputs, not internal state
10. **No dead code** - if it's not called and not part of base class, delete it
11. **A value-domain guard becomes shared when it has a second caller** - the guards
    in `strands_robots/utils.py` (`positive_finite_number_error` and friends) exist so
    the refusal for a rate, a count or a name is identical everywhere rather than
    merely equivalent in verdict, and each one has between 5 and 123 call sites. Do
    not add one for a single field. Keep the rule local to the config that needs it,
    state the domain in that field's docstring, and lift it into `utils.py` when a
    second caller appears - two copies are the evidence that a shared name is the
    right one, and one caller is evidence of nothing. #2008 asked this for a path
    field and the answer was local: the only other `if not self.<path field>` in the
    tree (`training/_inproc.py`) is a branch that skips logging, not a validation.

12. **A security floor on a transitive package is a constraint, not an override** -
    a package that arrives only through another dependency has no version declared
    anywhere, so the resolver's choice is what stands between the dependency graph
    and a HIGH advisory. State the floor in `[tool.uv] constraint-dependencies`,
    at the first version clearing the advisory rather than the version currently
    resolved, and name the GHSA id in a comment beside it. Use a constraint and
    not an override: measured against an extra pinning `gymnasium==0.29.1`,
    `gymnasium>=1.1.1` as a constraint fails `uv lock` and names the contradicting
    pin, while the same floor as an override resolves silently and
    discards that requirement - so an override hides exactly the signal a security
    floor exists to raise. `[project]` is the wrong home while the package stays
    transitive; move the bound there if it ever becomes direct. Pinned by
    tests/test_dependency_audit.py.

13. **Every parameter an agent tool exposes needs its own `Args:` entry** - a
    `@tool` function's input schema is derived from its docstring by
    `docstring_parser`, and the decorator substitutes the placeholder
    `"Parameter <name>"` for any parameter it cannot find there. The model
    driving the tool reads that schema and nothing else, so a placeholder makes
    the parameter undiscoverable however carefully the source explains it.
    Three spellings produce one, and the last two read as documentation in the
    source, which is what makes the loss silent: the entry is absent; the entry
    sits under a section header other than `Args:`, which the parser discards
    entirely (prose reaches the tool description only when it appears *before*
    `Args:`); or one entry names several parameters at once (`a / b: ...`),
    which is read as a single parameter literally named `"a / b"` and therefore
    describes neither. Pinned by
    tests/tools/test_agent_tool_parameter_descriptions.py.
14. **`__repr__` must not raise** - it is what a traceback, a debugger and a
    failing assertion render, so it must not be the thing that hides a failure.
    A class that validates its own arguments raises before it assigns the
    attributes its `__repr__` reads, and the raising frame keeps that half-built
    instance alive: rendering it reports `[AttributeError ... raised in repr()]`
    naming an attribute that has nothing to do with the refusal under
    investigation. Wrap the body in `try` / `except AttributeError` and return
    `strands_robots.utils.partial_construction_repr(self)`, which reports the
    lifecycle fact and deliberately names no attribute so nobody is sent
    chasing one. That helper owns the wording, so the phrase a reader learns to
    recognise cannot diverge between layers. Pinned by
    tests/test_repr_survives_partial_construction.py.

15. **A recording test names its own dataset root** - `DatasetRecorder.create`
    and every backend's `start_recording` resolve a `repo_id` with no `root` to
    `$HF_LEROBOT_HOME/{repo_id}`, i.e. `~/.cache/huggingface/lerobot/{repo_id}`
    by default, and `_prepare_create_target` *inspects* that directory before
    any injected fake dataset class is reached. So a unit test that writes
    nothing to the shared cache still reads it, and its verdict depends on what
    the developer's cache already holds. Measured across 39 such call sites: one
    unrelated dataset planted at `local/probe` turned 133 passed into 22 failed,
    every failure a `FileExistsError` naming a path in `$HOME` rather than the
    test's own resolution - which is what makes it hard to attribute. Pass
    `root=str(tmp_path / "dataset")`, including at the sites refused before the
    root is resolved: requiring it of those too keeps the rule one line with no
    exemptions, where the alternative has to model which guard fires first. Note
    a `repo_id` is as often positional as keyword (`create("user/data", ...)`) and
    the two forms are the same exposure. Rebinding the dataset home suite-wide
    would close the class in one line and would also break the one test that
    legitimately asserts the documented default, which is why the rule lives at
    the call site. `tests_integ/` records real datasets and is out of scope.
    Naming *a* root is not enough: a fixed absolute path such as `root="/tmp/ds"`
    is not the shared cache and is shared all the same, with every other test
    that names it and with every process on the host, so the guard reads the
    value and refuses a `root` given as a string literal - a shape that cannot
    be per-test unique. `root=root` with `root` bound from `tmp_path` is the
    idiomatic form and is accepted.
    Pinned by tests/test_recording_root_is_not_the_shared_cache.py.

16. **An example attests the records it shows, not the whole audit log** -
    `verify_audit_integrity()` with no argument re-reads the entire log, and an
    example's log is the developer's real `~/.strands_robots/mesh_audit.jsonl`,
    because examples deliberately do not redirect `STRANDS_MESH_AUDIT_DIR`. So
    an example that scopes its read to the run (`read_audit_log(since=...)`)
    and then attests everything prints one document describing two record sets.
    Measured on `e4fe2f9` with 4000 records of prior history in the log,
    `examples/fleet/04_emergency_evacuation.py` rendered
    `Audit integrity: ok=False (signed=5/4005)` above a five-row timeline. The
    `ok` value is the worse half: history written before a PSK was configured is
    unsigned, and an unsigned record is a forgery by definition once a PSK is
    set at verification time, so a completely successful run reports tamper
    evidence. Scoped to the records shown the same run reports
    `ok=True (signed=5/5)`. Pass the records (`verify_audit_integrity(records)`),
    or have the report pair them itself so the caller cannot get it wrong.
    `tests/` is exempt mechanically rather than by trust - a test redirects the
    audit dir to `tmp_path`, so there the whole log *is* the record set it
    means. Pinned by tests/test_examples_attest_only_what_they_report.py.

17. **A transport delegates to the raw backend path, never to the router that
    resolves it** - `strands_robots.mesh.session` exposes every Zenoh operation
    twice: a public, backend-aware entry point (`get_session`, `put`,
    `release_session`, `session_alive`) that resolves whatever
    `STRANDS_MESH_BACKEND` selects, and a private `_*_directly` helper that
    always takes the raw Zenoh path. A `MeshTransport` implementation must use
    the second kind for *every* delegation, because under
    `STRANDS_MESH_BACKEND=bridge` the router resolves the `BridgeTransport` that
    owns that very transport - so a backend-aware call routes straight back into
    the caller. Both re-entries are silent, which is what makes the rule worth
    stating rather than leaving to review: a re-entrant `put` raises
    `RecursionError`, which is a `RuntimeError` subclass and so is absorbed by
    the narrow `except (RuntimeError, ConnectionError, OSError)` that idempotent
    transport paths are required to use, and a re-entrant `close` blocks on the
    factory's non-reentrant lock from the thread already holding it. Neither
    reports anything a caller can act on. Every fixture that injects a *fake*
    leg into a composite transport hides this by construction, so the raw path
    has to be pinned structurally. Pinned by
    tests/mesh/test_zenoh_transport_bypasses_backend_routing.py.

18. **A new bridge is un-gated until its surfaces are named, and refused until
    it forwards the context** - `use_ros`'s operator-approval gate is keyed on
    the graph name of the surface a command targets, so it knows nothing about
    a transport added later: a bridge whose topics and services match no
    `_command_gate.COMMAND_BLOCKLIST` entry sends LLM-initiated motion with no
    prompt, no allowlist check and no audit row, and its tests stay green
    because a bridge suite patches the module's `use_ros` symbol at the
    boundary the gate lives behind. The two halves are one change and neither
    ships alone: naming the surfaces without threading `tool_context` through
    every `use_ros` call and declaring the command tools `@tool(context=True)`
    converts the silent bypass into a fail-closed refusal of the whole command
    surface, `stop` included, and threading the context without naming the
    surfaces changes nothing at all. Spell blocklist entries bare - matching is
    on the final path segment, so one `/manual_drive` entry covers every
    namespaced instance - and state the posture where an operator sizes a
    pre-approval (`docs/reference/ros2-integration.md`, `docs/reference/security.md`, the example),
    because `STRANDS_ROS2_COMMAND_ALLOW` is what makes a headless run work and
    it cannot be discovered from a refusal that has not happened yet. Pinned by
    tests/mesh/test_ackermann_command_gate.py, whose inventory of bridges owing
    a gate suite is derived from the tree, so the next transport is graded on
    arrival rather than at the review that happens to look.
19. **A backend may add a parameter but must never permute one it shares** -
    `create_simulation(backend=...)` makes the same `sim` variable a different
    class, and nothing in the type system relates two backends' signatures for a
    method both implement: `add_camera` is on no ABC at all and `randomize` is
    on `SimEngine` only as a `**kwargs` sink whose docstring hands the signature
    to the backend. So a permutation is invisible to every gate and to Python.
    Isaac declared `add_camera(..., width, height, fov, ...)` where MuJoCo and
    Newton declare `(..., fov, width, height, ...)`, so
    `add_camera("wrist", pos, target, 100, 200, 90)` asked for a 200x90 view at
    fov 100 on two backends and a 100x200 view at fov 90 on the third - every
    value valid under either reading, so nothing refused it and the caller found
    out from the pixels. Newton's `randomize` had the same shape, listing the
    three ranges in reverse. Adding a parameter is fine and shifts no shared
    name's *relative* order (Isaac's `mjcf_path`/`usd_path`, Newton's `source`,
    MuJoCo's `randomize_positions`); reordering one is what makes a positional
    call ambiguous. State the shared order where a reader will copy it - the
    signature line in `docs/reference/simulation/newton.md` and the call in
    `docs/reference/simulation/domain-randomization.md` are both graded against the
    signatures - and note that this rule is deliberately weaker than index
    parity, which a mid-signature insertion still breaks. Pinned by
    tests/simulation/test_backend_shared_parameter_order.py, whose backend
    inventory is derived from the table `create_simulation` resolves, so a
    fourth backend is held to the rule the hour it lands.

## PR Workflow

1. Branch on **your fork**. The `default` ruleset applies `creation` to every
   ref with no bypass actors, so pushing a new branch to the base repository is
   refused as a `repository rule violation` for every role.
2. Gate locally: `hatch run format && hatch run lint && hatch run test`. A run
   narrowed to your area needs `hatch run whole-tree-check` beside it - those
   graders read the rest of the repository and no `-k` filter collects them.
3. Add a news fragment `changelog.d/<pr-number>-<slug>.md` (see
   [`changelog.d/README.md`](changelog.d/README.md)); never edit
   `## [Unreleased]` in `CHANGELOG.md`. Enforced by the `Guards` step
   (`scripts/check_changelog_fragment.py`).
4. All tests pass and lint is clean.
5. Open the PR from your fork and address every review comment once per
   concern. A push after approval needs a new approval
   (`require_last_push_approval`), and so does the "Update branch" button.
6. Track follow-ups as issues on the [project board](https://github.com/orgs/strands-labs/projects/2).
7. Squash merge into `main`.
8. Read a PR's state back before and after changing it (`mergeStateStatus`,
   `reviewDecision`, unresolved threads); never infer it from the call you made.

## Registry conventions (strands_robots/registry/robots.json)

- **Flat asset paths** (e.g. `"model_xml": "scene.xml"`) are the common case.
- **Nested asset paths** (e.g. `"model_xml": "xmls/asimov.xml"`) are allowed when
  the upstream source repo uses a subdir layout. Example: `asimov_v0` maps to
  `asimovinc/asimov-v0` which has `sim-model/xmls/asimov.xml` +
  `sim-model/assets/`. The `safe_join` helper in `strands_robots/utils.py`
  guards against traversal (`..`).
- **Auto-download strategy** - every robot with an `asset` block must declare
  exactly one of:
    1. `asset.robot_descriptions_module` (preferred)
    2. `asset.source` with `type: "github"`
    3. `asset.auto_download: false` (explicit opt-out)
  Enforced by `tests/registry/test_integrity.py`.


## History

This file is capped at 30 KB (`tests/test_agents_md_is_within_budget.py`).
Rationale belongs in the PR that made a change; the review learnings and
merge-gate runbook this file carried until 9f011d3e0 are in git history:
`git show 9f011d3e0:AGENTS.md`.
