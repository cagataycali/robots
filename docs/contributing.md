---
description: How to land a PR — hatch envs, ruff/mypy, lazy-import discipline, JSON registries, test layout.
---

# Contributing

The repo lives at [`strands-labs/robots`](https://github.com/strands-labs/robots).
Bug reports go to the issue tracker; code contributions go through pull requests.

## Setup

```bash
git clone https://github.com/strands-labs/robots
cd robots
pip install -e ".[all,dev]"
```

The `[dev]` extra installs `hatch`, `pytest`, `ruff`, `mypy`. Hatch is the build /
env runner — we use it for lint, test, and release.

## Common commands

```bash
# Run the full test suite
hatch run test

# Just the unit tests (fast)
hatch run test --no-cov tests/

# Lint
hatch run lint                  # ruff + mypy
hatch run format                # ruff fix + format

# Build the docs locally
mkdocs serve                    # http://localhost:8000
mkdocs build --strict           # CI gate
```

`hatch run test -x --strict-markers` is what CI runs.

## Module conventions

The architecture page covers the high-level rules; the contributing-specific bits:

### Lazy imports are mandatory

Heavy modules (`mujoco`, `lerobot`, `torch`, `zenoh`) must NOT load at top-level. Use
PEP 562 `__getattr__` in `__init__.py` to defer them. CI fails (`tests/test_init.py`)
if you add an eager heavy import.

### Tests mirror source

`tests/policies/test_groot.py` mirrors `strands_robots/policies/groot/`. Keep this
1:1 — it makes "where do I add a test" obvious.

### No host paths in code or tests

`/Users/cagatay/...` is not allowed anywhere. Use `tmp_path` fixtures, `~/.cache`,
or environment variables. CI grep-blocks host paths.

### JSON registries are the source of truth

Robots and policies live in JSON, not Python. Adding a new robot or policy is mostly
a JSON edit + tests. Don't add hardcoded lookups in `.py` files.

### Tool errors return, don't raise

Strands `@tool`-decorated functions and `AgentTool` actions never raise out of their
dispatch path. They return:

```python
{"status": "error", "content": [{"text": "human-readable error"}]}
```

This keeps the agent's loop deterministic.

## PR workflow

1. **Branch from main** (or a feature branch if the change is large and needs
   coordination).
2. **Write tests first.** A failing test, then the fix. The CI runs tests on every
   push.
3. **Keep PRs small.** ~300 lines of diff is the sweet spot. Stack PRs if the work
   is bigger — see the "PR chain" patterns on past releases (e.g. PR #82-#87).
4. **Update docs.** Any user-visible change needs a docs update. The docs site is
   built with `mkdocs build --strict` in CI.
5. **Pass lint and test.** `hatch run lint && hatch run test`.
6. **Open the PR.** Use the bug / feature templates if applicable.
7. **Address review.** We try to get a first response in <48h.
8. **Squash on merge.** One commit per merged PR.

## Release process

Releases are cut from `main` by maintainers using `hatch version` + a GitHub release.
Versioning follows semver: minor for additive features (new robots, new policies),
patch for fixes, major for breaking changes (rare).

## Code of conduct

Be excellent. Disagreements are fine; rudeness is not. The maintainers will moderate
public threads as needed.

## Where to ask

| Topic | Where |
|-------|-------|
| Bug report | [Issues](https://github.com/strands-labs/robots/issues) |
| Feature request | [Issues](https://github.com/strands-labs/robots/issues) (use the feature template) |
| How do I do X | [Discussions](https://github.com/strands-labs/robots/discussions) |
| Security | private email — see SECURITY.md |

## See also

- [Architecture](architecture.md) — module conventions in depth.
- [Tutorial 9 — Advanced](tutorial/09-advanced.md) — extension points.
- [API reference](api-reference.md) — public symbols you should leave alone unless
  the PR is explicitly about them.
