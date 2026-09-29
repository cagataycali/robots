---
description: Set up a development install, run the checks a pull request must pass, and open the PR in the shape the check accepts.
---

# Contributing

The short version of `AGENTS.md`, the file that governs this repository: a development install, the checks a pull request must pass, and the PR shape the required check accepts.

## Set up

```bash
git clone https://github.com/<you>/robots && cd robots
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[all,dev]"
strands-robots doctor
```

Python 3.12 or newer. `hatch` drives the scripts below with `uv` as its installer; nothing else needs installing.

## Check before you push

```bash
hatch run format            # ruff check --fix, ruff format
hatch run lint              # ruff check, ruff format --check, mypy
hatch run test              # unit tests, one worker per file
hatch run whole-tree-check  # the graders whose input is the rest of the repo
hatch run test-integ        # integration tests: GPU, model weights, hardware
```

Ruff runs at line length 120 targeting `py312`; mypy runs with `disallow_untyped_defs`. A narrow test run (`pytest tests/drivers -k g1`) is fine for iteration; pair it with `whole-tree-check`, since many graders read the whole repository and no path filter collects them.

## Conventions the graders enforce

| rule | why |
|---|---|
| `__init__.py` is exports only; heavy imports are lazy with a documented reason | `import strands_robots` must leave numpy, torch and mujoco unloaded |
| optional dependencies go through `require_optional()` in `utils.py` | the refusal names the extra that fixes it |
| raise on a fatal error, never warn and continue; never return a zero action on failure | an agent reads a silent default as success |
| every parameter an agent tool exposes has its own `Args:` entry | the docstring is the schema the model sees |
| dependency bounds: `>=1.0` packages cap the major, `<1.0` packages cap the minor | a floor without a ceiling is not a bound |
| a security floor on a transitive package is a `[tool.uv] constraint-dependencies` entry, not an override | the lockfile parity check reads it there |
| a new policy provider ships an integration test with real inference | mocks cannot grade a checkpoint |
| no dead code; test behaviour, not implementation | the tree is graded for both |

## Log the change

Every pull request that changes behaviour adds one file under `changelog.d/`, `<pr-number>-<slug>.md`, with the `### <Category>: <summary>` heading and body that would have gone into `CHANGELOG.md`. Never edit `## [Unreleased]` directly; the `Guards` step (`scripts/ci_guards.py`) refuses a PR that does, and refuses the placeholders `0000` and `999x`. Push the fragment right after opening the PR, or open as a draft and add it before marking ready: a push after an approval dismisses the approval.

## Open the pull request

1. Branch on your fork: a ruleset refuses branch creation in `strands-labs/robots` for every account, with a rule violation that does not name the rule.
2. Check that no open PR claims the issue or edits the file: `python3 .github/scripts/check_duplicate_claim.py --repo strands-labs/robots --issue <N>` and `python3 scripts/check_merge_base_overlap.py --github-repo strands-labs/robots --paths <files>`.
3. Put `Closes #N` in the PR body, not the title.
4. The required check evaluates the merge commit: ruff, mypy, the unit suite, the whole-tree graders, the guards, lockfile parity and CodeQL (`ci.yml`).

Security findings do not go through issues: [security policy](security-policy.md).
