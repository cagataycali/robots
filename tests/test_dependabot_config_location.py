"""Contract pins for the Dependabot configuration's location.

``dependabot.yml`` is a *configuration* file, not a workflow, and the two live in
different directories. Dependabot reads version-update config from exactly one
path -- ``.github/dependabot.yml`` -- while anything under
``.github/workflows/`` is handed to Actions and parsed as a workflow. This
repository had the file in the second place, which fails in both directions at
once and is silent in the direction that matters:

- **Actions** rejected it on every push to ``main``. It declares ``version:`` and
  ``updates:`` and therefore has neither ``on:`` nor ``jobs:``, so the run failed
  at parse time with ``total_jobs: 0`` and ``created_at == updated_at``. Measured
  on 15 consecutive pushes before the move, i.e. a red run on 100% of pushes,
  which is the kind of permanent red that teaches everyone to stop reading CI.
- **Dependabot** never saw it. ``.github/dependabot.yml`` returned 404, so the
  grouping (``ml-stack``, ``sim-stack``, ``dev-tools``) and both ecosystem entries
  were inert. Nothing reported this, because an absent config is indistinguishable
  from a repository that has not configured Dependabot.

The second half is why this is pinned rather than just fixed. ``AGENTS.md`` >
"Action Pinning" delegates a supply-chain control to the ``github-actions``
ecosystem entry:

    **Dependabot keeps these fresh** via the ``github-actions`` ecosystem entry.
    Do not manually bump tags; merge the Dependabot PR.
    Especially ``pypa/gh-action-pypi-publish`` [...] This pin is non-negotiable.

A documented non-negotiable control that no longer runs is worse than one that was
never claimed, so the entry that implements it is asserted here by name.

``test_every_workflow_file_declares_the_two_keys_actions_requires`` is the guard
for the *class* rather than for this file: it is what would have caught the
original placement, and it catches the next config file put in ``workflows/`` as
well. It was non-vacuous at the time of the fix -- 12 of the 13 files there
declared both keys and ``dependabot.yml`` was the only one declaring neither.

These are text assertions rather than parsed YAML because that is the shape the
existing CI-config pins use (``tests/test_codeql_query_filters.py`` and
``tests/test_merge_base_overlap.py`` both read their file this way) and because
``pyyaml`` is an optional dependency here -- a pin that skips when a dep is
missing is not a pin.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPO_ROOT / ".github" / "dependabot.yml"
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"
_AGENTS_PATH = _REPO_ROOT / "AGENTS.md"

#: Matches a top-level ``on:`` key, including the quoted spellings. YAML 1.1
#: reads a bare ``on`` as the boolean ``true``, so ``"on":`` and ``'on':`` are
#: both legitimate and must not be read as a missing trigger.
_ON_KEY = re.compile(r"""^(?:on|"on"|'on'):""", re.MULTILINE)
_JOBS_KEY = re.compile(r"^jobs:", re.MULTILINE)

#: The ecosystems this repository depends on Dependabot covering. ``pip`` keeps
#: the ML/sim dependency tree current; ``github-actions`` is the one AGENTS.md
#: names as non-negotiable, because every ``uses:`` here is pinned to a 40-char
#: SHA and Dependabot is what advances those pins.
_REQUIRED_ECOSYSTEMS = ("pip", "github-actions")

#: Lower bound for the workflow scan, so the class guard cannot pass by finding
#: nothing. The directory held 13 files when this landed.
_MIN_WORKFLOWS = 10


def _config_text() -> str:
    return _CONFIG_PATH.read_text(encoding="utf-8")


def _workflow_files() -> list[Path]:
    return sorted(p for p in _WORKFLOWS_DIR.iterdir() if p.is_file() and p.suffix in {".yml", ".yaml"})


class TestTheConfigIsWhereDependabotReadsIt:
    """Dependabot reads one path. The config has to be on it."""

    def test_the_config_exists_at_the_only_path_dependabot_reads(self) -> None:
        assert _CONFIG_PATH.is_file(), (
            f"{_CONFIG_PATH.relative_to(_REPO_ROOT)} is missing. Dependabot reads "
            "version-update config from this path only; anywhere else it is inert "
            "and nothing reports that."
        )

    def test_the_config_is_a_dependabot_config_and_not_a_workflow(self) -> None:
        text = _config_text()
        assert re.search(r"^version:\s*2\s*$", text, re.MULTILINE), "Dependabot config must declare 'version: 2'."
        assert re.search(r"^updates:", text, re.MULTILINE), "Dependabot config must declare an 'updates:' list."

    @pytest.mark.parametrize("ecosystem", _REQUIRED_ECOSYSTEMS)
    def test_the_ecosystem_entries_the_repo_depends_on_are_present(self, ecosystem: str) -> None:
        pattern = rf"^\s*-\s*package-ecosystem:\s*[\"']?{re.escape(ecosystem)}[\"']?\s*$"
        assert re.search(pattern, _config_text(), re.MULTILINE), (
            f"No 'package-ecosystem: {ecosystem}' entry. AGENTS.md > 'Action "
            "Pinning' delegates SHA-pin freshness to the github-actions entry and "
            "calls that pin non-negotiable, so removing an entry silently retires a "
            "documented control."
        )

    def test_agents_md_still_delegates_pin_freshness_to_dependabot(self) -> None:
        """The claim has two homes; the pin covers both.

        If the delegation in AGENTS.md is ever dropped, the assertion above stops
        describing a real obligation and becomes a rule with no stated reason.
        """
        text = _AGENTS_PATH.read_text(encoding="utf-8")
        assert "github-actions` ecosystem entry" in text, (
            "AGENTS.md no longer delegates action-pin freshness to the "
            "github-actions ecosystem entry. Reconcile this module with it."
        )


class TestNoConfigFileSitsInTheWorkflowsDirectory:
    """Everything in ``workflows/`` is parsed as a workflow, so nothing else fits."""

    def test_the_dependabot_config_is_not_in_the_workflows_directory(self) -> None:
        stray = _WORKFLOWS_DIR / "dependabot.yml"
        assert not stray.exists(), (
            "dependabot.yml is back in .github/workflows/. Actions parses every "
            "file there as a workflow and this one has no 'on:' or 'jobs:', so it "
            "fails at parse time on every push while Dependabot still reads nothing."
        )

    def test_every_workflow_file_declares_the_two_keys_actions_requires(self) -> None:
        workflows = _workflow_files()
        assert len(workflows) >= _MIN_WORKFLOWS, (
            f"Only {len(workflows)} workflow files found; expected at least "
            f"{_MIN_WORKFLOWS}. A scan that finds nothing asserts nothing."
        )

        malformed = [
            p.name
            for p in workflows
            if not (_ON_KEY.search(t := p.read_text(encoding="utf-8")) and _JOBS_KEY.search(t))
        ]
        assert not malformed, (
            "These files under .github/workflows/ do not declare both 'on:' and "
            f"'jobs:' and will fail at parse time on every push: {malformed}. A "
            "configuration file belongs outside the workflows directory."
        )
