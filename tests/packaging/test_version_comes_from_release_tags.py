"""The package version must be derived from release tags only.

Measured at 0fa5ded90 on a clone of strands-labs/robots with its 167 tags:
``git describe --tags`` resolves to ``artifact-dataset-target-34233699345``
(a CI artifact tag 309 commits back, newer than ``v0.5.1`` 1,141 commits back),
so hatch-vcs built the package as ``34233699346.dev309+g0fa5ded90`` and
``strands-robots doctor`` printed ``PASS strands-robots 34233699346.dev309``.
With ``--match v*`` the same tree is ``0.5.2.dev1141+g0fa5ded90``.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _raw_options() -> dict:
    with (_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["tool"]["hatch"]["version"].get("raw-options", {})


def test_describe_is_restricted_to_release_tags() -> None:
    command = _raw_options().get("git_describe_command")
    assert command, "hatch-vcs must restrict `git describe` to release tags (see module docstring)"
    assert "--match" in command
    assert command[command.index("--match") + 1] == "v*"


def test_the_configured_describe_yields_a_release_version() -> None:
    command = _raw_options()["git_describe_command"]
    try:
        described = subprocess.run(command, cwd=_ROOT, capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pytest.skip("not a git checkout with release tags")
    assert described.startswith("v"), described
    assert described[1].isdigit(), described
