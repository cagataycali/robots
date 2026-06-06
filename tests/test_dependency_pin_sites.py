"""Repo hygiene: forbid pinning a dependency in two pyproject sites at once.

History: issue #214. Seven deps (Pillow, pytest, pytest-cov, mypy, msgpack,
pyzmq, ruff) were pinned both in the project metadata
(``[project.dependencies]`` / ``[project.optional-dependencies]``) and in
``[tool.hatch.envs.default].dependencies``. That forced every Dependabot bump
to edit both sites in lockstep. The hatch ``default`` env now inherits those
deps via ``features`` (the "all" and "dev" extras), so the duplication was
removed.

This test ratchets the consolidation in: it fails if any project-managed
dependency name is re-introduced into the hatch ``default`` env dependency
list. Deps with no project-level home (declared via the allowlist below) are
permitted to live in the hatch env only.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# Dependency names allowed to be pinned ONLY in the hatch default env because
# they have no [project] home. Keep this narrow.
HATCH_ONLY_ALLOWLIST = {"requests"}


def _name(requirement: str) -> str:
    """Extract the lowercased distribution name from a requirement string."""
    head = requirement.strip()
    for sep in (">", "<", "=", "!", "~", ";", "[", " "):
        head = head.split(sep, 1)[0]
    return head.strip().lower()


def _project_dependency_names(data: dict) -> set[str]:
    names: set[str] = set()
    for req in data["project"].get("dependencies", []):
        names.add(_name(req))
    for reqs in data["project"].get("optional-dependencies", {}).values():
        for req in reqs:
            names.add(_name(req))
    return names


def _hatch_default_dependency_names(data: dict) -> set[str]:
    env = data["tool"]["hatch"]["envs"]["default"]
    return {_name(req) for req in env.get("dependencies", [])}


def test_hatch_default_env_does_not_duplicate_project_pins() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project_names = _project_dependency_names(data)
    hatch_names = _hatch_default_dependency_names(data)

    duplicated = sorted(hatch_names & project_names)
    assert not duplicated, (
        "These deps are pinned in both [project] metadata and "
        "[tool.hatch.envs.default].dependencies; inherit them via env "
        f"features instead: {duplicated}"
    )


def test_hatch_only_dependencies_are_allowlisted() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project_names = _project_dependency_names(data)
    hatch_names = _hatch_default_dependency_names(data)

    hatch_only = hatch_names - project_names
    unexpected = sorted(hatch_only - HATCH_ONLY_ALLOWLIST)
    assert not unexpected, (
        "Hatch-only deps must be added to HATCH_ONLY_ALLOWLIST with a rationale "
        f"or moved into a [project] extra: {unexpected}"
    )
