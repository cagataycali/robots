"""A job calling a reusable workflow grants every scope that workflow asks for.

A called workflow's job may hold no permission its caller did not grant: the
caller's ``permissions`` block is a ceiling, not a starting point.  When a called
job asks for a scope above that ceiling GitHub refuses the *entire run* before
any job exists, so there is no job log to read, no check row to click, and no
annotation on the workflow that caused it -- only a run whose conclusion is
``startup_failure``.

That is how publishing ``v0.5.2`` stopped.  The pull-request guards moved into
``test-lint.yml`` and its job asked for ``pull-requests: read`` to read
``closingIssuesReferences`` (#3822), while ``pypi-publish-on-release.yml`` still
granted the ceiling it had needed for v0.5.0 and v0.5.1, ``contents: read``.  The
release was published, the tag existed, no job ran, and PyPI kept serving the
previous version.  ``actionlint`` reads each workflow alone and does not model
the relationship between the two, so the ceiling is graded here.

Parsing is line-based rather than via ``yaml``: ``tests/`` is type-checked under
``ignore_missing_imports = false`` and ``types-PyYAML`` is not a dev dependency,
so importing it would either fail ``mypy`` or require a dependency change and a
``uv.lock`` relock.  The sibling workflow contracts
(``tests/test_workflow_jobs_are_bounded.py``,
``tests/test_codeql_query_filters.py``) read their YAML the same way.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"

#: Ordered weakest to strongest, so "does the grant cover the request" is one
#: comparison. A scope the caller does not name is ``none`` for the called job.
_LEVELS = {"none": 0, "read": 1, "write": 2}

_JOBS_KEY = re.compile(r"^jobs:\s*$")
_TOP_LEVEL_KEY = re.compile(r"^\S")
_JOB_HEADER = re.compile(r"^ {2}([A-Za-z0-9_-]+):\s*$")
_JOB_KEY = re.compile(r"^ {4}([A-Za-z0-9_-]+):(.*)$")
_SCOPE = re.compile(r"^ {6}([a-z-]+):\s*(\S+)\s*$")
_REUSABLE_CALL = "./.github/workflows/"


def _lines(path: Path) -> list[str]:
    """The workflow's lines with comment-only lines dropped."""
    return [line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")]


def _job_blocks(path: Path) -> dict[str, list[str]]:
    """Map each ``jobs.<job_id>`` to the lines of its block."""
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    in_jobs = False
    for line in _lines(path):
        if _JOBS_KEY.match(line):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        if _TOP_LEVEL_KEY.match(line):
            break
        if header := _JOB_HEADER.match(line):
            current = header.group(1)
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)
    return blocks


def _permissions(block: list[str]) -> dict[str, str]:
    """The ``permissions:`` mapping declared in one job block.

    Args:
        block: The lines of a ``jobs.<job_id>`` block, as ``_job_blocks`` yields.

    Returns:
        Scope to level for an explicit block. Empty when the job declares no
        ``permissions`` key or writes it as a shorthand string (``read-all``,
        ``write-all``), neither of which names a scope this can compare.
    """
    scopes: dict[str, str] = {}
    inside = False
    for line in block:
        if key := _JOB_KEY.match(line):
            inside = key.group(1) == "permissions" and not key.group(2).strip()
            continue
        if inside and (scope := _SCOPE.match(line)):
            scopes[scope.group(1)] = scope.group(2)
    return scopes


class Call:
    """One job that calls another workflow of this repository."""

    def __init__(self, caller: Path, job_id: str, called: Path) -> None:
        self.caller = caller
        self.job_id = job_id
        self.called = called

    @property
    def ref(self) -> str:
        return f"{self.caller.name}:{self.job_id}"

    def __repr__(self) -> str:
        return f"{self.ref}->{self.called.name}"


def _reusable_calls() -> list[Call]:
    """Every job in ``.github/workflows`` that calls a workflow from this repository."""
    calls: list[Call] = []
    for path in sorted(_WORKFLOWS.glob("*.yml")):
        for job_id, block in _job_blocks(path).items():
            for line in block:
                key = _JOB_KEY.match(line)
                if key and key.group(1) == "uses" and key.group(2).strip().startswith(_REUSABLE_CALL):
                    called = _REPO_ROOT / key.group(2).strip()[len("./") :]
                    calls.append(Call(path, job_id, called))
    return calls


_CALLS = _reusable_calls()


def test_the_repository_calls_its_own_reusable_workflows() -> None:
    """The contract below is only meaningful while such a call exists."""
    assert _CALLS, f"no job in {_WORKFLOWS} calls a local reusable workflow"


@pytest.mark.parametrize("call", _CALLS, ids=repr)
def test_a_caller_grants_the_scopes_the_workflow_it_calls_asks_for(call: Call) -> None:
    granted = _permissions(_job_blocks(call.caller).get(call.job_id, []))
    assert granted, f"{call.ref} calls {call.called.name} without declaring a permissions ceiling"
    assert call.called.is_file(), f"{call.ref} calls {call.called}, which does not exist"

    short = [
        f"{call.called.name}:{job_id} asks {scope}: {level}"
        for job_id, block in _job_blocks(call.called).items()
        for scope, level in _permissions(block).items()
        if _LEVELS.get(level, 0) > _LEVELS.get(granted.get(scope, "none"), 0)
    ]
    assert not short, (
        f"{call.ref} grants {granted}, a ceiling the workflow it calls exceeds: "
        f"{'; '.join(short)}. A called job may hold no permission its caller withheld, "
        f"and GitHub refuses the whole run at startup rather than failing that job, so no "
        f"log says this. Name the scope in the permissions block of `{call.job_id}`."
    )
