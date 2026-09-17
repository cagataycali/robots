"""A workflow that calls another grants every scope the callee's jobs declare.

A reusable workflow's job-level ``permissions:`` block is a request, not a
grant: the caller's block is a ceiling, and GitHub refuses to start a run whose
callee asks for a scope the caller did not admit. The refusal arrives as a
``startup_failure`` with zero jobs, so there is no step, no log and no
annotation to read -- the workflow simply did not happen.

That is how `v0.5.2` reached GitHub and never reached PyPI. #3822 gave
test-lint.yml's job ``pull-requests: read`` for its guards step and granted it
from ci.yml, the caller that runs on every pull request. The same workflow is
also called by pypi-publish-on-release.yml, which still granted ``contents:
read`` alone; that path only runs on release day, so the mismatch was invisible
for the sixteen days between the two events.

The pin is therefore over every caller the tree contains rather than over the
callers a reader thought to name: a scope added to a called job is a change to
each of its call sites, and the set of call sites is discovered here so that a
new one is graded the moment it is committed.

Read as text rather than parsed. ``pyyaml`` is an optional dependency of this
package, so a pin that imports it is a pin that silently skips -- which is the
same failure this module exists to prevent.
"""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOW_DIR = _ROOT / ".github" / "workflows"

# GitHub's own ordering: a caller admits a scope when it grants at least the
# level the callee asks for. ``none`` is what an unnamed scope resolves to.
_LEVELS = {"none": 0, "read": 1, "write": 2}


def _jobs(text: str) -> dict[str, str]:
    """Map job id to job body for a workflow's ``jobs:`` mapping.

    Args:
        text: The full text of a workflow file.

    Returns:
        One entry per job, keyed by job id. Empty when the file declares no
        ``jobs:`` mapping.
    """
    marker = "\njobs:\n"
    if marker not in text:
        return {}
    jobs: dict[str, str] = {}
    name: str | None = None
    body: list[str] = []
    for line in text[text.index(marker) + len(marker) :].splitlines(keepends=True):
        start = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if start:
            if name is not None:
                jobs[name] = "".join(body)
            name, body = start.group(1), []
        elif line.strip() and not line.startswith(" "):
            break  # a new top-level key ends the mapping
        elif name is not None:
            body.append(line)
    if name is not None:
        jobs[name] = "".join(body)
    return jobs


def _permissions(job: str) -> dict[str, str]:
    """Map scope to level for a job's ``permissions:`` block.

    Args:
        job: One job body as returned by :func:`_jobs`.

    Returns:
        The scopes the block names. Empty when the job declares no block, which
        is the inherit-the-default case and not a request for anything.
    """
    header = re.search(r"^    permissions:\s*$", job, re.MULTILINE)
    if header is None:
        return {}
    scopes: dict[str, str] = {}
    for line in job[header.end() :].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        entry = re.match(r"^      ([a-z-]+):\s*(read|write|none)\s*$", line)
        if entry is None:
            break  # the next key at the job's own level ends the block
        scopes[entry.group(1)] = entry.group(2)
    return scopes


def _calls() -> list[tuple[str, str, Path, dict[str, str]]]:
    """Every in-repository reusable-workflow call the tree contains.

    Returns:
        Tuples of caller filename, calling job id, the called workflow's path
        and the scopes that call site grants.
    """
    calls = []
    for path in sorted(_WORKFLOW_DIR.glob("*.yml")):
        for job_id, body in _jobs(path.read_text(encoding="utf-8")).items():
            call = re.search(r"^    uses:\s*\./(\.github/workflows/[\w.-]+)\s*$", body, re.MULTILINE)
            if call is not None:
                calls.append((path.name, job_id, _ROOT / call.group(1), _permissions(body)))
    return calls


_CALLS = _calls()


def test_the_tree_still_calls_a_workflow_from_another() -> None:
    """The discovery is the pin, so an empty result is a broken pin, not a pass.

    A call site that reads as no scopes, or a callee that reads as asking for
    none, makes the comparison below pass without comparing anything -- the
    parser going blind and the tree being correct look identical from the
    outside. Both ends are required to be non-empty here so that they cannot.
    """
    assert _CALLS, f"no `uses: ./.github/workflows/...` call found under {_WORKFLOW_DIR}"
    asked = {
        scope
        for _, _, callee, _ in _CALLS
        for body in _jobs(callee.read_text(encoding="utf-8")).values()
        for scope in _permissions(body)
    }
    assert asked, "no called job declares a scope; the grant comparison would be vacuous"
    assert any(granted for *_, granted in _CALLS), "no call site grants a scope"


_SYNTHETIC = """name: x

on:
  push:
    branches: [main]

jobs:
  caller:
    uses: ./.github/workflows/callee.yml
    permissions:
      contents: read
      # a comment between entries does not end the block
      pull-requests: read
    with:
      ref: main

  after:
    runs-on: ubuntu-latest

concurrency:
  group: x
"""


def test_a_permissions_block_is_read_whole_and_no_further() -> None:
    """Comments sit between entries, and ``with:`` ends the block rather than a scope.

    The keys under ``on:`` share the indentation of a job id, and a top-level
    key ends the mapping, so a parser that reads either as a job reports scopes
    for something that cannot hold them.
    """
    jobs = _jobs(_SYNTHETIC)
    assert sorted(jobs) == ["after", "caller"]
    assert _permissions(jobs["caller"]) == {"contents": "read", "pull-requests": "read"}
    assert _permissions(jobs["after"]) == {}


@pytest.mark.parametrize(
    ("caller", "job_id", "callee", "granted"),
    _CALLS,
    ids=[f"{caller}:{job_id}" for caller, job_id, _, _ in _CALLS],
)
def test_a_call_site_grants_every_scope_the_called_jobs_declare(
    caller: str, job_id: str, callee: Path, granted: dict[str, str]
) -> None:
    """A callee asking for more than its caller admits is a startup failure.

    Args:
        caller: Filename of the calling workflow.
        job_id: The calling job's id.
        callee: Path to the called workflow.
        granted: The scopes the call site grants.
    """
    assert callee.is_file(), f"{caller} job '{job_id}' calls {callee}, which does not exist"
    for callee_job, body in _jobs(callee.read_text(encoding="utf-8")).items():
        for scope, asked in _permissions(body).items():
            have = granted.get(scope, "none")
            assert _LEVELS[have] >= _LEVELS[asked], (
                f"{caller} job '{job_id}' grants {scope}: {have}, but {callee.name} job "
                f"'{callee_job}' declares {scope}: {asked}. A caller's block is a ceiling, "
                "so GitHub refuses to start the run: startup_failure with no job to read."
            )
