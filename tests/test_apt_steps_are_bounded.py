"""An ``apt-get`` step in CI cannot stall past a bound that names it.

``call-test-lint / Test and Lint`` reported ``FAILURE`` having never run a test,
twice inside one 80-minute window.  Both jobs were reaped by the job's own
``timeout-minutes: 45`` while still inside step 4, ``Install system
dependencies (OpenGL for MuJoCo)`` -- measured from the attempt-1 job records::

    run           subject                  step 4      duration   lint / tests
    32163486141   main @ 74d078ab (push)   cancelled   45m12s     skipped
    32165565982   #2440 @ 2c489f92 (PR)    cancelled   45m11s     skipped

Steps 1-3 succeeded in both, steps 5-9 are ``skipped`` in both, and both jobs
ran 45m23s -- the job ceiling, not a decision either step made.  Two runners,
overlapping windows, the same wedge point.  See #2442.

Two distinct bounds are missing, and each covers a case the other cannot.

**A retry loop driven by an exit status cannot survive a hang.**  The step
already anticipates a bad mirror and retries, but the loop reads *what apt-get
returned*::

    for attempt in 1 2 3; do
      if sudo apt-get update; then break; fi
      echo "apt-get update failed (attempt ${attempt}/3), retrying in 5s..."
      sleep 5
    done

A mirror that answers wrongly exits 100 and is retried.  A mirror that never
answers leaves the ``if`` unevaluated: attempt 2 is never reached, the backoff
never runs, and the ``echo`` that would have named apt never prints -- so the
log simply stops mid-step and the failure is silent as well as slow.  Wrapping
the command in ``timeout`` converts the hang into exit 124, which is the
failure the loop already handles.  A step-level bound cannot substitute here:
it kills the step rather than the attempt, so the loop still never retries.

**A step-level bound is what makes the reap legible.**  A job-level reap
cancels whatever step is running and aggregates to ``FAILURE`` carrying no
reason -- the same false red #1800/#2304 documented for concurrency cancels,
from a third producer that their ``github.sha`` keying does not touch.  A
step-level ``timeout-minutes`` fires first and marks the step, so the verdict
reads "the mirror is wedged" instead of "something in your diff".

**The bounds are sized from measured runs, and the ceiling is not the tight
end.**  #2442 suggested ``timeout 180`` for ``update`` and ``timeout 300`` for
``install``.  Over 31 successful ``Test and Lint`` jobs the apt step ran p50
33s, p90 264s, max 455s, with 7 runs above 180s and 5 of those *outside* the
incident window -- so a 300s bound on ``install`` would have reaped healthy
runs.  Reading the slowest one's log (job 95914546572, step 455s) splits the
cost: ``apt-get update`` finished in ~5s and the remaining ~449s was
``apt-get install`` fetching ffmpeg's ~115 packages.  The same tail appears in
``agent-api-check.yml``, whose apt step installs no ffmpeg and still ran 474s
once over 30 runs against a p50 of 15s.

So the two commands want different treatment, which is why this contract does
not simply demand ``timeout`` everywhere: ``update`` is bounded per attempt
because that is the command the loop retries and its honest cost is seconds,
while ``install`` is bounded by the step, because any per-command bound tight
enough to be useful is inside the healthy distribution.

What is asserted, therefore:

1. every step that runs ``apt-get`` declares a literal ``timeout-minutes``,
   strictly less than its job's, so the step owns the reap and names itself;
2. every ``apt-get`` whose exit status drives a loop condition carries a
   literal ``timeout``, so a hang becomes the failure the loop handles;
3. each such ``timeout`` is smaller than its own step's bound, or it could
   never fire.

Parsing is deliberately line-based rather than via ``yaml``: ``tests/`` is
type-checked under ``ignore_missing_imports = false`` and ``types-PyYAML`` is
not a dev dependency, so importing it would either fail ``mypy`` or require a
dependency change and a ``uv.lock`` relock.  The neighbouring workflow contract
pins (``tests/test_workflow_jobs_are_bounded.py``,
``tests/test_codeql_query_filters.py``) read their YAML the same way, and
``TestTheParserActuallySeesTheApt`` cross-checks the parse against a raw text
scan so a parser that silently stops matching fails instead of passing
vacuously.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"

#: ``apt-get`` in a command position. The raw scan uses the same token, so the
#: parser and the cross-check cannot disagree about what counts as one.
_APT = re.compile(r"\bapt-get\b")

#: ``timeout <seconds> [-flags] [sudo] apt-get ...``. ``sudo`` may sit on
#: either side of ``timeout``; both spellings bound the command.
_BOUNDED_APT = re.compile(r"\btimeout\s+(\d+)(?:\s+-{1,2}\S+)*\s+(?:sudo\s+)?apt-get\b")

#: A command whose exit status is read by a loop or branch condition. This is
#: the form a hang defeats, because the condition never evaluates.
_CONDITION_DRIVEN = re.compile(r"^\s*(?:if|while|until)\b")

#: A double- or single-quoted run of text. Stripped before anything is matched,
#: so the retry loop's own diagnostic - `echo "apt-get update failed ..."` -
#: reads as the string it is rather than as a third invocation of apt-get.
_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")


def _commands_only(line: str) -> str:
    """``line`` with quoted text removed, leaving what the shell would run."""
    return _QUOTED.sub(" ", line)


_JOBS_KEY = re.compile(r"^jobs:\s*$")
_TOP_LEVEL_KEY = re.compile(r"^\S")
_JOB_HEADER = re.compile(r"^ {2}([A-Za-z0-9_-]+):\s*$")
_JOB_KEY = re.compile(r"^ {4}([A-Za-z0-9_-]+):(.*)$")
_STEP_START = re.compile(r"^ {6}- (?:([A-Za-z0-9_-]+):(.*))?$")
_STEP_KEY = re.compile(r"^ {8}([A-Za-z0-9_-]+):(.*)$")


def _literal_minutes(raw: str | None) -> int | None:
    """``timeout-minutes`` as an int, or ``None`` when absent or an expression.

    An expression (``${{ ... }}``) reads as absent on purpose: this guard can
    only vouch for a value it can compare against a job's bound.
    """
    if raw is None:
        return None
    try:
        return int(raw.split("#", 1)[0].strip())
    except ValueError:
        return None


class AptInvocation(NamedTuple):
    """One ``apt-get`` command line, with the bounds that apply to it."""

    workflow: str
    job_id: str
    step_name: str
    command: str
    job_timeout_minutes: int | None
    step_timeout_minutes: int | None
    bound_seconds: int | None
    condition_driven: bool

    @property
    def ref(self) -> str:
        return f"{self.workflow}:{self.job_id}:{self.step_name}: {self.command}"

    def __repr__(self) -> str:  # pragma: no cover - test ids only
        return self.ref


def _parse(path: Path) -> list[AptInvocation]:
    """Collect every ``apt-get`` line under ``jobs.<id>.steps[].run``."""
    found: list[AptInvocation] = []
    lines = path.read_text().splitlines()

    in_jobs = False
    job_id = ""
    job_keys: dict[str, str] = {}
    step_name = ""
    step_keys: dict[str, str] = {}
    in_run_block = False

    def flush_step() -> None:
        nonlocal in_run_block
        in_run_block = False

    for line in lines:
        if _JOBS_KEY.match(line):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        if _TOP_LEVEL_KEY.match(line):
            # A sibling of `jobs:` ends the section.
            in_jobs = False
            flush_step()
            continue

        header = _JOB_HEADER.match(line)
        if header:
            job_id, job_keys = header.group(1), {}
            step_name, step_keys = "", {}
            flush_step()
            continue

        job_key = _JOB_KEY.match(line)
        if job_key:
            job_keys.setdefault(job_key.group(1), job_key.group(2))
            step_name, step_keys = "", {}
            flush_step()
            continue

        step_start = _STEP_START.match(line)
        if step_start:
            # A new list item ends the previous step, whatever it was running.
            step_name, step_keys = "", {}
            flush_step()
            if step_start.group(1):
                step_keys[step_start.group(1)] = step_start.group(2)
                if step_start.group(1) == "name":
                    step_name = step_start.group(2).strip()
            continue

        step_key = _STEP_KEY.match(line)
        if step_key:
            key, value = step_key.group(1), step_key.group(2)
            step_keys.setdefault(key, value)
            if key == "name":
                step_name = value.strip()
            # `run: |` opens a block; `run: cmd` is a single command line.
            in_run_block = key == "run"
            if in_run_block and _APT.search(_commands_only(value)):
                found.append(_invocation(path, job_id, step_name, value, job_keys, step_keys))
            continue

        if in_run_block:
            body = line.strip()
            if body.startswith("#"):
                # A comment quoting apt-get is not an invocation of it.
                continue
            if _APT.search(_commands_only(body)):
                found.append(_invocation(path, job_id, step_name, body, job_keys, step_keys))

    return found


def _invocation(
    path: Path,
    job_id: str,
    step_name: str,
    command: str,
    job_keys: dict[str, str],
    step_keys: dict[str, str],
) -> AptInvocation:
    bound = _BOUNDED_APT.search(_commands_only(command))
    return AptInvocation(
        workflow=path.name,
        job_id=job_id,
        step_name=step_name or "(unnamed)",
        command=command.strip(),
        job_timeout_minutes=_literal_minutes(job_keys.get("timeout-minutes")),
        step_timeout_minutes=_literal_minutes(step_keys.get("timeout-minutes")),
        bound_seconds=int(bound.group(1)) if bound else None,
        condition_driven=bool(_CONDITION_DRIVEN.match(command)),
    )


def _raw_apt_lines(path: Path) -> list[str]:
    """Every non-comment line naming ``apt-get``, ignoring structure.

    The cross-check for the parser: a structural parse that stops matching the
    tree would otherwise report an empty set and pass everything below.
    """
    out = []
    for line in path.read_text().splitlines():
        body = line.strip()
        if body.startswith("#") or not _APT.search(_commands_only(body)):
            continue
        # `run: |`'s own key line and the shell body both count; a YAML key
        # naming apt-get elsewhere does not exist in this tree.
        out.append(body)
    return out


_WORKFLOW_FILES = sorted(_WORKFLOWS.glob("*.yml"))
_INVOCATIONS = [inv for path in _WORKFLOW_FILES for inv in _parse(path)]
_APT_WORKFLOWS = sorted({inv.workflow for inv in _INVOCATIONS})


class TestTheParserActuallySeesTheApt:
    """A structural guard that parses nothing passes vacuously. Pin that it does not."""

    def test_workflows_are_found(self) -> None:
        assert _WORKFLOWS.is_dir(), f"{_WORKFLOWS} is missing"
        assert _WORKFLOW_FILES, "no workflow files were found"

    def test_apt_invocations_are_found(self) -> None:
        assert len(_INVOCATIONS) >= 3, (
            f"parsed only {len(_INVOCATIONS)} apt-get invocations; the tree has at least three "
            f"(one looped update plus an install, in each of two workflows)"
        )
        assert len(_APT_WORKFLOWS) >= 2, (
            f"apt-get was found in {_APT_WORKFLOWS}; it is installed by more than one workflow, "
            f"so a single-file result means the parser stopped seeing the tree"
        )

    @pytest.mark.parametrize("path", _WORKFLOW_FILES, ids=lambda p: p.name)
    def test_the_parse_finds_every_raw_apt_line(self, path: Path) -> None:
        """Structural parse vs. raw text scan, per file, so drift cannot hide."""
        parsed = [inv.command for inv in _INVOCATIONS if inv.workflow == path.name]
        raw = _raw_apt_lines(path)
        assert len(parsed) == len(raw), (
            f"{path.name}: parsed {len(parsed)} apt-get invocations but a raw scan finds "
            f"{len(raw)}\nparsed: {parsed}\nraw:    {raw}"
        )

    @pytest.mark.parametrize("inv", _INVOCATIONS, ids=lambda i: i.ref)
    def test_every_invocation_is_attributed_to_a_job(self, inv: AptInvocation) -> None:
        """An invocation attributed to no job is a parse failure, not a finding."""
        assert inv.job_id, f"{inv.command} was parsed outside any job"

    def test_at_least_one_invocation_is_condition_driven(self) -> None:
        """The retry-loop clause below is checked against a real loop, not none."""
        assert [inv for inv in _INVOCATIONS if inv.condition_driven], (
            "no apt-get invocation reads as loop/branch-condition driven, so the clause that "
            "requires those to carry `timeout` is vacuous"
        )


class TestEveryAptStepIsBoundedByItsOwnCeiling:
    """A job-level reap names no step; a step-level one does (#2442)."""

    @pytest.mark.parametrize("inv", _INVOCATIONS, ids=lambda i: i.ref)
    def test_the_step_declares_a_timeout(self, inv: AptInvocation) -> None:
        assert inv.step_timeout_minutes is not None, (
            f"{inv.workflow}:{inv.step_name} runs apt-get and declares no literal "
            f"timeout-minutes, so a wedged mirror is reaped by the job instead - which spends "
            f"the whole job budget and reports FAILURE with no step named (#2442)"
        )

    @pytest.mark.parametrize("inv", _INVOCATIONS, ids=lambda i: i.ref)
    def test_the_step_bound_fires_before_the_job_bound(self, inv: AptInvocation) -> None:
        """Equal bounds are a coin flip over which reap the log records."""
        step, job = inv.step_timeout_minutes, inv.job_timeout_minutes
        assert job is not None, (
            f"{inv.workflow}:{inv.job_id} declares no literal timeout-minutes; see "
            f"tests/test_workflow_jobs_are_bounded.py"
        )
        assert step is not None and step < job, (
            f"{inv.workflow}:{inv.step_name} declares timeout-minutes: {step} inside a job "
            f"bounded at {job}; the step bound has to be the tighter one or the job reaps first "
            f"and the verdict names no step"
        )


class TestARetryLoopReadsAnExitStatusSoItsCommandIsBounded:
    """``timeout`` turns a hang into exit 124 - the failure the loop handles."""

    @pytest.mark.parametrize(
        "inv",
        [inv for inv in _INVOCATIONS if inv.condition_driven],
        ids=lambda i: i.ref,
    )
    def test_a_condition_driven_invocation_carries_a_timeout(self, inv: AptInvocation) -> None:
        assert inv.bound_seconds is not None, (
            f"{inv.workflow}:{inv.step_name} reads apt-get's exit status in a loop/branch "
            f"condition without `timeout`: a mirror that never answers leaves the condition "
            f"unevaluated, so the next attempt, the backoff and the diagnostic echo are all "
            f"unreachable and only the job ceiling ends the step (#2442)\n  {inv.command}"
        )

    @pytest.mark.parametrize(
        "inv",
        [inv for inv in _INVOCATIONS if inv.bound_seconds is not None],
        ids=lambda i: i.ref,
    )
    def test_the_command_bound_can_actually_fire(self, inv: AptInvocation) -> None:
        """A per-command bound at or above its step's ceiling never runs out."""
        step = inv.step_timeout_minutes
        assert step is not None, f"{inv.ref} carries a command bound inside an unbounded step"
        assert inv.bound_seconds is not None and inv.bound_seconds < step * 60, (
            f"{inv.workflow}:{inv.step_name} bounds a command at {inv.bound_seconds}s inside a "
            f"step bounded at {step} min ({step * 60}s), so the step is reaped first and the "
            f"command bound never fires"
        )
