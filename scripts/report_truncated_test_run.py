#!/usr/bin/env python3
"""Report whether the required test suite ran to completion, and name what it skipped.

Why this exists
---------------
``.github/workflows/test-lint.yml`` runs the one required check as::

    hatch run test -x --strict-markers

``-x`` stops the session at the first failure. The check is not wrong -- it is
red when the tree is red -- but **the report is not a count**. A red
``call-test-lint`` names one failing cell, and the number of failing cells is
unknown until the run is repeated on a tree where that one passes.

Measured on run 33690980247, the ``Test and Lint`` job of #3161 at ``48881ea``::

    line  2207:  collecting ... collected 46583 items
    line 37134:  !!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!
    line 37135:  ==== 1 failed, 34268 passed, 278 skipped, 57 warnings in 1259.55s ====

34268 + 278 + 1 = 34547 of 46583 items executed, so **12036 items -- 25.8% of
the suite -- never ran**. The failure that was hidden was neither hypothetical
nor distant: reproduced on the same commit, the next failure was four lines
below the first, in the same class, with the same cause. Both would have been in
one report from an un-truncated run.

Why the numbers above are not already legible
---------------------------------------------
pytest *does* emit the ``stopping after N failures`` banner, so the truncation is
not literally unrecorded. Three things stop that from being an answer:

- The banner does not say how much was skipped. The count needs the
  ``collected`` line as well, and the two sit **34,927 lines apart** in a 6.4 MB
  log -- nobody subtracts them by hand.
- The banner sits one line above the summary, which is the line every reader is
  already looking at, so it reads as part of the failure rather than as a
  statement about the run's extent.
- Neither number reaches a surface short of downloading that log. A check's
  annotations and the job summary are what a reviewer sees, and the truncation
  appears in neither.

``--cov-fail-under=80`` is not a backstop either: on that truncated run coverage
reported ``Total coverage: 81.54%`` and the gate read as a pass, so it adds no
signal that the run was short.

Why the flag stays
------------------
Removing ``-x`` would make a red run cost what a green one costs, and that is
already close to the bound. Measured over the last 40 ``main`` pushes of
``pr-and-push.yml`` (2026-09-01T23:51Z -> 2026-09-03T02:11Z), the ``Test and
Lint`` job on the 37 runs that completed took **32.6 to 57.5 minutes, median
46.9**, against the job's ``timeout-minutes: 60`` -- and run 33631903156 was
already reaped at **60.12 minutes** with every step reporting success. The
bound cannot absorb the difference, and raising it is explicitly spent: the
comment on that ``timeout-minutes`` line records that "the next raise cannot be
spent here" (#2457, #2239).

So the early exit is load-bearing and this module does not touch it. What it
removes is the *ambiguity*: an aborted run and a complete run with one failure
are indistinguishable in the summary line, and that -- not the early exit -- is
what costs a round, because each hidden failure costs ~21 minutes of runner time
plus one review approval, approval being this repository's scarcest resource
(#1905).

A run killed mid-suite reports differently again
-----------------------------------------------
A job killed while pytest is still running writes no summary line at all, which
is the other way a run can be incomplete. It is reported as
``incomplete-no-summary`` rather than folded into either of the two above, so a
run that never finished reporting is not read as a run with one failure.

That branch is defensive rather than observed, and the distinction is worth
keeping straight. The one reaped job in the measured window, run 33631903156 at
60.12 minutes, turns out **not** to be an instance: its suite completed
(46449 of 46449 items, ``46152 passed, 297 skipped``) at 54.5 minutes and the
job was killed in a later step. So the reap in #3143 did not truncate that
suite, and this module reports it as ``complete``, which is the honest answer.

Why this never fails the job
----------------------------
It is wired into the required check's own job, so a nonzero exit here would be
indistinguishable from the suite failing -- and on a *green* run a parsing bug
would turn the one required check red for every open pull request. The verdict
this module produces is a description of a run whose outcome is already decided
by pytest, so it is reported and never gated: ``main`` returns 0 for every
input, including a log it cannot read. That is deliberate and is pinned by
tests/test_truncated_test_run_is_reported.py.

One owner for the subtraction
-----------------------------
``tests/session_truncation.py`` states the same extent in process, from
``len(session.items)`` and a count of the items it saw entered, and writes it
into the log this module reads. Two derivations of one number can disagree, so
when that section is present it settles the extent and the text arithmetic here
is only the fallback -- for a log written before the reporter was wired, or one
whose reporter never got to speak.

The fallback is measured on items, not on outcome labels. The summary line counts
every outcome the session produced, including the ones *collection* produced: a
module that fails to import is an ``error`` and one that skips itself for an
absent optional dependency is a ``skipped``, and neither was ever one of the
selected items. Both are already reported on the collection line, which is also
where the post-deselection ``selected`` count comes from, so both numbers in the
subtraction are read off that one line.

Usage
-----
::

    python3 scripts/report_truncated_test_run.py "$RUNNER_TEMP/pytest.log"

See issue #3164, #3143 for the bound, and the "PR Workflow" section of
AGENTS.md.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field

# pytest's collection line. Matched with ``search`` rather than anchored,
# because the same text is read both from the raw file this script is pointed at
# and, when a reader pastes one, from a job log whose lines carry a timestamp
# prefix added by the Actions log service.
#
# Only the item count is read positionally. Everything after it is a sequence of
# ``/ <n> <label>`` tokens that pytest appends conditionally, in this order
# (``TerminalReporter.report_collect``)::
#
#     collected N items[ / E errors][ / D deselected][ / S skipped][ / X selected]
#
# so the tokens are scanned by label rather than matched in a fixed sequence. A
# pattern that expected ``deselected`` and ``selected`` adjacently reads
# ``selected`` as absent the moment either of the other two lands between them -
# which happens on any run that hits a collection error or a module-level skip -
# and then falls back to the pre-deselection total, inflating "never ran" by the
# deselected count on a run that ran everything it selected.
_COLLECTED = re.compile(r"collected (?P<collected>\d+) items?(?P<tokens>[^\r\n]*)")
_COLLECT_TOKEN = re.compile(r"/ (?P<count>\d+) (?P<label>deselected|selected|skipped|errors?)")

# The banner ``-x`` / ``--maxfail`` prints when it ends the session early.
_STOPPING = re.compile(r"!+\s*stopping after (?P<failures>\d+) failures?\s*!+")

# The same subtraction, already computed in process and written into this very
# log by ``tests/session_truncation.py``. That reporter holds
# ``len(session.items)`` and a count of the items it saw entered, so when its
# section is present it is the authoritative answer and this module reads it
# instead of deriving a second one from text. Anchored on the heading rather
# than on the surrounding rule so the section is found whatever width the
# terminal chose for it.
_SESSION_STATEMENT = re.compile(r"session truncated: (?P<started>\d+) of (?P<collected>\d+) collected tests ran")

# The final ``= 1 failed, 34268 passed, ... in 1259.55s =`` line. Only the
# surrounding rule and the trailing duration are fixed; the counts in between
# vary, so they are read by scanning rather than by one exhaustive alternation.
_SUMMARY_LINE = re.compile(r"^=+ .*\bin \d+(?:\.\d+)?s.*=+$")
_COUNT = re.compile(r"(?P<count>\d+) (?P<label>[a-z]+)")

# The Actions log service prefixes every line it stores with an RFC3339 stamp.
# The file this script is pointed at in CI is the raw `tee` copy and carries
# none, but a maintainer diagnosing a past run downloads the stored log, and the
# summary line is the one pattern anchored to the start of a line -- so without
# this the script would silently report ``incomplete-no-summary`` on exactly the
# input a human reaches for. Stripped rather than tolerated in each pattern so
# there is one place that knows about the prefix.
_LOG_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z ")

# Outcome labels that mean a test item was actually executed. ``warnings`` and
# ``deselected`` are deliberately absent: a warning is not an item, and a
# deselected item was never going to run, so counting either would understate
# what the abort skipped. ``rerun`` is absent because a rerun is a second
# attempt at an item already counted under its final outcome.
_EXECUTED_LABELS = frozenset({"failed", "passed", "skipped", "xfailed", "xpassed", "error", "errors"})

_COMPLETE = "complete"
_TRUNCATED = "truncated"
_NO_SUMMARY = "incomplete-no-summary"
_UNREADABLE = "unreadable"


@dataclass
class RunReport:
    """What one pytest log says about its own extent."""

    outcome: str
    collected: int | None = None
    selected: int | None = None
    executed: int | None = None
    never_ran: int | None = None
    stopped_after: int | None = None
    counts: dict[str, int] = field(default_factory=dict)
    detail: str = ""
    #: Whether the extent was read from the session reporter's own statement
    #: rather than re-derived from the counts line. Reported so a maintainer
    #: reading the summary can tell which owner produced the numbers.
    stated_by_session: bool = False

    @property
    def share_skipped(self) -> float | None:
        """Fraction of the selected items that never ran, or None if unknown."""
        if self.selected in (None, 0) or self.never_ran is None:
            return None
        assert self.selected is not None
        return self.never_ran / self.selected

    @property
    def summary(self) -> str:
        """One line naming the run's extent, suitable for an annotation."""
        if self.outcome == _COMPLETE:
            return f"the suite ran to completion: {self.executed} of {self.selected} items executed"
        if self.outcome == _TRUNCATED:
            share = self.share_skipped
            share_text = f" ({share:.1%} of the suite)" if share is not None else ""
            return (
                f"the run was truncated: {self.executed} of {self.selected} items executed, "
                f"{self.never_ran} never ran{share_text}. The failure count is a lower "
                f"bound, not a total."
            )
        if self.outcome == _NO_SUMMARY:
            return (
                "the run produced no summary line, so it did not finish reporting and "
                "neither its failure count nor its extent is known."
            )
        return f"the run's extent could not be read: {self.detail}"


def _collection_tokens(tail: str) -> dict[str, int]:
    """Read the ``/ <n> <label>`` tokens pytest appends to its collection line.

    Args:
        tail: Whatever followed the item count on that line.

    Returns:
        One entry per token, keyed by label. ``errors`` is normalised to
        ``error`` so that one key means one thing whatever the count was.
    """
    tokens: dict[str, int] = {}
    for match in _COLLECT_TOKEN.finditer(tail):
        label = match.group("label")
        tokens["error" if label.startswith("error") else label] = int(match.group("count"))
    return tokens


def _outcome_count(counts: dict[str, int], label: str) -> int:
    """Read one summary count under either its singular or plural spelling.

    pytest writes ``1 error`` and ``2 errors``, so a label read off the summary
    line is not a stable key on its own.
    """
    return counts.get(label, 0) + counts.get(label + "s", 0)


def parse_run(text: str) -> RunReport:
    """Read a pytest log and report whether the session covered every selected item.

    Args:
        text: The captured stdout of one pytest session.

    Returns:
        A RunReport. ``outcome`` is ``complete`` when every selected item was
        accounted for, ``truncated`` when the session ended early, and
        ``incomplete-no-summary`` when no summary line was written at all --
        which is what a job killed at its timeout bound leaves behind.
    """
    collected: int | None = None
    selected: int | None = None
    stopped_after: int | None = None
    summary: str | None = None
    stated: tuple[int, int] | None = None
    collect_tokens: dict[str, int] = {}

    for raw in text.splitlines():
        line = _LOG_TIMESTAMP.sub("", raw)
        if collected is None:
            found = _COLLECTED.search(line)
            if found:
                collected = int(found.group("collected"))
                collect_tokens = _collection_tokens(found.group("tokens"))
                if "selected" in collect_tokens:
                    selected = collect_tokens["selected"]
                elif "deselected" in collect_tokens:
                    # pytest prints ``selected`` whenever it differs from the
                    # item count, so this branch is unreached today. It is the
                    # subtraction rather than a second fallback to ``collected``
                    # so that a release which stopped printing the token would
                    # narrow the report, not silently widen it.
                    selected = collected - collect_tokens["deselected"]
                else:
                    selected = collected
        stopping = _STOPPING.search(line)
        if stopping:
            stopped_after = int(stopping.group("failures"))
        said = _SESSION_STATEMENT.search(line)
        if said:
            stated = (int(said.group("started")), int(said.group("collected")))
        stripped = line.strip()
        if _SUMMARY_LINE.match(stripped):
            summary = stripped

    if collected is None:
        return RunReport(
            outcome=_UNREADABLE,
            stopped_after=stopped_after,
            detail="no 'collected N items' line was found",
        )

    if summary is None:
        return RunReport(
            outcome=_NO_SUMMARY,
            collected=collected,
            selected=selected,
            stopped_after=stopped_after,
            detail="no pytest summary line was found",
        )

    counts = {
        match.group("label"): int(match.group("count"))
        for match in _COUNT.finditer(summary)
        if match.group("label") in _EXECUTED_LABELS
    }
    executed = sum(counts.values())
    # The summary line labels every outcome the session produced, including the
    # ones *collection* produced: a module that failed to import is an ``error``
    # and one that skipped itself is a ``skipped``, and neither was ever one of
    # the selected items. Counting them as executed items compares a count of
    # outcomes against a count of items, and the difference between those two is
    # not "never ran" - it understates it by exactly the collection-phase
    # outcomes. The collection line already reported them, so they are
    # subtracted here from the same line that supplied ``selected``.
    for label in ("error", "skipped"):
        at_collection = collect_tokens.get(label, 0)
        if at_collection:
            executed -= min(at_collection, _outcome_count(counts, label))
    executed = max(executed, 0)
    assert selected is not None
    never_ran = max(selected - executed, 0)
    if stated is not None:
        # One owner for the subtraction. The reporter held the session's own
        # item list, so its pair settles the extent and the arithmetic above
        # stays as the fallback for a log that carries no such section - a run
        # from before it was wired, or one whose reporter never got to speak.
        started, session_collected = stated
        selected = session_collected
        executed = started
        never_ran = max(session_collected - started, 0)
    truncated = stopped_after is not None or never_ran > 0

    return RunReport(
        outcome=_TRUNCATED if truncated else _COMPLETE,
        collected=collected,
        selected=selected,
        executed=executed,
        never_ran=never_ran,
        stopped_after=stopped_after,
        counts=counts,
        stated_by_session=stated is not None,
    )


def render(report: RunReport) -> str:
    """Render a report as the markdown written to the job summary."""
    lines = ["## Did the required suite run to completion?", ""]
    lines.append(report.summary)
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append(f"| outcome | `{report.outcome}` |")
    if report.collected is not None:
        lines.append(f"| items collected | {report.collected} |")
    if report.selected is not None and report.selected != report.collected:
        lines.append(f"| items selected | {report.selected} |")
    if report.executed is not None:
        lines.append(f"| items executed | {report.executed} |")
    if report.never_ran is not None:
        lines.append(f"| items that never ran | {report.never_ran} |")
    if report.stopped_after is not None:
        lines.append(f"| stopped after | {report.stopped_after} failure(s) |")
    if report.stated_by_session:
        lines.append("| extent stated by | the session reporter, in process |")
    for label in sorted(report.counts):
        lines.append(f"| {label} | {report.counts[label]} |")

    if report.outcome == _TRUNCATED:
        lines.extend(
            [
                "",
                "The suite runs under `-x`, so it stops at the first failure and the",
                "remaining items are not evidence of anything. Fix the failure named",
                "above and expect the next run to reach further -- possibly onto another",
                "failure that this run could not reach. The flag is deliberate: a full",
                "run of this suite measures 32.6 to 57.5 min against a 60 min bound, so",
                "letting a red run continue would put it in reach of the reap (#3143).",
            ]
        )
    elif report.outcome == _NO_SUMMARY:
        lines.extend(
            [
                "",
                "No summary line means pytest did not finish reporting, so neither the",
                "failure count nor the extent of this run is known. A job killed while",
                "the suite was still running looks like this; a reap at the",
                "`timeout-minutes` bound renders as CANCELLED and so reads as a",
                "concurrency cancel rather than a timeout (#3143). Check the job's",
                "duration before re-running it.",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Report a run's extent to the job summary and as an annotation.

    Always returns 0. See this module's docstring: the verdict describes a run
    whose pass/fail outcome pytest has already decided, and this runs inside the
    one required check's job, so failing here would either duplicate the suite's
    own red or -- on a parsing bug -- invent one on a green tree.
    """
    parser = argparse.ArgumentParser(description="Report whether a pytest run was truncated.")
    parser.add_argument("log", help="Path to the captured pytest output.")
    args = parser.parse_args(argv)

    try:
        with open(args.log, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        report = RunReport(outcome=_UNREADABLE, detail=f"{args.log} could not be read ({exc.strerror})")
    else:
        report = parse_run(text)

    document = render(report)
    print(document)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as handle:
                handle.write(document)
        except OSError as exc:
            print(f"note: could not write the job summary ({exc.strerror})")

    if report.outcome == _TRUNCATED:
        print(f"::warning title=The test run was truncated::{report.summary}")
    elif report.outcome == _NO_SUMMARY:
        print(f"::warning title=The test run did not finish reporting::{report.summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
