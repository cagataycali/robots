"""Regression: a refusal that defers pending work may not cite a change this
repository has already landed.

A tracker reference inside a refusal is a *remedy*: the reader follows it to
learn what is missing and when it arrives.  Two speech acts carry such a
reference and they have opposite requirements.

A *backward citation* explains where text came from - ``"#168 bug I: the cached
scene diverges"`` - and the reader wants the history, so a merged pull request
is exactly the right referent.  A *forward deferral* says the capability is not
here yet - ``"not wired yet (issue #359 bus)"`` - and the reader wants to know
when it lands, so the referent must be work that is still outstanding.  A
forward deferral pointing at a change this repository already shipped is
self-contradicting: the reader opens it, finds a merged change about an
unrelated subsystem, and concludes the deferral is stale rather than that the
capability is missing.

That is a strictly worse failure than an unresolvable reference.  An unowned
``harness#361`` fails loudly - the reader has no link to follow, so they know
the remedy is incomplete.  A bare ``#359`` resolves, looks authoritative, and
misinforms.  ``tests/test_source_strings_resolve_their_issue_references.py``
owns the resolvability half of this contract; this module owns the destination
half, and both read the same caller-reachable scope so the two rules cannot
disagree about which strings an operator sees.

It would have failed while three caller-reachable strings deferred work to
numbers this repository had already merged:

* ``drivers/dynamixel/driver.py`` refused four verbs with ``"not wired yet
  (issue #359 bus)"``, and ``#359`` is "fix(sim): drive tendon-transmission
  actuators via joint name" - merged, and about the simulator rather than a
  servo bus.  The same number reached a planning agent through the driver's
  ``tool_spec`` description.
* ``drivers/g1.py`` deferred ``start_task`` to ``#358``, which is "test(mesh):
  fix flaky test_session_config" - merged, and about zenoh mock isolation.

Neither capability has an open issue in this repository, so the fix removes the
misdirecting reference and names the missing thing instead.  The rule needs both
conditions: dropping the deferral test would flag the four legitimate backward
citations, and dropping the landed test would flag every deferral including the
one that correctly points at open issue ``#2765``.

Scope: both caller-reachable literals and docstrings, split by *speech act*
rather than by audience.  A docstring may name a merged change freely when it
is explaining where text came from - that is the backward citation above, and a
maintainer reading it has the git history to hand.  A docstring that *defers*
is a different claim: it tells the next contributor a capability is still
outstanding and where to watch for it, and git history cannot answer "when will
this land".  So the same two conditions apply to a docstring sentence, and the
scope is the sentence rather than the whole docstring - one paragraph routinely
carries a deferral and a backward citation, and only the deferral's own
sentence is graded.

It would have failed while seven docstring sentences in
``strands_robots/drivers/g1.py`` deferred to ``#358`` and ``#361``, both merged
pull requests about unrelated subsystems.  One of them was worse than
misdirecting: ``send_action`` told the reader "the loop lands in the follow-up
PR that closes issue #361 in full" after that loop had shipped, so a
contributor was told a capability the module already provides was future work.

Out of scope: ``#`` comments.  A comment is neither read by a caller nor part
of the rendered API surface, and the extraction the two rules share does not
collect them.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

from tests.test_source_strings_resolve_their_issue_references import (
    _PACKAGE_DIR,
    _caller_reachable_literals,
    _python_sources,
)

_REPO_ROOT = _PACKAGE_DIR.parent

# A bare in-repository reference.  A slug-qualified ``owner/repo#12`` names a
# different tracker and is the sibling module's subject, not this one.
_BARE_ISSUE_REF = re.compile(r"(?<![A-Za-z0-9_.\-/])#(\d+)\b")

# The squash subject this repository writes for a landed pull request.  One
# historical subject carries shell quoting around the number.
_LANDED_SUBJECT = re.compile(r"\(#'?(\d+)'?\)\s*$", re.MULTILINE)

# A forward deferral: the text says the capability is still outstanding.  The
# last two alternatives were added with the docstring rule below; measured
# against the whole package they add no caller-reachable offender, so the
# widening costs the string rule nothing and lets the docstring rule reach five
# sentences the original vocabulary walked past.
_DEFERRAL = re.compile(
    r"not wired|not yet|unwired|pending|until .{0,40}\blands?\b|\blands? in\b|future work",
    re.IGNORECASE,
)

# A shallow clone carries one commit, which would make the oracle empty and the
# rule vacuous.  The history held 2108 landed numbers when this landed.
_MINIMUM_LANDED_NUMBERS = 500


def _landed_pull_request_numbers_or_skip() -> frozenset[int]:
    """Numbers this repository's own history records as landed pull requests.

    Every path out of this helper is explicit - a return or a raise - so the
    caller's binding is unconditional and ``completed`` is never live only on
    the success path.  ``pytest.skip.Exception`` is raised rather than
    :func:`pytest.skip` called for the same reason, which is the idiom
    ``tests/test_optional_dependency_skips_bind_their_names.py`` prescribes.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "log", "--format=%s"],  # noqa: S607 - git resolved from PATH by design
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - no git on PATH
        raise pytest.skip.Exception(f"git history unavailable, so landed numbers cannot be resolved: {exc}") from exc
    if completed.returncode != 0:  # pragma: no cover - not a git checkout
        raise pytest.skip.Exception("git log failed, so landed numbers cannot be resolved")
    return frozenset(int(match.group(1)) for match in _LANDED_SUBJECT.finditer(completed.stdout))


def _landed_citations(text: str, landed: frozenset[int]) -> list[int]:
    """Numbers a forward deferral cites that this repository already landed."""
    if not _DEFERRAL.search(text):
        return []
    return [int(m.group(1)) for m in _BARE_ISSUE_REF.finditer(text) if int(m.group(1)) in landed]


def _offenders(landed: frozenset[int]) -> list[str]:
    found: list[str] = []
    for path in _python_sources():
        for lineno, literal in _caller_reachable_literals(path):
            for number in _landed_citations(literal, landed):
                found.append(f"{Path(path).relative_to(_REPO_ROOT)}:{lineno}: #{number}")
    return found


# =========================================================================
# Premises: the oracle and the scanned population are both non-empty.
# =========================================================================


def test_the_landed_number_oracle_is_populated() -> None:
    """Non-vacuity: a shallow clone must skip here rather than pass the rule.

    The rule below scans caller-reachable literals against a set of landed
    pull-request numbers.  If the environment carries fewer than
    ``_MINIMUM_LANDED_NUMBERS`` landed numbers (a shallow clone, a fork-PR
    checkout whose ancestry back to ``origin/main`` was not part of the
    fetched pack, a fresh worktree that has not resolved the base branch),
    the rule silently degrades to a vacuous pass.  Skipping names the
    environment as the reason the rule cannot grade its subject, which is a
    stronger signal than a green tick from a scan that had nothing to scan
    against.

    A full-history checkout (a maintainer's ``git clone`` of the base
    repository, or CI's ``fetch-depth: 0`` that resolved the PR head against
    ``origin/main``) carries ~2100 numbers and passes this test; the sibling
    rule below then grades the tree.
    """
    landed = _landed_pull_request_numbers_or_skip()
    if len(landed) < _MINIMUM_LANDED_NUMBERS:
        pytest.skip(
            f"only {len(landed)} landed pull-request numbers found in git history; "
            "the rule below would be vacuous in this environment. A full-history "
            "checkout (git clone without --depth, or actions/checkout with "
            "fetch-depth: 0 whose PR-head fetch resolved against origin/main) is "
            "required to grade this contract. A fork-PR run whose checkout only "
            "received the head sha's own line reads as shallow here even when the "
            "workflow asked for fetch-depth: 0, because the sha's ancestry back to "
            "main was not part of the fetched pack."
        )


def test_the_scanned_population_is_not_empty() -> None:
    """Non-vacuity: the shared extraction still yields caller-reachable text."""
    total = sum(len(_caller_reachable_literals(path)) for path in _python_sources())
    assert total > 1000, f"only {total} caller-reachable literals found; the extraction is too narrow"


# =========================================================================
# The rule.
# =========================================================================


def test_no_deferral_cites_a_change_this_repository_landed() -> None:
    """A string promising future work must not point at work already shipped."""
    landed = _landed_pull_request_numbers_or_skip()
    if len(landed) < _MINIMUM_LANDED_NUMBERS:
        pytest.skip(
            f"only {len(landed)} landed pull-request numbers found in git history; the "
            "rule cannot grade its subject in a shallow environment. The oracle test "
            "above owns the environment check; this rule declines rather than reads "
            "green from an empty oracle."
        )
    offenders = _offenders(landed)
    assert not offenders, (
        "A caller-reachable string defers a capability and cites a change this repository has "
        "already merged. The reader follows it, finds a landed change about another subsystem, "
        "and concludes the refusal is stale. Cite an open issue that tracks the work, or name "
        "the missing capability without a tracker reference:\n" + "\n".join(offenders)
    )


# =========================================================================
# Constructed exemplars: the corpus is clean after the fix, so the rule is
# graded on inputs rather than on the tree it scans.
# =========================================================================


def test_a_deferral_citing_a_landed_change_is_flagged() -> None:
    """The shape this guard exists to keep out."""
    landed = frozenset({359})
    assert _landed_citations("not wired yet (issue #359 bus)", landed) == [359]


def test_a_deferral_citing_an_open_issue_is_not_flagged() -> None:
    """The correct forward deferral: an issue that is still outstanding."""
    landed = frozenset({358, 359})
    text = "FSM id unknown - motion-switcher source has not been wired; see issue #2765"
    assert _landed_citations(text, landed) == []


def test_a_backward_citation_to_a_landed_change_is_not_flagged() -> None:
    """Explaining where text came from may name a merged change - that is its job."""
    landed = frozenset({168, 300, 2459})
    for text in (
        "#168 bug I: the cached scene diverges from upstream LIBERO's scene",
        "is missing on disk - refusing to degrade the object to a silent box proxy (#2459 fail-loud)",
    ):
        assert _landed_citations(text, landed) == [], f"should not be flagged: {text!r}"


def test_the_predicate_answers_both_ways() -> None:
    """Non-vacuity: the predicate is not a constant over the exemplars."""
    landed = frozenset({359, 168})
    outcomes = {
        bool(_landed_citations(text, landed))
        for text in ("not wired yet (issue #359 bus)", "#168 bug I: diverges", "see issue #2765")
    }
    assert outcomes == {True, False}


def test_both_conditions_are_load_bearing() -> None:
    """Either condition alone would flag text the other correctly accepts.

    Dropping the deferral test flags a backward citation; dropping the landed
    test flags the deferral that correctly points at an open issue.  The rule
    needs the conjunction, and this states why in a form that fails if either
    half is quietly widened.
    """
    landed = frozenset({168, 359})
    backward = "#168 bug I: the cached scene diverges"
    open_deferral = "not wired yet; see issue #2765 for the wire-side decision"

    # Deferral test dropped: every landed citation would be flagged.
    assert [int(m.group(1)) for m in _BARE_ISSUE_REF.finditer(backward) if int(m.group(1)) in landed] == [168]
    assert _landed_citations(backward, landed) == []

    # Landed test dropped: every deferral would be flagged.
    assert _DEFERRAL.search(open_deferral) is not None
    assert _landed_citations(open_deferral, landed) == []


# =========================================================================
# The same rule over docstrings, scoped to the sentence that defers.
# =========================================================================

# A docstring paragraph routinely carries a deferral and a backward citation
# in adjacent sentences, so the reference must sit in the deferring sentence
# to be graded.  Splitting is deliberately eager: an abbreviation splits a
# sentence in two, which can only narrow what the rule reads, and a narrower
# read declines to flag rather than flagging text it should not.
_SENTENCE_BREAK = re.compile(r"(?<=[.:;])\s+")

_DOCUMENTED_NODES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _docstring_sentences(path: Path) -> list[tuple[int, str, str]]:
    """Every ``(lineno, owner, sentence)`` triple in a module's docstrings.

    ``clean=False`` keeps the text as written; the sentence is whitespace-
    normalised so a reference wrapped across two source lines is still one
    token to the reference pattern.
    """
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, _DOCUMENTED_NODES):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if not docstring:
            continue
        owner = getattr(node, "name", "<module>")
        lineno = getattr(node, "lineno", 1)
        for sentence in _SENTENCE_BREAK.split(" ".join(docstring.split())):
            found.append((lineno, owner, sentence))
    return found


def _docstring_offenders(landed: frozenset[int]) -> list[str]:
    found: list[str] = []
    for path in _python_sources():
        for lineno, owner, sentence in _docstring_sentences(path):
            for number in _landed_citations(sentence, landed):
                found.append(f"{Path(path).relative_to(_REPO_ROOT)}:{lineno} ({owner}): #{number}")
    return found


def test_the_docstring_population_is_not_empty() -> None:
    """Non-vacuity: the docstring walk still yields sentences to grade."""
    total = sum(len(_docstring_sentences(path)) for path in _python_sources())
    assert total > 1000, f"only {total} docstring sentences found; the walk is too narrow"


def test_no_docstring_deferral_cites_a_change_this_repository_landed() -> None:
    """A docstring promising future work must not point at work already shipped."""
    landed = _landed_pull_request_numbers_or_skip()
    if len(landed) < _MINIMUM_LANDED_NUMBERS:
        pytest.skip(
            f"only {len(landed)} landed pull-request numbers found in git history; the "
            "rule cannot grade its subject in a shallow environment. The oracle test "
            "above owns the environment check; this rule declines rather than reads "
            "green from an empty oracle."
        )
    offenders = _docstring_offenders(landed)
    assert not offenders, (
        "A docstring defers a capability and cites a change this repository has already "
        "merged. The next contributor follows it, finds a landed change about another "
        "subsystem, and cannot tell whether the capability is missing or the note is "
        "stale. Cite an open issue that tracks the work, or name the missing capability "
        "without a tracker reference:\n" + "\n".join(offenders)
    )


def test_a_deferral_and_a_backward_citation_in_one_paragraph_are_graded_apart() -> None:
    """Sentence scope: the landed number must sit in the deferring sentence.

    This is the shape the whole-docstring reading got wrong.  ``g1.py``'s module
    docstring defers in one sentence and credits merged work in the next, and a
    paragraph-wide read blamed the deferral for the credit's reference.
    """
    landed = frozenset({354, 358})
    paragraph = "The provider registry is not yet plumbed here. The driver's job in issue #354 was the transport."
    sentences = _SENTENCE_BREAK.split(paragraph)
    assert len(sentences) == 2, sentences
    assert [n for s in sentences for n in _landed_citations(s, landed)] == []
    assert _landed_citations(paragraph, landed) == [354], "the paragraph-wide read is the defect"


def test_a_docstring_deferral_citing_a_landed_change_is_flagged() -> None:
    """The shape this half of the guard exists to keep out."""
    landed = frozenset({361})
    text = "The loop lands in the follow-up PR that closes issue #361 in full."
    assert _landed_citations(text, landed) == [361]


def test_a_docstring_deferral_citing_an_open_issue_is_not_flagged() -> None:
    """Deferring to outstanding work is what a deferral is for."""
    landed = frozenset({358, 361})
    text = "Until the motion-switcher source is wired (issue #2765), the gate refuses honestly."
    assert _landed_citations(text, landed) == []


def test_the_widened_vocabulary_reaches_the_sentences_it_was_added_for() -> None:
    """Non-vacuity of the widening: the two added phrases are load-bearing.

    Without them the docstring rule reads five of the seven ``g1.py`` sentences
    as prose rather than as deferrals, so the widening is what makes the rule
    cover the class instead of two instances of it.
    """
    landed = frozenset({358, 361})
    for text in (
        "The rich verb set lands in issue #358 as vendored neon tools.",
        "The rt/armsdk topic is future work for the g1_tools client in issue #358.",
    ):
        assert _landed_citations(text, landed), f"widened vocabulary should reach: {text!r}"
