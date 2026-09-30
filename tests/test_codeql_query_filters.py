"""Contract pins for the CodeQL query filters.

``.github/codeql/codeql-config.yml`` exists because a CodeQL alert on a pull
request is a hard merge gate in this repository, and two note-severity *quality*
rules in the ``security-and-quality`` suite fire only on idioms the codebase is
obliged to use. The gate is not the CodeQL job, which never fails on an alert:
``github-advanced-security`` opens a review thread per new alert and the
``default`` branch ruleset sets ``required_review_thread_resolution: true``, so
severity never enters into it. That interaction is invisible to the workflow,
which is how its own comment came to describe a policy the repository does not
implement. See #1810.

A suppression is the kind of change that decays by widening: the cheapest way to
clear any future alert is to append its rule id here, one line at a time, until
the file quietly opts out of the whole quality suite. So the properties below are
about *scope*, not about CodeQL working:

- the filter set is **exactly two** rule ids, named individually, so adding a
  third is a deliberate edit that fails this test until someone changes it;
- ``py/empty-except`` is **absent**, which #1810 names as an explicit non-goal --
  it is the largest class (88 open), a swallowed exception genuinely hides bugs,
  and the instances need reading one at a time;
- the config is **reachable**, i.e. the workflow actually passes it, since an
  unreferenced config file silently filters nothing;
- ruff still selects **B015 and B018**, which is the load-bearing one. Excluding
  ``py/ineffectual-statement`` is only a no-loss trade because the real no-op
  statement class moved to a check that is merge-blocking here where CodeQL is
  advisory. Drop those two codes and the exclusion silently becomes a capability
  loss, with nothing else in the tree recording the connection.

These are text assertions rather than parsed YAML because that is the shape the
existing CI-config pin uses (``tests/test_merge_base_overlap.py`` reads
``.github/workflows/merge-base-overlap.yml`` the same way) and because ``pyyaml``
is an optional dependency here -- a pin that skips when a dep is missing is not a
pin.
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from tests._package_ast import parse_file

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPO_ROOT / ".github" / "codeql" / "codeql-config.yml"
_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"

#: The only two rule ids this repository filters, and the reason each is here.
#:
#: ``py/ineffectual-statement`` -- 27 of 27 open alerts were ``...`` used as a
#: typing-construct body (``Protocol`` methods, ``@abstractmethod`` bodies,
#: ``@overload`` signatures, ``TYPE_CHECKING`` stubs). No rewrite exists.
#:
#: ``py/import-and-import-from`` -- 63 of 64 open alerts were the pytest
#: monkeypatch idiom, where the module alias is the patch target and the ``from``
#: import names the subject, so both are load-bearing.
_EXPECTED_EXCLUDED_RULES = frozenset(
    {
        "py/ineffectual-statement",
        "py/import-and-import-from",
    }
)

#: Ruff codes carrying the no-op-statement capability that the
#: ``py/ineffectual-statement`` exclusion would otherwise give up.
_RELOCATED_RUFF_CODES = ("B015", "B018")

#: Matches the two-line ``- exclude:`` / ``id:`` form the config is written in.
_EXCLUDED_ID_RE = re.compile(
    r"^[ \t]*-[ \t]*exclude:[ \t]*\r?\n[ \t]*id:[ \t]*(?P<rule>[A-Za-z0-9/_-]+)[ \t]*$",
    re.MULTILINE,
)


def _excluded_rule_ids() -> list[str]:
    return _EXCLUDED_ID_RE.findall(_CONFIG_PATH.read_text(encoding="utf-8"))


class TestTheFilterSetStaysNarrow:
    def test_the_config_file_exists(self):
        assert _CONFIG_PATH.is_file(), (
            f"{_CONFIG_PATH.relative_to(_REPO_ROOT)} is missing. If the CodeQL filters were "
            "removed on purpose, delete this module in the same change so the tree does not "
            "carry a pin for a file nobody has."
        )

    def test_exactly_the_two_documented_rules_are_excluded(self):
        found = _excluded_rule_ids()
        assert len(found) == len(set(found)), f"a rule id is excluded twice: {found}"
        assert set(found) == set(_EXPECTED_EXCLUDED_RULES), (
            "the CodeQL filter set changed. Every id here suppresses a real query for the whole "
            "repository, so adding one is a decision that needs its own reasoning recorded next to "
            "it in the config -- and then this expectation updated deliberately.\n"
            f"  expected: {sorted(_EXPECTED_EXCLUDED_RULES)}\n"
            f"  found:    {sorted(found)}"
        )

    def test_empty_except_is_not_excluded(self):
        text = _CONFIG_PATH.read_text(encoding="utf-8")
        assert "py/empty-except" not in _excluded_rule_ids(), (
            "py/empty-except must keep gating merges. It is the largest alert class, a swallowed "
            "exception genuinely hides bugs, and its instances are not one mechanical idiom - "
            "#1810 names quieting it as an explicit non-goal."
        )
        assert "py/empty-except" in text, (
            "the config should keep naming py/empty-except as the deliberate non-exclusion, so the "
            "next reader looking for it finds the reason rather than an omission."
        )

    def test_each_exclusion_carries_its_reasoning(self):
        """A bare rule id is how the next reader loses the argument for it."""
        text = _CONFIG_PATH.read_text(encoding="utf-8")
        assert "#1810" in text, "the config must link the issue that measured the cost"
        for rule in _EXPECTED_EXCLUDED_RULES:
            # The id appears once in a comment block explaining it and once in the
            # filter itself; a filter with no prose above it is the decay case.
            assert text.count(rule) >= 2, (
                f"{rule} is excluded without a comment naming why. A suppression with no stated "
                "reason cannot be re-litigated, only inherited."
            )


class TestTheConfigIsReachable:
    def test_the_workflow_passes_the_config_file(self):
        workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
        assert "config-file: ./.github/codeql/codeql-config.yml" in workflow, (
            "ci.yml must pass config-file, or the filters above are dead text: an "
            "unreferenced CodeQL config silently filters nothing and every alert keeps gating."
        )

    def test_the_workflow_no_longer_claims_alerts_do_not_block(self):
        """The comment that was false is the reason #1810 was filed."""
        workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
        assert "PRs are not blocked on" not in workflow, (
            "ci.yml (formerly codeql.yml) used to state that PRs are not blocked on CodeQL alerts. Thread-resolution "
            "on bot-authored review threads makes every new alert a merge gate, so that sentence "
            "described a policy the repository does not implement. Do not restore it."
        )
        assert "hard merge gate" in workflow, (
            "ci.yml must say what actually happens, not merely stop saying the wrong thing. A "
            "contributor reading it needs to know an alert blocks the merge before they spend a "
            "round wondering why an approved, green PR will not go in."
        )


class TestTheRelocatedCapabilityStaysSelected:
    def test_ruff_still_selects_the_no_op_statement_codes(self):
        pyproject = _PYPROJECT_PATH.read_text(encoding="utf-8")
        for code in _RELOCATED_RUFF_CODES:
            assert f'"{code}"' in pyproject, (
                f"ruff must keep selecting {code}. Excluding py/ineffectual-statement from CodeQL "
                "is only a no-loss trade because the real no-op-statement class is enforced by "
                "ruff, which gates merges here where CodeQL is advisory. Removing this code while "
                "the exclusion stands drops the capability with nothing recording it."
            )


class TestTheRulesFileSettlesTheCrossThreadMarshalClass:
    """``py/catch-base-exception`` clears under none of the three dispositions.

    The three tools the section offers are fix / dismiss-if-test-only /
    filter-if-every-instance-is-obliged. This rule's whole alert surface in the
    tree is one construct - a cross-thread exception-marshal box, the only
    ``except BaseException`` handler in the tree that does not re-raise lexically,
    measured by ``TestTheMarshalBoxCensusIsDerivedFromTheTree`` below - and
    each tool refuses it in turn: narrowing deletes a ``SystemExit`` outright
    (pinned below), the flagged site is not test-only, and not every instance is
    obliged because ``concurrent.futures`` is a genuine route whenever the caller
    owns the thread.

    Left unwritten, that gap does not read as a gap. It reads as a judgment call,
    and it cost #1899 two threads that each argued the idiom at length and then
    deferred to a human rather than applying a rule nobody had written down. So
    what is pinned here is the *distinction* that resolves it - which thread the
    box marshals onto - since a passage restating the three tools without it would
    pass any assertion about the rule id alone.
    """

    def test_the_rule_id_is_not_filtered(self):
        assert "py/catch-base-exception" not in _excluded_rule_ids(), (
            "py/catch-base-exception must not join the filter set. Filtering requires every "
            "instance to be an obliged idiom, and the marshal-onto-a-new-thread case is a "
            "standing counter-example: concurrent.futures does it, so the exclusion would opt "
            "the repository out of a rule that is right about half its own alerts."
        )


#: Trees searched for ``except BaseException`` handlers. The passage's claim is
#: about the whole repository, so the scan is too - a handler added under
#: ``examples/`` is as much a new alert as one under ``strands_robots/``.
_HANDLER_TREES = ("strands_robots", "tests", "tests_integ", "examples", "scripts")

#: The one handler the section is about, as ``path::function``.
_MARSHAL_BOX = "strands_robots/simulation/isaac/simulation.py::_job"

#: Floor for the census, so a scan that stops finding handlers fails loudly
#: instead of reporting a clean tree it never read.
_MIN_HANDLERS = 10


class _Handler(NamedTuple):
    """One ``except BaseException`` handler, keyed the way the table names it."""

    path: str
    lineno: int
    owner: str
    reraises: bool

    @property
    def key(self) -> str:
        return f"{self.path}::{self.owner}"


def _owning_definition(tree: ast.AST, lineno: int) -> str:
    """Name of the innermost def/class containing ``lineno``."""
    best: ast.AST | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            end = node.end_lineno or node.lineno
            if node.lineno <= lineno <= end and (best is None or node.lineno > best.lineno):  # type: ignore[attr-defined]
                best = node
    return getattr(best, "name", "<module>")


@lru_cache(maxsize=1)
def _base_exception_handlers() -> tuple[_Handler, ...]:
    """Every ``except BaseException`` handler in the tree, with its disposition.

    ``reraises`` is whether the handler's last statement is a lexical ``raise``,
    which is exactly what ``py/catch-base-exception`` accepts - so this is the
    census the marshal-box disposition rests on.
    """
    found: list[_Handler] = []
    for tree_name in _HANDLER_TREES:
        root = _REPO_ROOT / tree_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            try:
                parsed = parse_file(path)
            except (OSError, SyntaxError):  # pragma: no cover - unreadable source
                continue
            for node in ast.walk(parsed):
                if not isinstance(node, ast.ExceptHandler) or node.type is None:
                    continue
                named = {n.id for n in ast.walk(node.type) if isinstance(n, ast.Name)}
                if "BaseException" not in named:
                    continue
                found.append(
                    _Handler(
                        path=path.relative_to(_REPO_ROOT).as_posix(),
                        lineno=node.lineno,
                        owner=_owning_definition(parsed, node.lineno),
                        reraises=isinstance(node.body[-1], ast.Raise),
                    )
                )
    return tuple(found)


class TestTheMarshalBoxCensusIsDerivedFromTheTree:
    """The passage above argues from a census of the tree, so the census is measured here.

    Its load-bearing claim is that the rule's *entire* alert surface is one
    construct: every ``except BaseException`` handler re-raises lexically, which
    ``py/catch-base-exception`` accepts, except the cross-thread marshal box. That
    claim is what makes the section's disposition (dismiss the box, delegate
    everything else to ``concurrent.futures``) exhaustive rather than a guess.

    Nothing checked it. The class above grades which *concepts* the passage names -
    the rule id, ``concurrent.futures``, ``run_on_main``, ``SystemExit`` - all of
    which survive a census going stale underneath them, and it had: the passage
    counted seven handlers against a tree holding sixteen, omitted one in
    ``strands_robots/`` outright, and cited five of its seven sites at line numbers
    that had moved, the flagged one by 745 lines. Every assertion here passed
    throughout.

    Two failures matter differently. A row that names nothing real is a stale
    citation and costs a reader a search. A *second* handler that does not re-raise
    is the section becoming wrong: it is a new alert of a rule whose thread gates
    the merge, arriving with no recorded disposition, and the passage would still
    read as if it had one.
    """

    def test_the_scan_reaches_the_tree(self):
        """A census that finds nothing would satisfy every assertion below."""
        handlers = _base_exception_handlers()
        assert len(handlers) >= _MIN_HANDLERS, (
            f"only {len(handlers)} `except BaseException` handlers found across "
            f"{_HANDLER_TREES}; the scan stopped reading the tree, so the census "
            "assertions below hold vacuously"
        )

    def test_exactly_one_handler_does_not_reraise_lexically(self):
        """The claim the section's disposition rests on."""
        loose = [h for h in _base_exception_handlers() if not h.reraises]
        assert len(loose) == 1, (
            "py/catch-base-exception's whole alert surface here is meant to be one "
            f"construct, and the tree now holds {len(loose)}: "
            f"{[f'{h.path}:{h.lineno} in {h.owner}()' for h in loose]}. Each one that does not "
            "end in a lexical raise is a separate alert whose review thread gates the merge, so "
            "either give it a lexical raise (cleanup-and-reraise, the majority form above) or "
            "record its disposition in that section - a passage claiming a single construct "
            "while the tree holds several sends the next contributor to the wrong bullet."
        )

    def test_the_one_that_does_not_is_the_cross_thread_marshal_box(self):
        """Which handler it is decides which bullet applies, so it is pinned by identity."""
        loose = [h for h in _base_exception_handlers() if not h.reraises]
        assert [h.key for h in loose] == [_MARSHAL_BOX], (
            f"the handler that does not re-raise lexically is {[h.key for h in loose]}, not "
            f"{_MARSHAL_BOX}. The section's disposition is specific to a box marshalling onto an "
            "already-running foreign thread - obliged, because concurrent.futures cannot target "
            "one - and reads as advice to dismiss whatever is flagged if the flagged site is "
            "something else."
        )
