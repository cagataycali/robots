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
- ``AGENTS.md`` states the gate the same way the workflow does. It carried the
  same false sentence, and survived the correction for a structural reason: #1810
  fixed and pinned only the workflow's copy, so nothing failed while the file every
  contributor reads first still said the opposite. A claim with two homes needs the
  pin to cover both, which is why this assertion lives here beside the workflow's
  rather than in a module of its own;
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

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPO_ROOT / ".github" / "codeql" / "codeql-config.yml"
_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "codeql.yml"
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"
_AGENTS_PATH = _REPO_ROOT / "AGENTS.md"

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
            "codeql.yml must pass config-file, or the filters above are dead text: an "
            "unreferenced CodeQL config silently filters nothing and every alert keeps gating."
        )

    def test_the_workflow_no_longer_claims_alerts_do_not_block(self):
        """The comment that was false is the reason #1810 was filed."""
        workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
        assert "PRs are not blocked on" not in workflow, (
            "codeql.yml used to state that PRs are not blocked on CodeQL alerts. Thread-resolution "
            "on bot-authored review threads makes every new alert a merge gate, so that sentence "
            "described a policy the repository does not implement. Do not restore it."
        )
        assert "hard merge gate" in workflow, (
            "codeql.yml must say what actually happens, not merely stop saying the wrong thing. A "
            "contributor reading it needs to know an alert blocks the merge before they spend a "
            "round wondering why an approved, green PR will not go in."
        )


class TestTheRulesFileStatesTheGate:
    """``AGENTS.md`` is where a contributor looks before they read any workflow.

    #1810 corrected ``codeql.yml`` and pinned it above, but the identical false
    sentence in ``AGENTS.md`` went unpinned and survived - so the file that frames
    every contribution told the reader an alert is advisory while the ruleset
    blocked the merge on it. #1890 is the shape #1892 recorded: approved, required
    check green, one unresolved note-severity thread - free to merge by that file's
    account, and in fact not merging for 53 minutes, until the thread was resolved.

    A negative assertion alone would not hold this - the claim can be restated in
    new words and still be false - so it is paired with the two positives that make
    the section actionable: the ruleset property that does the gating, and the
    dismissal path that clears a deliberate finding without editing the code.
    """

    def test_it_no_longer_claims_a_finding_does_not_block(self):
        text = _AGENTS_PATH.read_text(encoding="utf-8")
        assert "not PR-blocking" not in text, (
            "AGENTS.md used to state that CodeQL findings are not PR-blocking. A "
            "`github-advanced-security` review thread on an alert is a merge gate under "
            "`required_review_thread_resolution`, whatever the severity, so that sentence "
            "described a policy the repository does not implement. Do not restore it - and see "
            "the workflow assertion above, which is the same claim in its other home."
        )

    def test_it_names_the_ruleset_property_that_gates(self):
        text = _AGENTS_PATH.read_text(encoding="utf-8")
        assert "required_review_thread_resolution" in text, (
            "AGENTS.md must name what actually blocks the merge, not merely stop denying it. A "
            "contributor who knows only that 'CodeQL is advisory' - true of the check context, "
            "which is not in the required set - has no account of an approved, green PR that "
            "will not go in, and spends the round #1892 was filed to prevent."
        )

    def test_it_records_the_dismissal_path(self):
        text = _AGENTS_PATH.read_text(encoding="utf-8")
        assert "dismissed_reason" in text, (
            "AGENTS.md must keep naming dismissal as the way to clear a deliberate, test-only "
            "finding. Without it the only visible options are editing the flagged code - which "
            "cost #1879 a round, and on #1890 would have left a fixture asserting nothing, since "
            "the IndexError the query asks for is what CPython clears to end iteration - or "
            "widening the filter set the test above pins at two."
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
