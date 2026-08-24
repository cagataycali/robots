"""Contract pin for where a feature branch can exist, and what refuses it.

``AGENTS.md`` > PR Workflow used to open with "Create feature branch from
``main``", which is not something this repository permits. The ``default``
ruleset's conditions are ``ref_name.include: ["~ALL"]`` rather than the default
branch alone, and its rules include ``creation`` with ``bypass_actors: []``, so
``git push <base> HEAD:refs/heads/<new>`` is refused for every account. The
remaining four steps already assume a fork - step 5 says "Open PR from your
fork" - so the file described a first step the rest of it did not use and the
ruleset does not allow.

What makes it worth writing down rather than leaving to be rediscovered is that
the refusal is *unattributed*. GitHub answers a ruleset violation with
``push declined due to repository rule violations`` and does not name the rule,
which makes it indistinguishable from the two refusals this file does describe:

=============================  =================================  ==============
refusal                        cause                              remedy
=============================  =================================  ==============
``repository rule violations`` the ``creation`` rule, no bypass   push to a fork
``Resource not accessible``    token lacks a permission           widen the token
``mergeStateStatus: BLOCKED``  ``.github/workflows/**`` write     use a PAT
=============================  =================================  ==============

Two of the three are answered by presenting a different token, so that is the
natural next move, and here it cannot work: a ruleset bypass is granted per
ruleset, so no role carries one, and there is no classic branch protection for an
account to be exempt from (``GET .../branches/main/protection`` -> ``404``). The
wrong reading looks exactly like diligence, and it is unbounded - there is no
token to escalate to.

So two classes are asserted here, and the first is what keeps the second honest.

``TestTheCreationRuleIsCheckableOffline`` *executes* the claim. It implements the
derivation the guidance asks a reader to perform - match the ref against
``conditions.ref_name``, look for a ``creation`` rule, check ``bypass_actors`` -
and runs it over the payload this repository publishes, recorded below from
``GET /repos/strands-labs/robots/rulesets/13012156``. A pin that merely asserted
``AGENTS.md`` *says* branch creation is refused would keep passing after a bypass
actor was added, leaving the guidance reading plausibly while sending
contributors to a fork they no longer need. The one-field variants are what
carry that: each drops the refusal, so the pin says *why* creation is refused
rather than only *that* it is, and it answers the question a reader will
actually have - a bypass actor would clear this, a wider token would not.

``TestTheGuidanceNamesTheForkConstraint`` pins the prose, because the prose is
the deliverable: a contributor reads ``AGENTS.md``, not this module. What is
asserted is *adjacency* rather than vocabulary - the fork instruction, the rule
that forces it, the empty bypass list and the cross-repo remedy have to stay in
the same breath, since the instruction alone reads as a house style someone may
tidy back to "create a branch". That is the same structural reason
``tests/test_merge_gate_viewer_scope.py`` and
``tests/test_graphql_node_id_targeting.py`` exist, and these text assertions
follow the shape those modules established.

Negative control: with ``origin/main``'s ``AGENTS.md`` restored, all 7 tests in
``TestTheGuidanceNamesTheForkConstraint`` fail - the four qualifiers, the two
context guards that locate the passage and the slice bound - while all 11 in
``TestTheCreationRuleIsCheckableOffline`` pass unchanged. The ruleset is a
property of the repository rather than of this change; only the guidance is new.

See #1959.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AGENTS_PATH = _REPO_ROOT / "AGENTS.md"

#: The ``default`` branch ruleset as published, from
#: ``GET /repos/strands-labs/robots/rulesets/13012156`` (created 2026-02-19,
#: last updated 2026-06-30). Rule ``parameters`` are omitted: the ``creation``
#: rule takes none, and the derivation below reads only the rule ``type``. The
#: sibling rule types are kept because "which rule refuses this" is precisely
#: what the refusal message does not say.
_PUBLISHED_RULESET: dict[str, Any] = {
    "id": 13012156,
    "name": "default",
    "target": "branch",
    "enforcement": "active",
    "conditions": {"ref_name": {"include": ["~ALL"], "exclude": []}},
    "bypass_actors": [],
    "rules": [
        {"type": "deletion"},
        {"type": "pull_request"},
        {"type": "non_fast_forward"},
        {"type": "required_status_checks"},
        {"type": "creation"},
    ],
}

#: A ref a contributor would create for a change like this one.
_NEW_BRANCH = "refs/heads/docs/branch-creation-is-forked"


def _ref_is_matched(conditions: dict[str, Any], ref_name: str, default_branch: str) -> bool:
    """Whether ``ref_name`` falls inside a ruleset's ``ref_name`` conditions.

    Implements the subset the guidance rests on - the ``~ALL`` and
    ``~DEFAULT_BRANCH`` selectors plus fnmatch patterns, with ``exclude``
    winning over ``include`` - rather than the whole ruleset condition language.
    """
    ref_conditions = conditions.get("ref_name", {})

    def matched(patterns: list[str]) -> bool:
        for pattern in patterns:
            if pattern == "~ALL":
                return True
            if pattern == "~DEFAULT_BRANCH":
                if ref_name == f"refs/heads/{default_branch}":
                    return True
                continue
            if fnmatch.fnmatchcase(ref_name, pattern):
                return True
        return False

    if not matched(ref_conditions.get("include", [])):
        return False
    return not matched(ref_conditions.get("exclude", []))


def _refuses_ref_creation(ruleset: dict[str, Any], ref_name: str, default_branch: str = "main") -> bool:
    """Whether ``ruleset`` refuses creating ``ref_name``, for every account.

    This is the check ``AGENTS.md`` step 1 asks a reader to run against the
    payload in hand instead of retrying with a wider token: an active ruleset
    that matches the ref, carries a ``creation`` rule and lists no bypass actor
    refuses the push regardless of who makes it.
    """
    if ruleset.get("enforcement") != "active":
        return False
    if ruleset.get("target", "branch") != "branch":
        return False
    if not _ref_is_matched(ruleset.get("conditions", {}), ref_name, default_branch):
        return False
    if not any(rule.get("type") == "creation" for rule in ruleset.get("rules", [])):
        return False
    return not ruleset.get("bypass_actors")


def _without_rule(ruleset: dict[str, Any], rule_type: str) -> dict[str, Any]:
    return {**ruleset, "rules": [r for r in ruleset["rules"] if r.get("type") != rule_type]}


#: Each entry is a single-field departure from the published payload that drops
#: the refusal. Together they say which field is load-bearing, and answer the
#: two questions a reader arrives with: would a bypass actor clear this (yes),
#: and is some other rule responsible (no).
_ONE_FIELD_VARIANTS: list[tuple[str, dict[str, Any]]] = [
    (
        "a bypass actor is listed",
        {**_PUBLISHED_RULESET, "bypass_actors": [{"actor_id": 5, "actor_type": "RepositoryRole"}]},
    ),
    ("the creation rule is removed", _without_rule(_PUBLISHED_RULESET, "creation")),
    ("enforcement is disabled", {**_PUBLISHED_RULESET, "enforcement": "disabled"}),
    (
        "the conditions name the default branch only",
        {**_PUBLISHED_RULESET, "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}}},
    ),
    (
        "the ref is excluded",
        {
            **_PUBLISHED_RULESET,
            "conditions": {"ref_name": {"include": ["~ALL"], "exclude": ["refs/heads/docs/**"]}},
        },
    ),
]


class TestTheCreationRuleIsCheckableOffline:
    """The refusal is derivable from the published payload, not a guess."""

    def test_the_published_ruleset_refuses_a_new_branch(self) -> None:
        assert _refuses_ref_creation(_PUBLISHED_RULESET, _NEW_BRANCH) is True

    def test_the_scope_is_every_ref_rather_than_the_default_branch(self) -> None:
        # The reason the rule bites at all: a branch ruleset is easy to read as
        # guarding `main`, and this one names every ref.
        conditions = _PUBLISHED_RULESET["conditions"]
        assert conditions["ref_name"]["include"] == ["~ALL"]
        assert _ref_is_matched(conditions, _NEW_BRANCH, "main") is True

    def test_the_default_branch_is_matched_too(self) -> None:
        # So the rule is not "every ref except the one you care about" - there is
        # no ref in this repository that may be created.
        assert _refuses_ref_creation(_PUBLISHED_RULESET, "refs/heads/main") is True

    def test_no_actor_is_listed_as_a_bypass(self) -> None:
        # The whole basis for "no token clears it". A permissions answer requires
        # somebody to escalate to, and this list is where they would be named.
        assert _PUBLISHED_RULESET["bypass_actors"] == []

    @pytest.mark.parametrize(
        "ruleset",
        [pytest.param(variant, id=name) for name, variant in _ONE_FIELD_VARIANTS],
    )
    def test_one_field_variants_drop_the_refusal(self, ruleset: dict[str, Any]) -> None:
        assert _refuses_ref_creation(ruleset, _NEW_BRANCH) is False

    def test_the_sibling_rules_alone_do_not_refuse_creation(self) -> None:
        # `pull_request` and `non_fast_forward` are the rules AGENTS.md already
        # documents, and blaming either of them for this refusal sends the reader
        # to the review settings. Only `creation` refuses a create.
        others = _without_rule(_PUBLISHED_RULESET, "creation")
        assert {r["type"] for r in others["rules"]} == {
            "deletion",
            "pull_request",
            "non_fast_forward",
            "required_status_checks",
        }
        assert _refuses_ref_creation(others, _NEW_BRANCH) is False

    def test_the_recorded_payload_carries_the_fields_the_derivation_reads(self) -> None:
        # Non-vacuity: an abbreviated fixture that had lost `conditions` or
        # `rules` would make every assertion above trivially true.
        assert set(_PUBLISHED_RULESET) >= {"target", "enforcement", "conditions", "rules", "bypass_actors"}
        assert "creation" in {r["type"] for r in _PUBLISHED_RULESET["rules"]}


def _agents_text() -> str:
    return _AGENTS_PATH.read_text(encoding="utf-8")


#: The instruction the correction introduces. Every other assertion is
#: positioned from it, so its absence fails outright rather than making the rest
#: vacuous.
_FORK_INSTRUCTION = "Create the feature branch **on your fork**"

#: Where step 1 ends: the next top-level item of the PR Workflow list.
_NEXT_STEP = "\n2. "


def _step_one(text: str) -> str | None:
    """PR Workflow step 1, from its instruction to the start of step 2.

    Bounding the window at the list boundary rather than at a character count is
    what makes "in the same step" the assertion. A qualifier reworded down into
    step 2 leaves the slice and fails; a qualifier reworded within step 1 stays
    inside it however much the step grows, so the pin does not have to be
    retuned every time the prose moves.
    """
    start = text.find(_FORK_INSTRUCTION)
    if start < 0:
        return None
    end = text.find(_NEXT_STEP, start)
    if end < 0:
        return None
    return text[start:end]


class TestTheGuidanceNamesTheForkConstraint:
    """The instruction is only actionable with the rule that forces it beside it."""

    def test_the_fork_instruction_is_present(self) -> None:
        # Context guard: the assertions below are positioned from this phrase, so
        # a silent rewording would move the pin rather than break it.
        assert _FORK_INSTRUCTION in _agents_text(), (
            f"AGENTS.md no longer contains {_FORK_INSTRUCTION!r}, which this class uses to "
            "locate PR Workflow step 1. If the instruction was deliberately reworded, "
            "update _FORK_INSTRUCTION to match rather than deleting these tests - the "
            "point is that the instruction and the rule that forces it stay together."
        )

    def test_the_base_repository_is_not_offered_as_an_option(self) -> None:
        assert "1. Create feature branch from `main`" not in _agents_text(), (
            "PR Workflow step 1 must not tell a contributor to branch in the base "
            "repository: the `default` ruleset refuses ref creation there for every "
            "account. See #1959."
        )

    def test_the_step_one_slice_is_bounded_by_the_list_boundary(self) -> None:
        # Non-vacuity: the qualifier assertions read a slice, so a slice that had
        # collapsed to nothing - or swallowed the rest of the file - would make
        # them pass or fail for the wrong reason.
        window = _step_one(_agents_text())
        assert window is not None
        assert len(window) > 800, f"step 1 slice is only {len(window)} chars"
        assert "2. Make changes, run" not in window, "the slice leaked into step 2"

    def test_the_guidance_names_the_rule_that_refuses_the_push(self) -> None:
        window = _step_one(_agents_text())
        assert window is not None and "creation" in window, (
            "AGENTS.md must name the `creation` rule beside the fork instruction. GitHub "
            "answers with 'repository rule violations' and names no rule, so without this "
            "the reader cannot tell the refusal from a missing permission. See #1959."
        )

    def test_the_guidance_states_that_no_account_clears_it(self) -> None:
        window = _step_one(_agents_text())
        assert window is not None and "bypass_actors" in window, (
            "AGENTS.md must say the bypass list is empty. 'Push to a fork' reads as a "
            "house style that a maintainer may assume does not apply to them; the empty "
            "bypass list is why it applies to everyone. See #1959."
        )

    def test_the_guidance_names_the_scope_of_the_rule(self) -> None:
        window = _step_one(_agents_text())
        assert window is not None and "~ALL" in window, (
            "AGENTS.md must say the ruleset targets every ref rather than the default "
            "branch. A branch ruleset reads as guarding `main`, which is why a "
            "contributor expects a feature branch to be exempt. See #1959."
        )

    def test_the_guidance_names_the_cross_repo_remedy(self) -> None:
        window = _step_one(_agents_text())
        assert window is not None and "headRepositoryId" in window, (
            "AGENTS.md must keep the cross-repo remedy beside the constraint: the pull "
            "request names the base repository and the fork as separate inputs "
            "(`repositoryId` and `headRepositoryId`). Naming the constraint without the "
            "remedy leaves the reader blocked. See #1959."
        )
