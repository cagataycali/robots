"""Pins the described-change key against the duplicate pairs the created-path key misses.

``scripts/check_duplicate_claim.py`` pairs open branches on what they *create* --
a path, or the slug of a changelog fragment. That key reads a **name**, and a name
is exactly what two authors describing one change need not share. #2820 and #2822
fixed one defect thirteen minutes apart and named it
``feetech-broadcast-is-not-a-reply-address`` and
``feetech-motor-id-excludes-the-broadcast``; two names, no collision, and
``--all-open`` reported ``unique-additions`` while both were open. Issue #2823 is
that instance.

So there is a second key: two branches whose fragments share at least
:data:`FRAGMENT_TOKEN_FLOOR` **words** and which both edit one pre-existing test.

:data:`_ECHO_PAIRS` fixes four measured pairs in place -- every path is the real
one, taken from ``pulls/<n>/files``, which outlives the state change that closed
one half of each. Each is a pair whose closed half names, in its own closing
comment, the pull request that superseded it.

Three pins carry design decisions rather than behaviour:

``TestTheCreatedPathKeyCannotSeeThesePairs``
    The non-vacuity half, and the reason this is a second key rather than a
    tightening of the first. Every pair here reports ``unique-additions`` when only
    what the branches create is read. Without this, a test that the sweep reports
    these pairs would pass on the first key alone.

``TestBothHalvesOfTheConjunctionAreLoadBearing``
    Shared words alone selects 33 of the 2199 co-open pairs in #2345..#2825 and
    fires on 37.2% of replayed sweeps, because the repository names its changes in
    a house style -- ``names`` and ``the`` appear in 39 and 20 of the window's 401
    fragments. A shared edited test alone selects 26. Together they select 14, of
    which 9 are declared duplicates. So neither half may be dropped, and both
    directions are asserted.

``TestTheTwoKeysReportApart``
    The keys support different conclusions and are not merged into one list. A
    shared created path is a fact -- neither file exists on the base, so one of the
    two branches is redundant. A shared description over a shared test is a
    question: it is 64.3% precise, so roughly one pair in three shares a subject
    without sharing a change. The report says which of the two it found, and the
    created-path reasoning must not be attached to an echo.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_duplicate_claim.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_duplicate_claim", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _load()


def _pull(created: tuple[str, ...] = (), edited: tuple[str, ...] = ()) -> Any:
    return check.PullFiles(created=tuple(sorted(created)), edited=tuple(sorted(edited)))


#: ``(label, {number: (created, edited)}, the words shared, the tests shared)``.
#: Real paths, from ``pulls/<n>/files``. The closed half of each pair names its
#: supersedor in its own closing comment: #2373 -> #2370, #2383 -> #2384,
#: #2431 -> #2429, #2822 -> #2820.
_ECHO_PAIRS: list[tuple[str, dict[int, tuple[tuple[str, ...], tuple[str, ...]]], tuple[str, ...], tuple[str, ...]]] = [
    (
        "#2370/#2373 - carry a latched applied force across a scene rebuild",
        {
            2370: (
                (
                    "changelog.d/2370-scene-rebuild-carries-latched-wrench.md",
                    "tests/simulation/mujoco/test_scene_rebuild_carries_latched_wrench.py",
                ),
                ("tests/simulation/mujoco/test_scene_ops_guardrails.py",),
            ),
            2373: (
                (
                    "changelog.d/2373-scene-rebuild-preserves-applied-forces.md",
                    "tests/simulation/mujoco/test_scene_rebuild_preserves_applied_forces.py",
                ),
                ("tests/simulation/mujoco/test_scene_ops_guardrails.py",),
            ),
        },
        ("rebuild", "scene"),
        ("tests/simulation/mujoco/test_scene_ops_guardrails.py",),
    ),
    (
        "#2383/#2384 - the actuator command a keyframe pairs with its pose",
        {
            2383: (
                ("changelog.d/2383-keyframe-ctrl-holds-the-keyed-pose.md",),
                ("tests/simulation/mujoco/test_add_robot_keyframe.py",),
            ),
            2384: (
                (
                    "changelog.d/2384-keyframe-applies-the-command-that-holds-its-pose.md",
                    "tests/simulation/mujoco/test_keyframe_restores_the_command_that_holds_it.py",
                ),
                ("tests/simulation/mujoco/test_add_robot_keyframe.py",),
            ),
        },
        ("holds", "keyframe", "pose", "the"),
        ("tests/simulation/mujoco/test_add_robot_keyframe.py",),
    ),
    (
        "#2429/#2431 - the shape contract EmpiricalNormalization refuses",
        {
            2429: (
                ("changelog.d/2429-normalization-batch-shape-contract.md",),
                ("tests/training/test_rl_normalization.py",),
            ),
            2431: (
                ("changelog.d/2431-rl-normalizer-shape-contract.md",),
                ("tests/training/test_rl_normalization.py",),
            ),
        },
        ("contract", "shape"),
        ("tests/training/test_rl_normalization.py",),
    ),
    (
        "#2820/#2822 - refuse the Feetech broadcast as a single servo's ID",
        {
            2820: (
                (
                    "changelog.d/2820-feetech-broadcast-is-not-a-reply-address.md",
                    "tests/tools/test_feetech_broadcast_is_not_a_reply_address.py",
                ),
                ("tests/tools/test_serial_tool_numeric_domain.py",),
            ),
            2822: (
                (
                    "changelog.d/2822-feetech-motor-id-excludes-the-broadcast.md",
                    "tests/tools/test_feetech_motor_id_excludes_the_broadcast.py",
                ),
                ("tests/tools/test_serial_tool_numeric_domain.py",),
            ),
        },
        ("broadcast", "feetech"),
        ("tests/tools/test_serial_tool_numeric_domain.py",),
    ),
]

_IDS = [row[0] for row in _ECHO_PAIRS]


def _open_set(rows: dict[int, tuple[tuple[str, ...], tuple[str, ...]]]) -> dict[int, Any]:
    return {number: _pull(created, edited) for number, (created, edited) in rows.items()}


class TestTheMeasuredEchoPairsAreReported:
    """Each measured pair is the finding, and names what both branches said and edited."""

    @pytest.mark.parametrize(("label", "rows", "words", "tests"), _ECHO_PAIRS, ids=_IDS)
    def test_the_pair_is_the_finding(
        self, label: str, rows: dict[int, Any], words: tuple[str, ...], tests: tuple[str, ...]
    ) -> None:
        verdict = check.classify_additions(_open_set(rows))
        assert verdict.outcome == check.DUPLICATE_ADDITION, label
        assert verdict.is_finding, label

    @pytest.mark.parametrize(("label", "rows", "words", "tests"), _ECHO_PAIRS, ids=_IDS)
    def test_the_shared_words_and_tests_are_reported(
        self, label: str, rows: dict[int, Any], words: tuple[str, ...], tests: tuple[str, ...]
    ) -> None:
        left, right = sorted(rows)
        assert check.classify_additions(_open_set(rows)).echoes == ((left, right, words, tests),), label

    @pytest.mark.parametrize(("label", "rows", "words", "tests"), _ECHO_PAIRS, ids=_IDS)
    def test_both_pull_requests_are_named(
        self, label: str, rows: dict[int, Any], words: tuple[str, ...], tests: tuple[str, ...]
    ) -> None:
        verdict = check.classify_additions(_open_set(rows))
        assert verdict.implicated == tuple(sorted(rows)), label
        for number in rows:
            assert f"#{number}" in verdict.summary, label

    @pytest.mark.parametrize(("label", "rows", "words", "tests"), _ECHO_PAIRS, ids=_IDS)
    def test_no_reported_word_is_invented(
        self, label: str, rows: dict[int, Any], words: tuple[str, ...], tests: tuple[str, ...]
    ) -> None:
        """Every word reported is one both branches' fragments really use.

        The report must not name a word neither author wrote, for the reason the
        created-path key must not name a file that exists on neither branch.
        """
        files = _open_set(rows)
        for left, right, reported, shared_tests in check.classify_additions(files).echoes:
            for word in reported:
                for number in (left, right):
                    assert word in check.fragment_tokens(files[number].created), f"{label}: #{number}"
            for path in shared_tests:
                assert path in files[left].edited and path in files[right].edited, label


class TestTheCreatedPathKeyCannotSeeThesePairs:
    """The non-vacuity half: this is a second key, not a tightening of the first."""

    @pytest.mark.parametrize(("label", "rows", "words", "tests"), _ECHO_PAIRS, ids=_IDS)
    def test_reading_only_what_the_branches_create_reports_nothing(
        self, label: str, rows: dict[int, Any], words: tuple[str, ...], tests: tuple[str, ...]
    ) -> None:
        created = {number: pull.created for number, pull in _open_set(rows).items()}
        assert check.find_addition_collisions(created) == (), label

    @pytest.mark.parametrize(("label", "rows", "words", "tests"), _ECHO_PAIRS, ids=_IDS)
    def test_the_two_branches_name_their_change_differently(
        self, label: str, rows: dict[int, Any], words: tuple[str, ...], tests: tuple[str, ...]
    ) -> None:
        """Which is why the exact-slug key misses them, and is the premise of this file."""
        left, right = sorted(rows)
        files = _open_set(rows)
        keys = [{check.addition_key(path) for path in files[number].created} for number in (left, right)]
        assert not (keys[0] & keys[1]), label


class TestBothHalvesOfTheConjunctionAreLoadBearing:
    """Neither shared words nor a shared test is usable on its own."""

    def test_one_shared_word_is_not_enough(self) -> None:
        files = {
            11: _pull(("changelog.d/11-the-refusal-names-its-cause.md",), ("tests/test_a.py",)),
            12: _pull(("changelog.d/12-the-camera-reports-a-rate.md",), ("tests/test_a.py",)),
        }
        assert check.fragment_tokens(files[11].created) & check.fragment_tokens(files[12].created) == {"the"}
        assert check.classify_additions(files).outcome == check.UNIQUE_ADDITIONS

    def test_shared_words_without_a_shared_test_is_not_enough(self) -> None:
        files = {
            11: _pull(("changelog.d/11-feetech-broadcast-refused.md",), ("tests/test_a.py",)),
            12: _pull(("changelog.d/12-feetech-broadcast-excluded.md",), ("tests/test_b.py",)),
        }
        assert check.classify_additions(files).outcome == check.UNIQUE_ADDITIONS

    def test_a_shared_edited_source_file_alone_is_not_enough(self) -> None:
        """The widening #2823 proposed, measured at 8.8% precision over #2345..#2825.

        Two branches editing one source file is the sibling sweep's composition
        question. It is read here only in conjunction with a shared description.
        """
        files = {
            11: _pull(("changelog.d/11-one-change.md",), ("strands_robots/utils.py",)),
            12: _pull(("changelog.d/12-another-change.md",), ("strands_robots/utils.py",)),
        }
        assert check.classify_additions(files).outcome == check.UNIQUE_ADDITIONS

    def test_the_floor_is_two_words(self) -> None:
        assert check.FRAGMENT_TOKEN_FLOOR == 2

    def test_a_shared_edited_file_outside_the_test_tree_is_not_the_shared_test(self) -> None:
        files = {
            11: _pull(("changelog.d/11-feetech-broadcast-refused.md",), ("strands_robots/tools/serial_tool.py",)),
            12: _pull(("changelog.d/12-feetech-broadcast-excluded.md",), ("strands_robots/tools/serial_tool.py",)),
        }
        assert check.classify_additions(files).outcome == check.UNIQUE_ADDITIONS

    def test_a_created_test_is_not_an_edited_one(self) -> None:
        """Both branches writing a new test is the created-path key's question.

        This half asks whether both corrected the *same existing* case, which is
        what two authors fixing one defect do.
        """
        files = {
            11: _pull(("changelog.d/11-feetech-broadcast-refused.md", "tests/test_shared.py"), ()),
            12: _pull(("changelog.d/12-feetech-broadcast-excluded.md", "tests/test_shared.py"), ()),
        }
        verdict = check.classify_additions(files)
        assert verdict.echoes == ()
        assert verdict.collisions == ((11, 12, ("tests/test_shared.py",)),)


class TestTheTwoKeysReportApart:
    """A created path is a fact; a described change is a pair to read."""

    @staticmethod
    def _echo_report() -> str:
        _, rows, _, _ = _ECHO_PAIRS[3]
        return check.render_additions(check.classify_additions(_open_set(rows)), "strands-labs/robots")

    def test_an_echo_lands_in_echoes_and_not_in_collisions(self) -> None:
        _, rows, _, _ = _ECHO_PAIRS[3]
        verdict = check.classify_additions(_open_set(rows))
        assert verdict.collisions == ()
        assert len(verdict.echoes) == 1

    def test_the_echo_section_names_the_words_and_the_test(self) -> None:
        report = self._echo_report()
        assert "#2820 + #2822" in report
        assert "feetech" in report and "broadcast" in report
        assert "tests/tools/test_serial_tool_numeric_domain.py" in report

    def test_the_echo_section_says_it_is_the_weaker_key(self) -> None:
        report = self._echo_report()
        assert "one change described twice" in report
        assert "weaker of the two keys" in report
        assert "64.3%" in report

    def test_an_echo_does_not_claim_the_created_path_reasoning(self) -> None:
        """ "Neither branch's file exists on the base" is false of an edited test."""
        assert "not a merge order to decide" not in self._echo_report()

    def test_a_created_path_pair_keeps_its_own_reasoning(self) -> None:
        files = {n: _pull(("tests/test_a.py",), ()) for n in (11, 12)}
        report = check.render_additions(check.classify_additions(files), "owner/name")
        assert "not a merge order to decide" in report
        assert "one change described twice" not in report

    def test_both_keys_firing_reports_both_sections(self) -> None:
        files = {
            11: _pull(("changelog.d/11-feetech-broadcast-refused.md", "tests/test_new.py"), ("tests/test_old.py",)),
            12: _pull(("changelog.d/12-feetech-broadcast-excluded.md", "tests/test_new.py"), ("tests/test_old.py",)),
        }
        verdict = check.classify_additions(files)
        assert verdict.collisions and verdict.echoes
        report = check.render_additions(verdict, "owner/name")
        assert "not a merge order to decide" in report
        assert "one change described twice" in report

    def test_the_clean_summary_names_the_second_key(self) -> None:
        files = {n: _pull((f"changelog.d/{n}-unrelated-{n}.md",), ()) for n in (11, 12)}
        verdict = check.classify_additions(files)
        assert verdict.outcome == check.UNIQUE_ADDITIONS
        assert "describe one change over one test" in verdict.summary


class TestWhatContributesAWord:
    """Only a fragment the assembler would accept, which is the safe direction."""

    @pytest.mark.parametrize(
        "path",
        [
            "tests/test_x.py",
            "strands_robots/utils.py",
            "changelog.d/README.md",
            "changelog.d/no-leading-number.md",
            "changelog.d/2765-Not_A_Slug.md",
            "changelog.d/nested/2765-slug.md",
            # ``docs/robots/`` is exactly as long as ``changelog.d/``, so this
            # path's tail is a well-formed fragment name. The directory has to be
            # checked; the name pattern alone would read words out of it.
            "docs/robots/2820-feetech-broadcast.md",
        ],
    )
    def test_anything_that_is_not_a_fragment_contributes_no_word(self, path: str) -> None:
        assert check.fragment_tokens((path,)) == frozenset()

    def test_a_fragment_contributes_its_slug_words_and_not_its_number(self) -> None:
        assert check.fragment_tokens(("changelog.d/2820-feetech-broadcast-refused.md",)) == {
            "feetech",
            "broadcast",
            "refused",
        }

    def test_a_reserved_name_contributes_nothing_even_if_it_looks_like_a_fragment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The assembler's list is the authority, not the pattern's happy accident."""
        monkeypatch.setattr(check._ASSEMBLER, "RESERVED_NAMES", frozenset({"0-release-notes.md"}))
        assert check.fragment_tokens(("changelog.d/0-release-notes.md",)) == frozenset()

    def test_several_fragments_contribute_every_slug(self) -> None:
        assert check.fragment_tokens(("changelog.d/11-one-thing.md", "changelog.d/12-other-thing.md")) == {
            "one",
            "other",
            "thing",
        }


class TestDeterminism:
    """Stable in all four axes, so a diff of two reports shows changed verdicts."""

    def test_pairs_words_and_tests_are_sorted(self) -> None:
        files = {
            99: _pull(("changelog.d/99-zebra-apple-mango.md",), ("tests/test_z.py", "tests/test_a.py")),
            11: _pull(("changelog.d/11-mango-apple-zebra.md",), ("tests/test_z.py", "tests/test_a.py")),
        }
        assert check.find_echo_collisions(files) == (
            (11, 99, ("apple", "mango", "zebra"), ("tests/test_a.py", "tests/test_z.py")),
        )

    def test_three_branches_describing_one_change_report_all_three_pairs(self) -> None:
        files = {
            n: _pull((f"changelog.d/{n}-feetech-broadcast-refused.md",), ("tests/test_a.py",)) for n in (11, 12, 13)
        }
        assert [(left, right) for left, right, _, _ in check.find_echo_collisions(files)] == [
            (11, 12),
            (11, 13),
            (12, 13),
        ]
        assert check.classify_additions(files).implicated == (11, 12, 13)

    def test_an_empty_open_set_finds_no_echo(self) -> None:
        assert check.find_echo_collisions({}) == ()

    def test_an_unreadable_open_set_is_not_an_echo_finding(self) -> None:
        verdict = check.classify_additions(None, "the API returned errors.")
        assert verdict.outcome == check.UNKNOWN_ADDITIONS
        assert verdict.echoes == ()


class TestTheNodeReaderSplitsOneFileListByChangeType:
    """Both keys read one response, and a truncated list is refused once for both."""

    @staticmethod
    def _node(number: int, files: list[dict[str, str]], total: int | None = None) -> dict[str, Any]:
        return {
            "number": number,
            "files": {"totalCount": len(files) if total is None else total, "nodes": files},
        }

    def test_created_and_edited_are_read_from_one_list(self) -> None:
        node = self._node(
            11,
            [
                {"path": "b.py", "changeType": check.ADDED_CHANGE_TYPE},
                {"path": "a.py", "changeType": check.ADDED_CHANGE_TYPE},
                {"path": "d.py", "changeType": check.EDITED_CHANGE_TYPE},
                {"path": "c.py", "changeType": check.EDITED_CHANGE_TYPE},
            ],
        )
        assert check.file_sets(node) == check.PullFiles(created=("a.py", "b.py"), edited=("c.py", "d.py"))

    @pytest.mark.parametrize("change_type", ["REMOVED", "RENAMED", "COPIED", "CHANGED"])
    def test_no_other_change_type_reaches_either_key(self, change_type: str) -> None:
        node = self._node(11, [{"path": "a.py", "changeType": change_type}])
        assert check.file_sets(node) == check.PullFiles()

    def test_a_truncated_list_is_refused_rather_than_read_short_for_either_key(self) -> None:
        node = self._node(11, [{"path": "a.py", "changeType": check.EDITED_CHANGE_TYPE}], total=101)
        with pytest.raises(check.ClaimSetUnreadable, match="truncated"):
            check.file_sets(node)
