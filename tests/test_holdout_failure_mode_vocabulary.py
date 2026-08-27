"""The failure-mode vocabulary holds on the holdout side of the calibration too.

``FAILURE_MODES`` is a field domain rather than one surface's convention: the
docs state every field domain "holds in both directions - the writer refuses a
value outside it, and so does the reader", and the tag arrives in three
spellings. Two of them are graded - the ``failure_mode`` argument of
:func:`~strands_robots.episode_labels.annotate_episode`, and a
``judge.failure_mode`` loaded from the sidecar - so the judge's tag is confined
to the vocabulary at both ends.

The third is a holdout entry's ``failure_mode`` in
:func:`~strands_robots.episode_labels.measure_agreement`, and it is the side of
the comparison the vocabulary was not applied to. Because the judge's tag can
only ever be ``None`` or a vocabulary member, a holdout tag outside the
vocabulary can never equal it: it was counted as a disagreement and pulled
``failure_mode_agreement`` BELOW the truth. That is the harm
``TestReadLabelsRefusesAnUnreadableRecord`` refuses one field over, in this same
function, and its wording applies verbatim - "the direction that makes a sound
judge look unsound" - on a measurement whose stated purpose is deciding whether
the judge should be filtering training data at all.

These cells pin the refusal, the two spellings the refusal must mirror, its
placement, and the readings the fix must leave alone.
"""

from __future__ import annotations

import ast
import inspect
import json
from typing import Any

import pytest

from strands_robots import episode_labels
from strands_robots.episode_labels import (
    FAILURE_MODES,
    QUALITY_GRADES,
    annotate_episode,
    labels_path,
    measure_agreement,
    read_labels,
    record_deterministic_verdicts,
)

#: Tags outside ``None | FAILURE_MODES``. Every one is asserted against the
#: vocabulary inside the test, so a tag promoted into the taxonomy stops being
#: graded here without an edit. The string spellings are the ones a human
#: holdout realistically carries - a typo, a case variant, a stray space, a
#: label a reader would read as the same tag - and the non-strings cover the
#: shapes a hand-written or machine-generated holdout produces.
UNUSABLE_TAGS: list[Any] = [
    "jerky",
    "Jerky_Motion",
    "jerky_motion ",
    "jerky-motion",
    "",
    "unknown",
    3,
    True,
    ["jerky_motion"],
    {"mode": "jerky_motion"},
]


def _judged_root(tmp_path: Any) -> Any:
    """A two-episode dataset whose judge tagged episode 1 ``jerky_motion``."""
    root = tmp_path / "dataset"
    root.mkdir()
    record_deterministic_verdicts(
        root,
        [
            {"episode": 0, "success": True, "failure": False, "steps": 120},
            {"episode": 1, "success": True, "failure": False, "steps": 110},
        ],
        benchmark="reach",
    )
    annotate_episode(root, 0, quality="high", failure_mode=None, model="judge")
    annotate_episode(root, 1, quality="high", failure_mode="jerky_motion", model="judge")
    return root


@pytest.fixture
def judged_root(tmp_path: Any) -> Any:
    return _judged_root(tmp_path)


class TestTheVocabularyIsAFieldDomainNotOneSurfacesConvention:
    """The premise: the other two spellings of the tag are already graded.

    Without this the refusal below would read as new strictness invented for
    the holdout. With it, the holdout is the spelling that was missing.
    """

    @pytest.mark.parametrize("tag", UNUSABLE_TAGS, ids=[str(t)[:18].replace(" ", "-") for t in UNUSABLE_TAGS])
    def test_the_annotation_writer_refuses_the_same_tag(self, judged_root: Any, tag: Any) -> None:
        assert tag not in FAILURE_MODES
        with pytest.raises(ValueError, match="failure_mode must be None or one of"):
            annotate_episode(judged_root, 0, quality="high", failure_mode=tag, model="judge")

    @pytest.mark.parametrize("tag", UNUSABLE_TAGS, ids=[str(t)[:18].replace(" ", "-") for t in UNUSABLE_TAGS])
    def test_the_sidecar_reader_refuses_the_same_tag(self, judged_root: Any, tag: Any) -> None:
        assert tag not in FAILURE_MODES
        path = labels_path(judged_root)
        document = json.loads(path.read_text())
        document["episodes"]["1"]["judge"]["failure_mode"] = tag
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="failure_mode"):
            read_labels(judged_root)

    def test_so_the_judges_tag_is_always_comparable(self, judged_root: Any) -> None:
        """Which is why an out-of-vocabulary holdout tag can never match one."""
        judge_tags = {record["judge"]["failure_mode"] for record in read_labels(judged_root)["episodes"].values()}
        assert judge_tags <= {None, *FAILURE_MODES}


class TestAnOutOfVocabularyHoldoutTagIsRefused:
    """The regression: the third spelling is held to the same vocabulary."""

    def test_a_typo_is_not_a_silently_understated_calibration(self, judged_root: Any) -> None:
        """The judge and the human agree; only the human's spelling is wrong.

        This is the reading the measurement is consulted for. The judge tagged
        ``jerky_motion`` and the human meant the same thing, so the truthful
        agreement is 1.0 - which the correctly spelled holdout measures. The
        typo made the same pair of opinions read as total disagreement.
        """
        agreeing = {1: {"quality": "high", "failure_mode": "jerky_motion"}}
        assert measure_agreement(judged_root, agreeing)["failure_mode_agreement"] == 1.0

        assert "jerky" not in FAILURE_MODES
        with pytest.raises(ValueError, match="failure_mode"):
            report = measure_agreement(judged_root, {1: {"quality": "high", "failure_mode": "jerky"}})
            pytest.fail(
                f"a typo was counted as a disagreement: failure_mode_agreement came back "
                f"{report['failure_mode_agreement']:.4f} where the same opinions spelled in the "
                f"vocabulary measure 1.0000, and the report names it as {report['disagreements']} "
                "rather than as an unusable holdout tag."
            )

    @pytest.mark.parametrize("tag", UNUSABLE_TAGS, ids=[str(t)[:18].replace(" ", "-") for t in UNUSABLE_TAGS])
    def test_every_unusable_spelling_is_refused(self, judged_root: Any, tag: Any) -> None:
        assert tag not in FAILURE_MODES
        with pytest.raises(ValueError):
            measure_agreement(judged_root, {1: {"quality": "high", "failure_mode": tag}})

    def test_the_refusal_names_the_entry_the_field_the_vocabulary_and_the_value(self, judged_root: Any) -> None:
        """A calibration report names what to look at; so must its refusal."""
        with pytest.raises(ValueError) as excinfo:
            measure_agreement(judged_root, {1: {"quality": "high", "failure_mode": "jerky"}})
        message = str(excinfo.value)
        assert "measure_agreement" in message
        assert "human_labels[1]" in message
        assert "failure_mode" in message
        assert str(FAILURE_MODES) in message
        assert "'jerky'" in message

    def test_the_refusal_mirrors_the_annotation_writers_wording(self, judged_root: Any) -> None:
        """One quantity, one vocabulary, one way of saying it is outside it.

        Each names the field in its own terms - the argument on one side, the
        holdout entry it came from on the other - and both then say what the
        domain is and what arrived, so a reader who has seen one refusal
        recognises the other.
        """
        with pytest.raises(ValueError) as holdout_error:
            measure_agreement(judged_root, {1: {"quality": "high", "failure_mode": "jerky"}})
        with pytest.raises(ValueError) as writer_error:
            annotate_episode(judged_root, 1, quality="high", failure_mode="jerky", model="judge")
        shared = f"must be None or one of {FAILURE_MODES}, got 'jerky'."
        assert shared in str(holdout_error.value)
        assert shared in str(writer_error.value)
        assert "human_labels[1]['failure_mode']" in str(holdout_error.value)
        assert "annotate_episode: failure_mode" in str(writer_error.value)


class TestTheRefusalDoesNotDependOnWhatTheJudgeGotThrough:
    """Placement: an unusable holdout entry is refused before the lookup.

    The same reason :func:`~strands_robots.episode_labels.read_labels` refuses
    an unreadable record rather than skipping it - a verdict that depends on how
    much of the input could be interpreted is not a verdict.
    """

    def test_an_unusable_tag_on_an_unjudged_episode_is_still_refused(self, tmp_path: Any) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        record_deterministic_verdicts(
            root,
            [{"episode": 0, "success": True}, {"episode": 1, "success": True}],
            benchmark="reach",
        )
        annotate_episode(root, 0, quality="high", failure_mode="drift", model="judge")
        # Episode 1 carries no judge block, so the comparison loop would
        # ``continue`` past it before ever reading its tag.
        assert read_labels(root)["episodes"]["1"].get("judge") is None
        with pytest.raises(ValueError, match="failure_mode"):
            measure_agreement(
                root,
                {0: {"quality": "high", "failure_mode": "drift"}, 1: {"quality": "high", "failure_mode": "jerky"}},
            )


class TestEveryReaderOfTheTagNamesTheVocabulary:
    """Derived, so a fourth reader of the tag is graded when it lands.

    The inventory is read off the module rather than listed, because the whole
    defect was one reader of this field that the vocabulary had not reached.
    """

    @staticmethod
    def _body_source(function: ast.FunctionDef) -> str:
        """The function's statements, with its docstring dropped."""
        body = function.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        return "\n".join(ast.unparse(statement) for statement in body)

    @classmethod
    def _readers_missing_the_vocabulary(cls, source: str) -> list[str]:
        """Functions whose code reads the tag without naming its vocabulary."""
        return [
            function.name
            for function in ast.walk(ast.parse(source))
            if isinstance(function, ast.FunctionDef)
            and "failure_mode" in (body := cls._body_source(function))
            and "FAILURE_MODES" not in body
        ]

    def test_the_shipped_module_has_no_such_reader(self) -> None:
        source = inspect.getsource(episode_labels)
        offenders = self._readers_missing_the_vocabulary(source)
        assert offenders == [], (
            f"these read a failure-mode tag without holding it to FAILURE_MODES: {offenders}. "
            "A tag outside the vocabulary cannot be compared against the judge's, which is "
            "confined to it at both ends."
        )

    def test_the_scan_finds_the_readers_it_is_meant_to_survey(self) -> None:
        """Non-vacuity: the survey reaches every function that reads the tag."""
        source = inspect.getsource(episode_labels)
        readers = {
            function.name
            for function in ast.walk(ast.parse(source))
            if isinstance(function, ast.FunctionDef) and "failure_mode" in self._body_source(function)
        }
        assert {"_record_problem", "annotate_episode", "measure_agreement"} <= readers

    def test_the_rule_separates_a_constructed_reader_from_a_constructed_offender(self) -> None:
        """The scan grades dataflow, not the presence of a docstring word."""
        exemplars = (
            "def holds(mode):\n"
            '    """A failure_mode named only here is not read."""\n'
            "    if mode is not None and mode not in FAILURE_MODES:\n"
            "        raise ValueError(mode)\n"
            "\n"
            "def drops(entry):\n"
            '    """Compares a tag without holding it to the vocabulary."""\n'
            '    return entry.get("failure_mode") == "drift"\n'
        )
        assert self._readers_missing_the_vocabulary(exemplars) == ["drops"]


class TestWhatIsUnchanged:
    """Readings the fix must leave alone. Every one holds before it too."""

    def test_a_holdout_entry_with_no_tag_measures_no_failure_mode_agreement(self, judged_root: Any) -> None:
        report = measure_agreement(judged_root, {1: {"quality": "high"}})
        assert report["failure_mode_agreement"] is None
        assert report["episodes_compared"] == 1
        assert report["disagreements"] == []

    def test_an_explicit_none_is_a_real_opinion_and_still_compares(self, judged_root: Any) -> None:
        """``null`` is in the documented domain: "no failure mode observed"."""
        assert (
            measure_agreement(judged_root, {0: {"quality": "high", "failure_mode": None}})["failure_mode_agreement"]
            == 1.0
        )
        disagreeing = measure_agreement(judged_root, {1: {"quality": "high", "failure_mode": None}})
        assert disagreeing["failure_mode_agreement"] == 0.0
        assert disagreeing["disagreements"] == [
            {"episode": 1, "field": "failure_mode", "judge": "jerky_motion", "human": None}
        ]

    @pytest.mark.parametrize("tag", FAILURE_MODES)
    def test_every_vocabulary_tag_is_a_usable_holdout_tag(self, judged_root: Any, tag: str) -> None:
        report = measure_agreement(judged_root, {1: {"quality": "high", "failure_mode": tag}})
        assert report["failure_mode_agreement"] == (1.0 if tag == "jerky_motion" else 0.0)

    def test_a_mixed_holdout_still_reports_the_fractions_and_the_rows(self, judged_root: Any) -> None:
        report = measure_agreement(
            judged_root,
            {0: {"quality": "high", "failure_mode": None}, 1: {"quality": "low", "failure_mode": "drift"}},
        )
        assert report["episodes_compared"] == 2
        assert report["quality_agreement"] == 0.5
        assert report["failure_mode_agreement"] == 0.5
        assert report["disagreements"] == [
            {"episode": 1, "field": "quality", "judge": "high", "human": "low"},
            {"episode": 1, "field": "failure_mode", "judge": "jerky_motion", "human": "drift"},
        ]

    def test_the_grade_refusal_keeps_its_own_message(self, judged_root: Any) -> None:
        with pytest.raises(ValueError) as excinfo:
            measure_agreement(judged_root, {1: {"quality": "excellent"}})
        assert str(excinfo.value) == (
            f"measure_agreement: human_labels[1] must be a dict with 'quality' in {QUALITY_GRADES}."
        )
