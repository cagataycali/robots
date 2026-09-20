"""Every reader of a dataset's ``meta/*.json`` degrades on a file it cannot read.

Four readers promise to answer "unreadable" rather than raise:
:func:`~strands_robots.verify_dataset.verify_dataset` (whose contract is that a
corrupt dataset yields a report), ``run_policy``'s parquet-truth gate (documented
to return a partial result rather than raising), and the two tri-state dataset
probes ``LerobotTrainer.validate`` consults. All four named
``json.JSONDecodeError`` as the way reading can fail.

Reading this file fails in two further ways, and neither is a JSON decode error:
bytes the declared encoding does not describe raise ``UnicodeDecodeError``, and a
number longer than ``sys.get_int_max_str_digits`` raises a plain ``ValueError``.
Both are the truncated / partially-synced metadata these surfaces exist to
survive, and both escaped - taking the whole report, tool payload or preflight
with them. The drift check inside ``verify_dataset`` itself already grades this
file by ``ValueError`` and reports the corruption, so the fix is parity with the
sibling reader 140 lines above the narrowest one.

A document that PARSES and is not a JSON object is the same damage arriving
readable: a partially-synced or foreign file (a bare ``null``, an array, a
scalar) carries no headers at all, so there is nothing to verify against, and
``.get`` on it raises ``AttributeError`` - a type none of these handlers name.
Three of the readers already answer their unknown for it (the parquet
cross-check reads its one header by subscript, so it lands in the ``TypeError``
its handler names; the rollout tool's gate and the judge's reader both grade the
parsed document), and the rest did not: ``verify_dataset`` escaped as a
traceback, taking the whole report with it, and the quantile probe answered
``False`` - the DEFINITE miss that refuses a training run - from a file it could
not use.

Cells are table-driven over (surface x spelling): one behaviour - a file that
cannot be USED is answered, not raised and not mistaken for a verdict - pinned
once per surface.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq

import strands_robots
from strands_robots.dataset_metadata import read_dataset_episode_indices
from strands_robots.tools.episode_judge import _read_info
from strands_robots.tools.run_policy import _read_parquet_truth
from strands_robots.training.lerobot import (
    _dataset_codebase_version,
    _dataset_quantile_stats_present,
)
from strands_robots.verify_dataset import verify_dataset

_HEALTHY: dict[str, Any] = {
    "total_episodes": 2,
    "total_frames": 6,
    "codebase_version": "v3.0",
    "fps": 30,
    "note": "recorded on rig A",
}

# A number this long is a well-formed JSON number that ``json.load`` refuses to
# build an int from (``sys.get_int_max_str_digits`` defaults to 4300), raising a
# plain ``ValueError`` that is not a ``json.JSONDecodeError``.
_OVERLONG_DIGITS = "1" * 4400


def _undecodable() -> bytes:
    """``meta/info.json`` bytes that UTF-8 does not describe.

    The damage sits inside a string value, as a partially-synced or truncated
    file carries it: the document is still well-formed JSON, so only the DECODE
    fails. Written through ``surrogateescape`` because the byte cannot be spelled
    in a ``str`` any other way.
    """
    return json.dumps(_HEALTHY).replace("rig A", "rig \udcff").encode("utf-8", "surrogateescape")


# (label, bytes). A list rather than a dict because the labels are prose, and the
# last entry is the control: a document that merely is not JSON was always
# answered, so it shows the fix widens what "unreadable" covers instead of
# changing what an unreadable file yields.
_UNREADABLE: list[tuple[str, bytes]] = [
    ("bytes utf-8 does not describe", _undecodable()),
    (
        "a number no int can hold",
        json.dumps(_HEALTHY).replace('"total_episodes": 2', f'"total_episodes": {_OVERLONG_DIGITS}').encode(),
    ),
    ("not JSON at all (control)", b"<html>404 Not Found</html>"),
]


# (label, bytes). Well-formed JSON documents that are not objects. ``json.loads``
# builds each one happily, so every reader gets a value and only USING it fails -
# which is why the handlers above cannot catch these. The array-of-object entry is
# the nastiest spelling: it holds the healthy headers, one level too deep.
_NOT_AN_OBJECT: list[tuple[str, bytes]] = [
    ("an empty array", b"[]"),
    ("an array holding the headers", json.dumps([_HEALTHY]).encode()),
    ("a bare null", b"null"),
    ("a bare string", b'"corrupt"'),
    ("a bare number", b"42"),
    ("a bare boolean", b"true"),
]

_NOT_AN_OBJECT_IDS = [label for label, _ in _NOT_AN_OBJECT]


def _dataset(root: Path, *, info: bytes, episode_indices: list[int]) -> Path:
    """Write a LeRobot v3 dataset whose ``meta/info.json`` holds ``info`` verbatim.

    ``meta/stats.json`` gets the same bytes, so one fixture exercises both files
    the training probes read.

    Args:
        root: Dataset root (the dir that will contain ``meta/``).
        info: Raw bytes for ``meta/info.json`` and ``meta/stats.json``.
        episode_indices: ``episode_index`` values for the episodes parquet.

    Returns:
        The dataset root.
    """
    ep_dir = root / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"episode_index": episode_indices, "length": [3] * len(episode_indices)}),
        ep_dir / "episodes_000.parquet",
    )
    (root / "meta" / "info.json").write_bytes(info)
    (root / "meta" / "stats.json").write_bytes(info)
    return root


class TestVerifyDatasetReportsTheCorruptionItLooksFor:
    """The checker written to report a corrupt dataset survives a corrupt dataset."""

    @pytest.mark.parametrize(("label", "payload"), _UNREADABLE, ids=[label for label, _ in _UNREADABLE])
    def test_an_unreadable_info_json_is_a_problem_not_a_traceback(
        self, tmp_path: Path, label: str, payload: bytes
    ) -> None:
        root = _dataset(tmp_path, info=payload, episode_indices=[0, 1])
        report = verify_dataset(root)
        assert report["status"] == "error"
        assert any("meta/info.json" in problem for problem in report["problems"]), report["problems"]

    @pytest.mark.parametrize(("label", "payload"), _NOT_AN_OBJECT, ids=_NOT_AN_OBJECT_IDS)
    def test_an_info_json_that_is_not_an_object_is_a_problem_not_a_traceback(
        self, tmp_path: Path, label: str, payload: bytes
    ) -> None:
        """The checker names the corruption instead of dying on it.

        The problem names what the file holds, because that is what tells a
        reader whether the header drifted or the whole file is the wrong
        document.
        """
        root = _dataset(tmp_path, info=payload, episode_indices=[0, 1])
        report = verify_dataset(root)
        assert report["status"] == "error"
        assert any("not an object" in problem for problem in report["problems"]), report["problems"]
        # No header was readable, so neither declared count is reported as one.
        assert report["info_total_episodes"] is None
        assert report["info_total_frames"] is None

    @pytest.mark.parametrize(("label", "payload"), _NOT_AN_OBJECT, ids=_NOT_AN_OBJECT_IDS)
    def test_the_video_check_reports_the_header_once(self, tmp_path: Path, label: str, payload: bytes) -> None:
        """Two checks read this file; a broken one is named once, not twice.

        The video check resolves its MP4 paths from the same header, so it sees
        the same unusable document - and it is the second reader that raised.
        """
        root = _dataset(tmp_path, info=payload, episode_indices=[0, 1])
        report = verify_dataset(root)
        assert report["video_files_checked"] == 0
        assert len([p for p in report["problems"] if "meta/info.json" in p]) == 1, report["problems"]

    def test_every_other_problem_survives_a_header_that_is_not_an_object(self, tmp_path: Path) -> None:
        """The report keeps the defects found before the header was read.

        This is what the escape cost: the mega-episode finding was already
        recorded when check 3 read the header, and the traceback discarded it.
        """
        root = _dataset(tmp_path, info=b"null", episode_indices=[0])
        report = verify_dataset(root, expected=3)
        assert report["total_episodes"] == 1
        assert any("expected 3 episode(s)" in problem for problem in report["problems"]), report["problems"]
        assert any("not an object" in problem for problem in report["problems"]), report["problems"]

    def test_every_other_problem_survives_an_unreadable_header(self, tmp_path: Path) -> None:
        """The report keeps the defects found before the header was read.

        This is what the escape cost: a mega-episode dataset (one
        ``episode_index`` where several were intended) whose header also carries
        an undecodable byte lost the mega-episode finding too, because the video
        check re-read the header after that finding was recorded.
        """
        root = _dataset(tmp_path, info=_undecodable(), episode_indices=[0])
        report = verify_dataset(root, expected=3)
        assert report["total_episodes"] == 1
        assert any("expected 3 episode(s)" in problem for problem in report["problems"]), report["problems"]
        assert any("meta/info.json" in problem for problem in report["problems"]), report["problems"]
        assert report["video_files_checked"] == 0


class TestReadersAnswerUnreadableRatherThanRaising:
    """The three other readers of these files answer their documented unknown."""

    @pytest.mark.parametrize(("label", "payload"), _UNREADABLE, ids=[label for label, _ in _UNREADABLE])
    def test_parquet_truth_gate_reports_the_file_it_could_not_read(
        self, tmp_path: Path, label: str, payload: bytes
    ) -> None:
        root = _dataset(tmp_path, info=payload, episode_indices=[0, 1])
        truth = _read_parquet_truth(root)
        assert truth["info_present"] is False
        assert truth["info_path"].endswith("info.json")
        assert truth["error"], truth

    @pytest.mark.parametrize(("label", "payload"), _UNREADABLE, ids=[label for label, _ in _UNREADABLE])
    def test_codebase_version_probe_is_unknown(self, tmp_path: Path, label: str, payload: bytes) -> None:
        root = _dataset(tmp_path, info=payload, episode_indices=[0])
        assert _dataset_codebase_version(str(root)) is None

    @pytest.mark.parametrize(("label", "payload"), _UNREADABLE, ids=[label for label, _ in _UNREADABLE])
    def test_quantile_stats_probe_is_unknown_not_a_definite_miss(
        self, tmp_path: Path, label: str, payload: bytes
    ) -> None:
        """Unknown, not ``False``: an unreadable stats file must not read as a miss.

        ``False`` is the DEFINITE miss that refuses a QUANTILES-normalizing
        training run, so answering it from a file nobody could read would refuse
        a dataset on evidence that was never gathered.
        """
        root = _dataset(tmp_path, info=payload, episode_indices=[0])
        assert _dataset_quantile_stats_present(str(root)) is None


class TestReadersAnswerNotAnObjectRatherThanRaising:
    """The same unknown for a document that parses and cannot be used."""

    @pytest.mark.parametrize(("label", "payload"), _NOT_AN_OBJECT, ids=_NOT_AN_OBJECT_IDS)
    def test_parquet_truth_gate_reports_the_document_it_cannot_read_headers_off(
        self, tmp_path: Path, label: str, payload: bytes
    ) -> None:
        root = _dataset(tmp_path, info=payload, episode_indices=[0, 1])
        truth = _read_parquet_truth(root)
        assert truth["info_present"] is False
        assert truth["error"], truth

    @pytest.mark.parametrize(("label", "payload"), _NOT_AN_OBJECT, ids=_NOT_AN_OBJECT_IDS)
    def test_codebase_version_probe_is_unknown(self, tmp_path: Path, label: str, payload: bytes) -> None:
        """Unknown, not a traceback: this probe runs inside ``validate()``.

        Raising here aborted the whole preflight, so a caller learned nothing
        about any other part of the spec.
        """
        root = _dataset(tmp_path, info=payload, episode_indices=[0])
        assert _dataset_codebase_version(str(root)) is None

    @pytest.mark.parametrize(("label", "payload"), _NOT_AN_OBJECT, ids=_NOT_AN_OBJECT_IDS)
    def test_quantile_stats_probe_is_unknown_not_a_definite_miss(
        self, tmp_path: Path, label: str, payload: bytes
    ) -> None:
        """``None``, never ``False``: this one answered a verdict, not a crash.

        ``False`` is the DEFINITE miss that refuses a QUANTILES-normalizing
        training run. The predicate over the stats mapping answers ``False`` for
        a non-mapping, which is correct for a predicate and is a refusal here -
        gathered from a file that carries no per-feature stats to look in.
        """
        root = _dataset(tmp_path, info=payload, episode_indices=[0])
        assert _dataset_quantile_stats_present(str(root)) is None

    @pytest.mark.parametrize(("label", "payload"), _NOT_AN_OBJECT, ids=_NOT_AN_OBJECT_IDS)
    def test_the_readers_that_already_answered_still_do(self, tmp_path: Path, label: str, payload: bytes) -> None:
        """The two controls: the rule was already honoured here, and still is.

        The parquet cross-check keeps the parquet truth and declares no header;
        the judge's reader answers the empty mapping it documents.
        """
        root = _dataset(tmp_path, info=payload, episode_indices=[0, 1])
        info = read_dataset_episode_indices(root)
        assert info["total_episodes"] == 2
        assert info["info_total_episodes"] is None
        assert info["info_problems"] == []
        assert _read_info(root) == {}


class TestAReadableFileStillReads:
    """Widening what "unreadable" covers narrows nothing that was readable."""

    def test_every_reader_still_answers_from_a_healthy_dataset(self, tmp_path: Path) -> None:
        root = _dataset(tmp_path, info=json.dumps(_HEALTHY).encode(), episode_indices=[0, 1])
        report = verify_dataset(root)
        assert report["status"] == "success"
        assert report["problems"] == []
        assert report["info_total_episodes"] == 2
        truth = _read_parquet_truth(root)
        assert truth["info_present"] is True
        assert truth["total_episodes"] == 2
        assert _dataset_codebase_version(str(root)) == "v3.0"
        # A readable stats file that carries no q01/q99 is the DEFINITE miss,
        # which is the answer an unreadable one must not give.
        assert _dataset_quantile_stats_present(str(root)) is False


def _metadata_json_reads() -> list[tuple[str, int, str]]:
    """Every ``try`` in the package that reads a dataset metadata JSON path.

    Located by AST rather than by a hand-kept roster so a new reader of these
    files cannot fall outside the rule: a ``try`` qualifies when its body calls
    ``json.load``/``json.loads`` on a path bound to a name carrying ``info`` or
    ``stats`` (``info_path``, ``info_json_path``, ``stats_path`` - the three
    spellings these modules use).

    Returns:
        ``(module path, line number, handler source)`` per qualifying handler.
    """
    package = Path(strands_robots.__file__).parent
    found: list[tuple[str, int, str]] = []
    for module in sorted(package.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            body = "\n".join(ast.unparse(stmt) for stmt in node.body)
            if not any(call in body for call in ("json.load(", "json.loads(")):
                continue
            if not any(name in body for name in ("info_path", "info_json_path", "stats_path")):
                continue
            for handler in node.handlers:
                found.append(
                    (
                        str(module.relative_to(package.parent)),
                        handler.lineno,
                        ast.unparse(handler.type) if handler.type else "bare",
                    )
                )
    return found


class TestNoMetadataReaderNamesOnlyTheJsonFailure:
    """The rule is checked over the package, so a new reader inherits it."""

    def test_the_scan_finds_the_known_metadata_reads(self) -> None:
        """Guard against a vacuous pass: the four known readers must be found."""
        modules = {module for module, _, _ in _metadata_json_reads()}
        assert {
            "strands_robots/verify_dataset.py",
            "strands_robots/tools/run_policy.py",
            "strands_robots/training/lerobot.py",
        } <= modules, modules

    def test_every_metadata_read_accepts_a_value_error(self) -> None:
        offenders = [
            f"{module}:{lineno} except {handler}"
            for module, lineno, handler in _metadata_json_reads()
            if "ValueError" not in handler
        ]
        assert not offenders, (
            "a reader of a dataset's meta/*.json names a narrower failure than reading it has: " + "; ".join(offenders)
        )
