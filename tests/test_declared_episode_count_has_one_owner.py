"""The episode count a dataset's ``meta/info.json`` declares has ONE verdict.

Four surfaces read ``meta/info.json``'s ``total_episodes`` header: the parquet
cross-check in :func:`~strands_robots.dataset_recorder.read_dataset_episode_indices`,
the drift check in :func:`~strands_robots.verify_dataset.verify_dataset`, the
validation-split denominator in ``strands_robots.training.lerobot``, and the
episode count ``strands_robots.tools.lerobot_train`` splits. Each graded the same
value its own way, so one file got four answers - and two of those answers were
silently destructive:

* ``int(2.5)`` truncates to ``2``, which is exactly the count a two-episode
  parquet holds, so a header no writer could have produced read as AGREEMENT
  between the two independent metadata sources.
* ``int(1e400)`` raises ``OverflowError``. ``1e400`` is a well-formed JSON number
  that ``json.load`` parses to ``inf``, so a readable file raised out of readers
  whose documented answer is "unknown" - and past the ``verify_dataset_episodes``
  envelope, which documents that a corrupt dataset is "reported as this same
  error dict, never raised".
* ``true`` counted as one episode (``bool`` is an ``int`` subclass), while the
  ``total_tasks`` reader in the SAME module already excluded it.
* ``"2"`` was two episodes to three readers and unusable to the fourth.

:func:`~strands_robots.utils.declared_count` is that one owner. A declaration
outside its domain is "no count declared", and a reader that must not read that
as an absent header reports it: the recorder's helper carries it in
``info_problems``, verify_dataset appends a problem.

Fixtures are pyarrow + a raw ``info.json`` string (not ``json.dumps``, which
cannot write ``1e400`` or ``NaN`` from a Python value), so these tests need
neither lerobot nor mujoco. The agent-callable facade is pinned against a real
recorded dataset in ``tests/simulation/test_run_policy_episode_contract.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq

import strands_robots.tools.lerobot_train as lerobot_train_tool
from strands_robots.dataset_recorder import read_dataset_episode_indices
from strands_robots.training.lerobot import LerobotTrainer
from strands_robots.utils import declared_count
from strands_robots.verify_dataset import verify_dataset

#: Every spelling a two-episode dataset's header could carry that is not a
#: count. ``2.0`` and ``"2"`` are the "right number, wrong type" pair; ``2.5``
#: is the one that truncated INTO agreement; ``1e400`` is the one that raised.
UNUSABLE_DECLARATIONS = ["2.0", "2.5", "true", '"2"', "1e400", "NaN", "-2", "null"]


def _dataset(root: Path, *, declaration: str | None, episodes: int = 2) -> str:
    """Write a two-episode dataset whose header declares ``declaration`` verbatim.

    Args:
        root: Dataset root to create.
        declaration: Raw JSON text for ``total_episodes``, or ``None`` to omit
            the key entirely (the documented "header declares nothing" case).
        episodes: Number of distinct episodes to write into the parquet.

    Returns:
        The dataset root as a string, ready for every reader under test.
    """
    ep_dir = root / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"episode_index": list(range(episodes)), "length": [4] * episodes}),
        ep_dir / "episodes_000.parquet",
    )
    body = "" if declaration is None else f', "total_episodes": {declaration}'
    (root / "meta" / "info.json").write_text('{"codebase_version": "v3.0"' + body + "}")
    return str(root)


class TestDeclaredCountDomain:
    """The owner answers with the count, or with "none declared"."""

    @pytest.mark.parametrize("value", [2.0, 2.5, True, False, "2", float("inf"), float("nan"), -1, None, {}, [2]])
    def test_a_value_that_is_not_a_count_declares_none(self, value: object) -> None:
        assert declared_count(value) is None

    @pytest.mark.parametrize("value", [0, 2, 99])
    def test_a_non_negative_int_is_the_count_it_declares(self, value: int) -> None:
        assert declared_count(value) == value


class TestEveryReaderReachesTheSameVerdict:
    """One header, one verdict - across all four readers of the file."""

    @pytest.mark.parametrize("declaration", UNUSABLE_DECLARATIONS)
    def test_the_parquet_cross_check_declares_no_count_and_says_why(self, declaration: str, tmp_path: Path) -> None:
        """No count, and a problem naming the header - never a nearby number.

        The problem is what stops a cross-check reading the ``None`` as an absent
        header, which is agreement (the parquet is then the sole truth).
        """
        info = read_dataset_episode_indices(_dataset(tmp_path, declaration=declaration))
        assert info["info_total_episodes"] is None
        assert len(info["info_problems"]) == 1
        assert "total_episodes" in info["info_problems"][0]
        assert info["total_episodes"] == 2, "the parquet truth is still reported"

    @pytest.mark.parametrize("declaration", UNUSABLE_DECLARATIONS)
    def test_the_validation_split_denominator_is_unknown(self, declaration: str, tmp_path: Path) -> None:
        root = _dataset(tmp_path, declaration=declaration)
        assert LerobotTrainer()._dataset_total_episodes(root) is None

    @pytest.mark.parametrize("declaration", UNUSABLE_DECLARATIONS)
    def test_the_train_tool_refuses_the_count_by_name(self, declaration: str, tmp_path: Path) -> None:
        root = _dataset(tmp_path, declaration=declaration)
        with pytest.raises(ValueError, match="total_episodes"):
            lerobot_train_tool._read_total_episodes(root)

    @pytest.mark.parametrize("declaration", UNUSABLE_DECLARATIONS)
    def test_the_drift_check_reports_the_corrupt_header(self, declaration: str, tmp_path: Path) -> None:
        report = verify_dataset(_dataset(tmp_path, declaration=declaration))
        assert report["ok"] is False
        assert report["info_total_episodes"] is None
        assert any("is not a count" in p for p in report["problems"]), report["problems"]

    @pytest.mark.parametrize("declaration", UNUSABLE_DECLARATIONS)
    def test_no_reader_raises_on_a_readable_file(self, declaration: str, tmp_path: Path) -> None:
        """``1e400`` parses to ``inf`` and ``int(inf)`` raises ``OverflowError``.

        The file is readable, so the answer must be a verdict, not an exception
        escaping from three readers that document "unknown".
        """
        root = _dataset(tmp_path, declaration=declaration)
        read_dataset_episode_indices(root)
        LerobotTrainer()._dataset_total_episodes(root)
        verify_dataset(root)

    def test_a_frame_total_that_is_not_a_count_is_reported_too(self, tmp_path: Path) -> None:
        """The sibling header in the same check shares the domain."""
        root = tmp_path / "frames"
        ep_dir = root / "meta" / "episodes" / "chunk-000"
        ep_dir.mkdir(parents=True)
        pq.write_table(pa.table({"episode_index": [0, 1], "length": [4, 4]}), ep_dir / "episodes_000.parquet")
        (root / "meta" / "info.json").write_text('{"total_episodes": 2, "total_frames": 8.5}')
        report = verify_dataset(str(root))
        assert report["info_total_frames"] is None
        assert any("total_frames=8.5 is not a count" in p for p in report["problems"]), report["problems"]


class TestTheHealthyAndUnknownCasesAreUnchanged:
    """Controls: the values a writer really produces keep their verdicts."""

    def test_a_matching_int_header_agrees_everywhere(self, tmp_path: Path) -> None:
        root = _dataset(tmp_path, declaration="2")
        info = read_dataset_episode_indices(root)
        assert info["info_total_episodes"] == 2
        assert info["info_problems"] == []
        assert LerobotTrainer()._dataset_total_episodes(root) == 2
        assert lerobot_train_tool._read_total_episodes(root) == 2
        report = verify_dataset(root)
        assert report["info_total_episodes"] == 2
        assert not any("total_episodes" in p for p in report["problems"]), report["problems"]

    def test_a_wrong_int_header_is_still_reported_as_drift(self, tmp_path: Path) -> None:
        """The existing drift message is unchanged - a wrong COUNT is not corrupt."""
        report = verify_dataset(_dataset(tmp_path, declaration="99"))
        assert report["info_total_episodes"] == 99
        assert any("total_episodes=99 disagrees with parquet" in p for p in report["problems"])
        assert not any("is not a count" in p for p in report["problems"])

    def test_an_absent_header_stays_the_unknown_case(self, tmp_path: Path) -> None:
        """No declaration is not a corrupt declaration: the parquet is sole truth."""
        root = _dataset(tmp_path, declaration=None)
        info = read_dataset_episode_indices(root)
        assert info["info_total_episodes"] is None
        assert info["info_problems"] == []
        assert LerobotTrainer()._dataset_total_episodes(root) is None
        assert not any("total_episodes" in p for p in verify_dataset(root)["problems"])
