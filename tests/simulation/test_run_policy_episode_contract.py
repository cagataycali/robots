"""Episode-count contract for ``run_policy`` + ``verify_dataset_episodes``.

A recording driven by a single ``run_policy(n_episodes=1)`` buffers all frames
into one merged ``episode_index=0`` mega-episode, while an agent narrating "20
episodes" believes it recorded 20. ``status=OK`` from the rollout does not prove
episode-count correctness. These tests pin the contract that closes that gap:

* both the single-episode fast path and the multi-episode path of
  ``SimEngine.run_policy`` return the episode-count truth fields
  (``n_episodes_requested`` / ``n_episodes_completed`` / ``episodes_saved`` /
  ``dataset_episode_indices``);
* ``verify_dataset_episodes(expected)`` reads the on-disk parquet (the ground
  truth) and returns ``status=error`` when the recorded episode count differs
  from what the caller intended;
* the pure-pyarrow ``read_dataset_episode_indices`` helper reports the actual
  ``episode_index`` set and per-episode frame counts.

The fast path's loud INFO log (#3 in the contract) is exercised implicitly by
the recording path; the structured fields are the machine-checkable surface.
"""

from __future__ import annotations

import json
import shutil

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")
pytest.importorskip("pyarrow")

from strands_robots.dataset_recorder import read_dataset_episode_indices
from strands_robots.simulation.mujoco.simulation import Simulation


@pytest.fixture
def sim():
    s = Simulation(tool_name="episode_contract", mesh=False)
    s.create_world()
    s.add_robot(name="so100", data_config="so100")
    yield s
    s.cleanup()


def _record(sim: Simulation, root: str, *, n_episodes: int, n_steps: int = 4) -> dict:
    """Drive a full start -> run_policy -> stop recording cycle; return run json."""
    shutil.rmtree(root, ignore_errors=True)
    start = sim.start_recording(repo_id="local/episode_contract", task="t", fps=30, root=root, overwrite=True)
    assert start["status"] == "success", start
    run = sim.run_policy(
        robot_name="so100",
        policy_provider="mock",
        n_steps=n_steps,
        n_episodes=n_episodes,
        control_frequency=50,
        fast_mode=True,
    )
    assert run["status"] == "success", run
    stop = sim.stop_recording()
    assert stop["status"] == "success", stop
    return _json_block(run)


def _json_block(result: dict) -> dict:
    for blk in result.get("content", []):
        if isinstance(blk, dict) and isinstance(blk.get("json"), dict):
            return blk["json"]
    raise AssertionError(f"no json content block in {result}")


class TestRunPolicyContractFields:
    """Both run_policy paths must return episode-count truth fields."""

    def test_fast_path_returns_contract_fields(self, sim, tmp_path):
        run_json = _record(sim, str(tmp_path / "fast"), n_episodes=1)
        # Fast path: one rollout, frames buffered into the current episode and
        # flushed at stop_recording, so episodes_saved is 0 (no boundary flushed
        # WITHIN run_policy) and the flush is flagged as deferred.
        assert run_json["n_episodes_requested"] == 1
        assert run_json["n_episodes_completed"] == 1
        assert run_json["episodes_saved"] == 0
        assert run_json["episode_flush_deferred"] is True
        assert "dataset_episode_indices" in run_json

    def test_multi_path_returns_contract_fields(self, sim, tmp_path):
        run_json = _record(sim, str(tmp_path / "multi"), n_episodes=3)
        assert run_json["n_episodes_requested"] == 3
        assert run_json["n_episodes_completed"] == 3
        assert run_json["episodes_saved"] == 3
        assert run_json["dataset_episode_indices"] == [0, 1, 2]

    def test_fields_present_without_recording(self, sim):
        """Contract fields exist even when not recording (dataset indices empty)."""
        run = sim.run_policy(
            robot_name="so100", policy_provider="mock", n_steps=4, control_frequency=50, fast_mode=True
        )
        run_json = _json_block(run)
        assert run_json["n_episodes_requested"] == 1
        assert run_json["dataset_episode_indices"] == []


class TestParquetTruth:
    """The on-disk parquet is the ground truth for episode count."""

    def test_n_episodes_20_writes_20_parquet_rows(self, sim, tmp_path):
        root = str(tmp_path / "twenty")
        _record(sim, root, n_episodes=20, n_steps=3)
        info = read_dataset_episode_indices(root)
        assert info["total_episodes"] == 20
        assert info["episode_indices"] == list(range(20))
        assert info["frames_per_episode"] == [3] * 20
        assert info["total_frames"] == 60

    def test_n_episodes_1_writes_1_parquet_row(self, sim, tmp_path):
        root = str(tmp_path / "one")
        _record(sim, root, n_episodes=1, n_steps=12)
        info = read_dataset_episode_indices(root)
        assert info["total_episodes"] == 1
        assert info["episode_indices"] == [0]
        assert info["total_frames"] == 12

    def test_read_helper_raises_on_empty_dataset(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_dataset_episode_indices(str(tmp_path / "does_not_exist"))


class TestVerifyDatasetEpisodes:
    """verify_dataset_episodes is the definitive post-stop_recording check."""

    def test_verify_matches_after_20_episode_run(self, sim, tmp_path):
        _record(sim, str(tmp_path / "v20"), n_episodes=20, n_steps=3)
        result = sim.verify_dataset_episodes(expected=20)
        assert result["status"] == "success"
        vj = _json_block(result)
        assert vj["expected"] == 20
        assert vj["actual"] == 20
        assert vj["episode_indices"] == list(range(20))

    def test_verify_errors_after_1_episode_run_expecting_20(self, sim, tmp_path):
        _record(sim, str(tmp_path / "v1"), n_episodes=1, n_steps=12)
        result = sim.verify_dataset_episodes(expected=20)
        assert result["status"] == "error"
        vj = _json_block(result)
        assert vj["expected"] == 20
        assert vj["actual"] == 1

    def test_verify_after_stop_recording_uses_last_root(self, sim, tmp_path):
        """verify works after stop_recording drops the recorder (uses stashed root)."""
        _record(sim, str(tmp_path / "vlast"), n_episodes=2, n_steps=3)
        assert sim._world._backend_state.get("dataset_recorder") is None
        result = sim.verify_dataset_episodes(expected=2)
        assert result["status"] == "success"
        assert _json_block(result)["actual"] == 2

    def test_verify_no_dataset_returns_error(self, sim):
        result = sim.verify_dataset_episodes(expected=5)
        assert result["status"] == "error"

    def test_verify_rejects_negative_expected(self, sim, tmp_path):
        _record(sim, str(tmp_path / "vneg"), n_episodes=1, n_steps=3)
        result = sim.verify_dataset_episodes(expected=-1)
        assert result["status"] == "error"

    def test_verify_missing_episode_parquet_returns_full_contract(self, sim, tmp_path):
        """A finalized dataset whose episode parquet later goes missing must
        fail loudly with the full json contract, not crash.

        An interrupted finalize or a partially-deleted dataset leaves the
        recorded root in place but with no ``meta/episodes/**/*.parquet``. The
        parquet read then raises FileNotFoundError; verify must convert that
        into a structured ``status="error"`` carrying the complete machine-
        checkable block (actual 0, no info header, sources disagree) so CI can
        fail programmatically instead of hitting an unhandled exception.
        """
        root = str(tmp_path / "vmissing")
        _record(sim, root, n_episodes=2, n_steps=3)
        # Confirm it verifies before we corrupt it (guards against a no-op test).
        assert sim.verify_dataset_episodes(expected=2)["status"] == "success"

        removed = list((tmp_path / "vmissing" / "meta" / "episodes").glob("**/*.parquet"))
        assert removed, "expected at least one episode parquet to remove"
        for pf in removed:
            pf.unlink()

        result = sim.verify_dataset_episodes(expected=2)
        assert result["status"] == "error"
        assert "never finalized" in result["content"][0]["text"]
        vj = _json_block(result)
        assert vj["expected"] == 2
        assert vj["actual"] == 0
        assert vj["info_total_episodes"] is None
        assert vj["sources_agree"] is False
        assert vj["episode_indices"] == []
        assert vj["total_frames"] == 0
        assert vj["total_frames_per_ep"] == []
        assert vj["root"] == root


class TestInfoParquetCrossCheck:
    """verify_dataset_episodes requires meta/info.json AND parquet to agree.

    A dataset can carry the right parquet episode count yet a stale/inconsistent
    ``meta/info.json`` header (e.g. an interrupted finalize). A parquet-only
    check would pass it; the two independent metadata sources must agree.
    """

    def test_read_helper_reports_info_total_episodes(self, sim, tmp_path):
        root = str(tmp_path / "info")
        _record(sim, root, n_episodes=3, n_steps=3)
        info = read_dataset_episode_indices(root)
        # info.json header agrees with the parquet on a healthy dataset.
        assert info["info_total_episodes"] == 3
        assert info["total_episodes"] == 3

    def test_verify_reports_agreement_on_healthy_dataset(self, sim, tmp_path):
        _record(sim, str(tmp_path / "agree"), n_episodes=2, n_steps=3)
        result = sim.verify_dataset_episodes(expected=2)
        assert result["status"] == "success"
        vj = _json_block(result)
        assert vj["info_total_episodes"] == 2
        assert vj["sources_agree"] is True

    def test_verify_errors_when_info_json_disagrees_with_parquet(self, sim, tmp_path):
        """info.json claims a different count than the parquet -> MISMATCH.

        The parquet holds exactly ``expected`` episodes, so a parquet-only check
        would (wrongly) pass. Cross-checking info.json catches the inconsistent
        dataset and fails loudly.
        """
        root = tmp_path / "skew"
        _record(sim, str(root), n_episodes=2, n_steps=3)

        info_path = root / "meta" / "info.json"
        meta = json.loads(info_path.read_text())
        meta["total_episodes"] = 99  # corrupt: header disagrees with 2 parquet episodes
        info_path.write_text(json.dumps(meta))

        result = sim.verify_dataset_episodes(expected=2)
        assert result["status"] == "error", "parquet matches expected but info.json disagrees"
        vj = _json_block(result)
        assert vj["actual"] == 2
        assert vj["info_total_episodes"] == 99
        assert vj["sources_agree"] is False

    def test_verify_treats_parquet_as_sole_truth_when_info_json_absent(self, sim, tmp_path):
        """Missing info.json -> parquet is the sole truth; verify still passes."""
        root = tmp_path / "noinfo"
        _record(sim, str(root), n_episodes=2, n_steps=3)
        (root / "meta" / "info.json").unlink()

        result = sim.verify_dataset_episodes(expected=2)
        assert result["status"] == "success"
        vj = _json_block(result)
        assert vj["info_total_episodes"] is None
        assert vj["sources_agree"] is True
