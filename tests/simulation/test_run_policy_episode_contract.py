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
