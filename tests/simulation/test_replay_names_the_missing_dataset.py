"""``replay_episode`` on a dataset that does not exist names what was checked.

``LeRobotDataset`` turns a miss on disk into a Hub download, so a typo'd or
never-recorded ``repo_id`` surfaced as a raw ``huggingface_hub`` 404 (request
id, ``repo_type`` advice, gated-repo paragraph) after a network round trip, with
no mention of the directory checked or of the dataset this session just
recorded. The refusal now names the directory, the Hub verdict, the datasets
beside it on disk, the session's most recent recording and the remedy.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
from strands_robots.simulation.policy_runner import missing_dataset_text


class RepositoryNotFoundError(Exception):
    pass


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def test_not_found_names_directory_hub_verdict_and_remedy(tmp_path):
    exc = Exception(
        "404 Client Error. (Request ID: Root=1-abc)\n\nRepository Not Found for url: https://huggingface.co/api/datasets/lab/nope/refs."
    )
    text = missing_dataset_text("replay_episode", "lab/nope", str(tmp_path / "nope"), exc)
    assert text.startswith(
        f"replay_episode: no dataset 'lab/nope' - not on disk at {tmp_path / 'nope'} and not on the Hugging Face Hub (Repository Not Found)."
    )
    assert "Request ID" not in text and "gated" not in text
    assert text.endswith(
        "Pass root= for a dataset written elsewhere, or record one with start_recording + run_policy + stop_recording."
    )


def test_not_found_by_exception_type_lists_neighbouring_datasets(tmp_path):
    (tmp_path / "lab" / "good" / "meta").mkdir(parents=True)
    (tmp_path / "lab" / "also" / "meta").mkdir(parents=True)
    (tmp_path / "lab" / "not-a-dataset").mkdir()
    text = missing_dataset_text(
        "replay_episode", "lab/nope", str(tmp_path / "lab" / "nope"), RepositoryNotFoundError("x")
    )
    assert f"Datasets on disk beside it ({tmp_path / 'lab'}): also, good." in text
    assert "not-a-dataset" not in text


def test_offline_reads_differently(tmp_path):
    exc = OSError("HTTPSConnectionPool: Max retries exceeded (Name or service not known)")
    text = missing_dataset_text("replay_episode", "owner/ds", str(tmp_path / "ds"), exc)
    assert text.startswith(
        f"replay_episode: no local copy of 'owner/ds' at {tmp_path / 'ds'}, and the Hugging Face Hub could not be reached."
    )


def test_other_errors_keep_their_own_text(tmp_path):
    assert (
        missing_dataset_text("replay_episode", "a/b", None, ValueError("Episode 3 out of range (0-1)"))
        == "Episode 3 out of range (0-1)"
    )


def test_default_root_is_lerobot_home_over_repo_id(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    text = missing_dataset_text("replay_episode", "lab/nope", None, RepositoryNotFoundError("x"))
    assert "not on disk at " in text and "lab/nope" in text.split("not on disk at ", 1)[1].split(" and not", 1)[0]


def test_replay_surface_names_the_sessions_last_recording(monkeypatch, tmp_path):
    sim = MuJoCoSimEngine(tool_name="rp_missing", mesh=False)
    sim.create_world()
    assert sim.add_robot(name="so101", data_config="so101")["status"] == "success"
    try:
        sim._world._backend_state["last_dataset_root"] = str(tmp_path / "recorded")

        def boom(repo_id, episode=0, root=None):
            raise RepositoryNotFoundError(
                "Repository Not Found for url: https://huggingface.co/api/datasets/lab/nope/refs."
            )

        import strands_robots.dataset_recorder as dataset_recorder

        monkeypatch.setattr(dataset_recorder, "load_lerobot_episode", boom)
        r = sim.replay_episode("lab/nope")
        assert r["status"] == "error"
        assert _text(r).startswith("replay_episode: no dataset 'lab/nope' - not on disk at ")
        assert f"This session's most recent recording is at {tmp_path / 'recorded'}." in _text(r)
    finally:
        sim.cleanup()
