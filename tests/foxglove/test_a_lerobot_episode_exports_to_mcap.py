"""A LeRobot v3 episode exports to an MCAP Foxglove can open, and ``mcap_info`` reads any MCAP back.

MCAP is an export and a sidecar here, never the training format: the dataset
stays where the recorder wrote it and a new file is derived from one episode.
Skipped without the ``[foxglove]`` extra; the export cell also needs lerobot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("foxglove")

from strands_robots.foxglove import export_episode, mcap_info  # noqa: E402

_LENGTHS = (3, 2)


@pytest.fixture
def recorded(tmp_path: Path) -> Path:
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "dataset"
    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        "observation.images.front": {"dtype": "image", "shape": [8, 8, 3], "names": ["height", "width", "channels"]},
    }
    writer = LeRobotDataset.create(repo_id="local/probe", fps=10, root=str(root), features=features, robot_type="probe")
    for episode, length in enumerate(_LENGTHS):
        for frame in range(length):
            sample = np.array([episode, frame], dtype=np.float32)
            image = np.full((8, 8, 3), 40 * (frame + 1), dtype=np.uint8)
            writer.add_frame(
                {"observation.state": sample, "action": sample, "observation.images.front": image, "task": "probe"}
            )
        writer.save_episode()
    writer.finalize()
    return root


class TestExport:
    def test_one_episode_becomes_one_seekable_file(self, recorded: Path, tmp_path: Path) -> None:
        out = tmp_path / "ep1.mcap"
        result = export_episode(recorded, 1, out)
        assert result["frames"] == _LENGTHS[1]
        assert result["fps"] == 10.0
        assert result["cameras"] == ["observation.images.front"]
        assert result["channels"] == [
            "/action/state",
            "/lerobot/episode",
            "/observation/images/front",
            "/observation/state",
        ]
        info = mcap_info(out)
        assert info["messages"] == 1 + 3 * _LENGTHS[1]
        assert info["channels"]["/observation/state"] == {
            "schema": "lerobot.Scalars",
            "encoding": "json",
            "messages": 2,
        }
        assert info["channels"]["/observation/images/front"]["schema"] == "foxglove.CompressedImage"
        assert info["seconds"] == pytest.approx((_LENGTHS[1] - 1) / 10.0, abs=1e-6), "stamped from the timestamp column"

    def test_the_scalars_carry_the_dataset_names_and_values(self, recorded: Path, tmp_path: Path) -> None:
        from mcap.reader import make_reader

        out = tmp_path / "ep0.mcap"
        export_episode(recorded, 0, out, repo_id="local/probe")
        with out.open("rb") as handle:
            states = [
                json.loads(message.data)
                for _, channel, message in make_reader(handle).iter_messages()
                if channel.topic == "/observation/state"
            ]
        assert [s["scalars"] for s in states][1] == [{"label": "a", "value": 0.0}, {"label": "b", "value": 1.0}]
        with out.open("rb") as handle:
            (episode,) = [
                json.loads(message.data)
                for _, channel, message in make_reader(handle).iter_messages()
                if channel.topic == "/lerobot/episode"
            ]
        assert episode == {"repo_id": "local/probe", "episode": 0, "task": "probe", "fps": 10.0, "frames": 3}

    def test_the_repo_id_defaults_to_the_directory_name(self, recorded: Path, tmp_path: Path) -> None:
        from mcap.reader import make_reader

        out = tmp_path / "named.mcap"
        export_episode(recorded, 0, out)
        with out.open("rb") as handle:
            (episode,) = [
                json.loads(m.data) for _, c, m in make_reader(handle).iter_messages() if c.topic == "/lerobot/episode"
            ]
        assert episode["repo_id"] == "local/dataset"

    def test_an_existing_file_is_refused(self, recorded: Path, tmp_path: Path) -> None:
        out = tmp_path / "taken.mcap"
        out.write_bytes(b"")
        with pytest.raises(ValueError, match=r"export_episode: out_path=.*already exists"):
            export_episode(recorded, 0, out)

    def test_an_episode_past_the_end_is_refused_by_number(self, recorded: Path, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"episode_index 5 is past the last episode \(1\)"):
            export_episode(recorded, 5, tmp_path / "x.mcap")

    @pytest.mark.parametrize("index", [-1, True, 1.5])
    def test_an_index_that_is_not_a_count_is_refused(self, recorded: Path, tmp_path: Path, index: object) -> None:
        with pytest.raises(ValueError, match=r"episode_index must be a non-negative int"):
            export_episode(recorded, index, tmp_path / "x.mcap")  # type: ignore[arg-type]


class TestInfo:
    def test_a_missing_file_is_a_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"mcap_info: .* is not a file"):
            mcap_info(tmp_path / "absent.mcap")

    def test_a_file_written_by_the_sdk_reports_channels_counts_and_span(self, tmp_path: Path) -> None:
        import foxglove
        from foxglove.channels import LogChannel
        from foxglove.messages import Log, Timestamp

        path = tmp_path / "log.mcap"
        context = foxglove.Context()
        writer = foxglove.open_mcap(str(path), context=context)
        channel = LogChannel("/strands/log", context=context)
        for i in range(4):
            channel.log(Log(timestamp=Timestamp(sec=100 + i, nsec=0), message=f"line {i}"), log_time=(100 + i) * 10**9)
        writer.close()
        info = mcap_info(path)
        assert info["messages"] == 4
        assert info["channels"] == {"/strands/log": {"schema": "foxglove.Log", "encoding": "protobuf", "messages": 4}}
        assert (info["start_ns"], info["end_ns"], info["seconds"]) == (100 * 10**9, 103 * 10**9, 3.0)
        assert info["bytes"] == path.stat().st_size
