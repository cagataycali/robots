"""Tests for strands_robots.motion_library — Motion, MotionLibrary."""

import json
import tempfile
from pathlib import Path

import numpy as np

from strands_robots.motion_library import Motion, MotionLibrary


class TestMotion:
    def test_basic_motion(self):
        data = {
            "name": "wave",
            "description": "Wave hand",
            "time": [0.0, 0.1, 0.2],
            "joint_positions": [[0.0, 0.0], [0.1, 0.2], [0.3, 0.4]],
        }
        m = Motion("wave", data)
        assert m.name == "wave"
        assert m.n_steps == 3
        assert m.duration == 0.2
        assert m.description == "Wave hand"

    def test_as_numpy(self):
        data = {
            "time": [0.0, 0.1],
            "joint_positions": [[1.0, 2.0], [3.0, 4.0]],
        }
        m = Motion("test", data)
        arr = m.as_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)
        assert arr[0, 0] == 1.0

    def test_as_numpy_dict_positions(self):
        data = {
            "time": [0.0, 0.1],
            "joint_positions": [{"j0": 1.0, "j1": 2.0}, {"j0": 3.0, "j1": 4.0}],
        }
        m = Motion("test", data)
        arr = m.as_numpy()
        assert arr.shape == (2, 2)

    def test_get_action_at(self):
        data = {
            "time": [0.0, 0.1, 0.2],
            "joint_positions": [{"j0": 0.0}, {"j0": 0.5}, {"j0": 1.0}],
        }
        m = Motion("test", data)
        action = m.get_action_at(0.0)
        assert action["j0"] == 0.0
        action = m.get_action_at(0.15)
        assert action is not None  # Should return nearest

    def test_repr(self):
        data = {"time": [0.0, 1.0], "joint_positions": [[0], [1]]}
        m = Motion("test", data)
        assert "test" in repr(m)
        assert "2" in repr(m)  # steps

    def test_empty_motion(self):
        m = Motion("empty", {})
        assert m.n_steps == 0
        assert m.duration == 0.0
        assert m.as_numpy() is None
        assert m.get_action_at(0.0) is None


class TestMotionLibrary:
    def test_register_and_get(self):
        lib = MotionLibrary()
        data = {"time": [0.0], "joint_positions": [[0.1, 0.2]]}
        lib.register("test_motion", data)
        m = lib.get("test_motion")
        assert m is not None
        assert m.name == "test_motion"

    def test_list_motions(self):
        lib = MotionLibrary()
        lib.register("a", {"time": [0.0], "joint_positions": [[0.0]]})
        lib.register("b", {"time": [0.0, 0.1], "joint_positions": [[0.0], [0.1]]})
        motions = lib.list_motions()
        assert len(motions) == 2
        names = [m["name"] for m in motions]
        assert "a" in names
        assert "b" in names

    def test_search(self):
        lib = MotionLibrary()
        lib.register("happy_wave", {"description": "A happy waving motion", "time": [], "joint_positions": []})
        lib.register("sad_drop", {"description": "A sad dropping motion", "time": [], "joint_positions": []})
        results = lib.search("happy")
        assert len(results) == 1
        assert results[0].name == "happy_wave"

    def test_count(self):
        lib = MotionLibrary()
        assert lib.count == 0
        lib.register("x", {"time": [], "joint_positions": []})
        assert lib.count == 1

    def test_load_file(self):
        data = {"name": "from_file", "time": [0.0, 0.5], "joint_positions": [[0.0], [1.0]]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name

        lib = MotionLibrary()
        motion = lib.load_file(path)
        assert motion is not None
        assert motion.name == "from_file"

    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"name": "dir_motion", "time": [0.0], "joint_positions": [[0.0]]}
            with open(Path(tmpdir) / "motion1.json", "w") as f:
                json.dump(data, f)

            lib = MotionLibrary()
            count = lib.load_directory(tmpdir)
            assert count == 1
            assert lib.get("dir_motion") is not None

    def test_save(self):
        lib = MotionLibrary()
        data = {"time": [0.0], "joint_positions": [[0.5]]}
        lib.register("save_test", data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = lib.save("save_test", path=str(Path(tmpdir) / "saved.json"))
            assert Path(path).exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["time"] == [0.0]

    def test_repr(self):
        lib = MotionLibrary()
        assert "0 motions" in repr(lib)
