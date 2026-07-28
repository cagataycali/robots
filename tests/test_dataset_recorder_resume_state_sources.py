# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A resumed recorder must still read VECTOR state from its source keys.

``create`` sets ``_state_source_keys`` (``["hip", "knee", "base_pos",
"base_quat"]``) so ``add_frame`` reads the SOURCE key ``base_quat`` and flattens it
into the four expanded ``base_quat.*`` schema slots. ``resume`` seeded only
``episode_count``/``frame_count``, leaving ``_state_source_keys`` at its ``None``
default - so ``add_frame`` fell back to the dataset's EXPANDED names
(``base_pos.x``, ``base_quat.w``, ...), none of which exist in the observation, and
zero-filled the whole block. Measured, one frame per session with an IDENTICAL
observation::

    schema names: ['hip','knee','base_pos.x',...,'base_quat.w',...]
    create  _state_source_keys: ['hip','knee','base_pos','base_quat']
    SESSION1 state: [0.1, 0.2, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    resumed _state_source_keys: None
    SESSION2 state: [0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

So every appended episode of a humanoid / quadruped / mobile-base / LeKiwi dataset
lost its entire base block: no height, no orientation, no velocities. And
``base_quat = (0,0,0,0)`` is an INVALID quaternion (norm 0), so downstream rotation
math is undefined rather than merely wrong. ``verify_dataset``'s dead-column check
requires the WHOLE vector to be zero, so the still-varying joints kept it quiet.

The fix reconstructs the source order from the inherited schema. A scalar-only
schema (every fixed-base arm, the common live path) keeps ``_state_source_keys``
at ``None`` and behaves exactly as before.
"""

from __future__ import annotations

import glob

import numpy as np
import pytest

pytest.importorskip("lerobot")

import strands_robots.dataset_recorder as dr  # noqa: E402

_OBS = {"hip": 0.1, "knee": 0.2, "base_pos": [1.0, 2.0, 3.0], "base_quat": [1.0, 0.0, 0.0, 0.0]}
_ACT = {"hip": 0.1, "knee": 0.2}
_SPECS = [("base_pos", ["x", "y", "z"]), ("base_quat", ["w", "x", "y", "z"])]


def _create(root, **extra):
    return dr.DatasetRecorder.create(
        repo_id="u/base",
        fps=30,
        robot_type="go2",
        joint_names=["hip", "knee"],
        action_names=["hip", "knee"],
        camera_keys=[],
        camera_dims={},
        task="t",
        root=str(root),
        **extra,
    )


def _recorded_states(root) -> list[list[float]]:
    """Every ``observation.state`` row on disk, in file order."""
    pandas = pytest.importorskip("pandas")
    rows: list[list[float]] = []
    for path in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True)):
        for value in pandas.read_parquet(path)["observation.state"].tolist():
            rows.append([float(v) for v in np.asarray(value)])
    return rows


class TestVectorStateSurvivesResume:
    def test_both_sessions_record_the_same_state(self, tmp_path):
        """The regression: session 2's base block came back all zeros."""
        root = tmp_path / "ds"
        first = _create(root, extra_state_specs=_SPECS)
        first.add_frame(dict(_OBS), dict(_ACT), task="t")
        first.save_episode()
        first.finalize()

        resumed = dr.DatasetRecorder.resume(repo_id="u/base", root=str(root), task="t")
        resumed.add_frame(dict(_OBS), dict(_ACT), task="t")
        resumed.save_episode()
        resumed.finalize()

        states = _recorded_states(root)
        assert len(states) == 2, states
        assert states[0] == pytest.approx(states[1]), (
            f"appended episode differs from the first for an identical observation: {states}"
        )

    def test_the_resumed_recorder_recovers_the_source_keys(self, tmp_path):
        root = tmp_path / "ds"
        first = _create(root, extra_state_specs=_SPECS)
        expected = list(first._state_source_keys or [])
        assert expected, "the fixture did not set source keys on create"
        first.add_frame(dict(_OBS), dict(_ACT), task="t")
        first.save_episode()
        first.finalize()

        resumed = dr.DatasetRecorder.resume(repo_id="u/base", root=str(root), task="t")

        assert resumed._state_source_keys == expected

    def test_the_base_quaternion_stays_normalized(self, tmp_path):
        """(0,0,0,0) is not a wrong rotation, it is an invalid one."""
        root = tmp_path / "ds"
        first = _create(root, extra_state_specs=_SPECS)
        first.add_frame(dict(_OBS), dict(_ACT), task="t")
        first.save_episode()
        first.finalize()

        resumed = dr.DatasetRecorder.resume(repo_id="u/base", root=str(root), task="t")
        resumed.add_frame(dict(_OBS), dict(_ACT), task="t")
        resumed.save_episode()
        resumed.finalize()

        for index, state in enumerate(_recorded_states(root)):
            quat = np.asarray(state[5:9], dtype=float)
            assert float(np.linalg.norm(quat)) == pytest.approx(1.0, abs=1e-5), (
                f"frame {index} quaternion {quat.tolist()} has norm {float(np.linalg.norm(quat)):.4f}"
            )

    def test_the_base_block_is_not_zero_filled(self, tmp_path):
        """Explicitly pin the values, not just their equality across sessions."""
        root = tmp_path / "ds"
        first = _create(root, extra_state_specs=_SPECS)
        first.add_frame(dict(_OBS), dict(_ACT), task="t")
        first.save_episode()
        first.finalize()

        resumed = dr.DatasetRecorder.resume(repo_id="u/base", root=str(root), task="t")
        resumed.add_frame(dict(_OBS), dict(_ACT), task="t")
        resumed.save_episode()
        resumed.finalize()

        appended = _recorded_states(root)[1]
        assert appended[2:5] == pytest.approx([1.0, 2.0, 3.0]), appended
        assert appended[5:9] == pytest.approx([1.0, 0.0, 0.0, 0.0]), appended


class TestScalarOnlySchemasAreUnchanged:
    def test_a_scalar_schema_keeps_source_keys_none(self, tmp_path):
        """The common live path (fixed-base arm) must be byte-identical."""
        root = tmp_path / "flat"
        first = dr.DatasetRecorder.create(
            repo_id="u/flat",
            fps=30,
            robot_type="so101",
            joint_names=["j1", "j2"],
            action_names=["j1", "j2"],
            camera_keys=[],
            camera_dims={},
            task="t",
            root=str(root),
        )
        first.add_frame({"j1": 0.1, "j2": 0.2}, {"j1": 0.1, "j2": 0.2}, task="t")
        first.save_episode()
        first.finalize()

        resumed = dr.DatasetRecorder.resume(repo_id="u/flat", root=str(root), task="t")

        assert resumed._state_source_keys is None

    def test_a_scalar_schema_still_records_correctly_after_resume(self, tmp_path):
        root = tmp_path / "flat"
        first = dr.DatasetRecorder.create(
            repo_id="u/flat",
            fps=30,
            robot_type="so101",
            joint_names=["j1", "j2"],
            action_names=["j1", "j2"],
            camera_keys=[],
            camera_dims={},
            task="t",
            root=str(root),
        )
        first.add_frame({"j1": 0.1, "j2": 0.2}, {"j1": 0.1, "j2": 0.2}, task="t")
        first.save_episode()
        first.finalize()

        resumed = dr.DatasetRecorder.resume(repo_id="u/flat", root=str(root), task="t")
        resumed.add_frame({"j1": 0.1, "j2": 0.2}, {"j1": 0.1, "j2": 0.2}, task="t")
        resumed.save_episode()
        resumed.finalize()

        states = _recorded_states(root)
        assert states[0] == pytest.approx(states[1])


class TestTheSchemaInversionHelper:
    """``_state_source_keys_from_schema`` unit cases."""

    class _Fake:
        def __init__(self, names) -> None:
            self.features = {"observation.state": {"names": list(names)}}

    def test_a_flat_schema_returns_none(self):
        assert dr._state_source_keys_from_schema(self._Fake(["a", "b"])) is None

    def test_an_expanded_schema_collapses_to_source_keys(self):
        result = dr._state_source_keys_from_schema(self._Fake(["a", "b", "base_pos.x", "base_pos.y", "base_pos.z"]))

        assert result == ["a", "b", "base_pos"]

    def test_first_appearance_order_is_preserved(self):
        result = dr._state_source_keys_from_schema(
            self._Fake(["base_quat.w", "hip", "base_pos.x", "base_pos.y", "base_quat.x"])
        )

        assert result == ["base_quat", "hip", "base_pos"]

    def test_a_dotted_joint_name_groups_from_the_right(self):
        """A component split must take the LAST dot, not the first."""
        result = dr._state_source_keys_from_schema(self._Fake(["arm.wrist.a", "arm.wrist.b", "j"]))

        assert result == ["arm.wrist", "j"]

    def test_an_empty_schema_returns_none(self):
        assert dr._state_source_keys_from_schema(self._Fake([])) is None

    def test_an_unreadable_dataset_returns_none(self):
        """Must not raise on an unexpected lerobot layout."""
        assert dr._state_source_keys_from_schema(object()) is None
