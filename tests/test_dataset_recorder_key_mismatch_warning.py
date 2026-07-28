# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A schema that matches NOTHING must not silently record a dead column.

``add_frame`` iterates the declared schema names and appends ``0.0`` for any name
absent from the observation/action dict. There was no match counter, no warning
and no all-missed detection. So if the schema was declared with bare joint names
while the runtime emits lerobot-style ``.pos`` keys (or the reverse, or a
multi-robot prefix that was not remapped), EVERY frame was written as an all-zero
state and action vector - and ``add_frame`` returned normally, ``frame_count``
incremented, ``save_episode`` succeeded and ``stop_recording`` reported success
with the full frame count. Measured pre-fix::

    declared joint_names=['shoulder_pan', ...]; fed '.pos'-suffixed keys
    frames=3  dropped=0  save_episode=success
    not one log line about the mismatch

``verify_dataset`` on the result reports "identically zero across episode 0 - dead
control column", i.e. the damage is detectable only after the fact.

The same file already does this correctly for CAMERAS: when no observed stream
matches a declared image key it emits a one-shot warning naming the observed keys,
the declared keys and the ``camera_key_map`` remedy. The state/action path - the
load-bearing signal - had no equivalent.

Only the ALL-missed case is flagged. A partial miss ("declare 6 joints, this frame
carries 5") is the legitimate zero-fill path and stays silent, which is what keeps
the existing ``test_add_frame_fills_missing_keys_with_zero`` green.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lerobot")

from strands_robots.dataset_recorder import DatasetRecorder  # noqa: E402

_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
_MARKER = "all-zero vector"


def _recorder(tmp_path, case: str, *, strict: bool = False) -> DatasetRecorder:
    """A recorder declaring BARE joint names, in its own dataset root."""
    recorder = DatasetRecorder.create(
        repo_id=f"local/{case}",
        fps=30,
        robot_type="so101",
        joint_names=_JOINTS,
        action_names=_JOINTS,
        camera_keys=[],
        camera_dims={},
        task="t",
        root=str(tmp_path / case),
    )
    # ``create`` does not expose strict; it defaults True on the constructor.
    recorder.strict = strict
    return recorder


def _warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if _MARKER in record.getMessage()]


def _pos_keys(value: float) -> dict[str, float]:
    """lerobot-style keys against a bare-name schema: nothing matches."""
    return {f"{joint}.pos": value for joint in _JOINTS}


class TestAllMissedIsReported:
    def test_a_fully_mismatched_state_schema_warns(self, tmp_path, caplog):
        """The regression: 3 frames recorded all-zero with no log line."""
        recorder = _recorder(tmp_path, "state_miss")

        with caplog.at_level("WARNING"):
            recorder.add_frame(_pos_keys(0.5), _pos_keys(0.6), task="t")

        warnings = _warnings(caplog)
        assert warnings, [record.getMessage() for record in caplog.records]

    def test_the_warning_names_both_key_sets_and_the_remedy(self, tmp_path, caplog):
        recorder = _recorder(tmp_path, "state_text")

        with caplog.at_level("WARNING"):
            recorder.add_frame(_pos_keys(0.5), _pos_keys(0.6), task="t")

        warning = _warnings(caplog)[0]
        assert "shoulder_pan" in warning, warning
        assert "shoulder_pan.pos" in warning, warning
        assert "dead column" in warning, warning
        assert warning.isascii()

    def test_state_and_action_are_reported_separately(self, tmp_path, caplog):
        """Two independent columns, two independent diagnostics."""
        recorder = _recorder(tmp_path, "both")

        with caplog.at_level("WARNING"):
            recorder.add_frame(_pos_keys(0.5), _pos_keys(0.6), task="t")

        warnings = _warnings(caplog)
        assert any("state names" in text for text in warnings), warnings
        assert any("action names" in text for text in warnings), warnings

    def test_only_the_action_schema_mismatching_still_warns(self, tmp_path, caplog):
        """A good state vector must not mask a dead action column."""
        recorder = _recorder(tmp_path, "action_only")

        with caplog.at_level("WARNING"):
            recorder.add_frame({joint: 0.5 for joint in _JOINTS}, _pos_keys(0.6), task="t")

        warnings = _warnings(caplog)
        assert any("action names" in text for text in warnings), warnings
        assert not any("state names" in text for text in warnings), warnings

    def test_the_warning_is_one_shot(self, tmp_path, caplog):
        """50Hz would flood the log, matching the camera diagnostic's guard."""
        recorder = _recorder(tmp_path, "oneshot")

        with caplog.at_level("WARNING"):
            for _ in range(5):
                recorder.add_frame(_pos_keys(0.5), _pos_keys(0.6), task="t")

        # One for state, one for action, and no repeats.
        assert len(_warnings(caplog)) == 2, _warnings(caplog)

    def test_zero_filled_frames_are_counted(self, tmp_path):
        """The count survives the one-shot log guard, so it can be surfaced."""
        recorder = _recorder(tmp_path, "counted")

        for _ in range(3):
            recorder.add_frame(_pos_keys(0.5), _pos_keys(0.6), task="t")

        assert recorder.zero_filled_frame_count == 3
        assert recorder.frame_count == 3, "frames were dropped instead of recorded"


class TestStrictRefuses:
    def test_strict_raises_on_an_all_missed_schema(self, tmp_path):
        recorder = _recorder(tmp_path, "strict", strict=True)

        with pytest.raises(ValueError, match="no declared state name matched"):
            recorder.add_frame(_pos_keys(0.5), _pos_keys(0.6), task="t")

    def test_the_raise_names_the_declared_keys_and_the_escape_hatch(self, tmp_path):
        recorder = _recorder(tmp_path, "strict_text", strict=True)

        with pytest.raises(ValueError) as excinfo:
            recorder.add_frame(_pos_keys(0.5), _pos_keys(0.6), task="t")

        message = str(excinfo.value)
        assert "shoulder_pan" in message, message
        assert "strict=False" in message, message
        assert message.isascii()

    def test_strict_does_not_raise_on_a_partial_miss(self, tmp_path):
        """strict defaults True and the sim never sets it False, so a partial
        miss must NOT become a hard failure - that would break every legitimate
        recording with an optional key."""
        recorder = _recorder(tmp_path, "strict_partial", strict=True)

        recorder.add_frame({joint: 0.5 for joint in _JOINTS[:5]}, {joint: 0.6 for joint in _JOINTS[:5]}, task="t")

        assert recorder.frame_count == 1
        assert recorder.zero_filled_frame_count == 0


class TestPartialMissesStaySilent:
    @pytest.mark.parametrize("present", [5, 3, 1])
    def test_a_partial_match_does_not_warn(self, tmp_path, caplog, present):
        """The legitimate zero-fill path: some keys matched."""
        recorder = _recorder(tmp_path, f"partial{present}")

        with caplog.at_level("WARNING"):
            recorder.add_frame(
                {joint: 0.5 for joint in _JOINTS[:present]},
                {joint: 0.6 for joint in _JOINTS[:present]},
                task="t",
            )

        assert not _warnings(caplog), _warnings(caplog)
        assert recorder.zero_filled_frame_count == 0

    def test_a_full_match_does_not_warn(self, tmp_path, caplog):
        recorder = _recorder(tmp_path, "full")

        with caplog.at_level("WARNING"):
            recorder.add_frame({j: 0.5 for j in _JOINTS}, {j: 0.6 for j in _JOINTS}, task="t")

        assert not _warnings(caplog)
        assert recorder.zero_filled_frame_count == 0

    def test_the_zero_fill_still_happens_on_a_partial_miss(self, tmp_path):
        """Behaviour unchanged: the missing slot is zero-filled IN PLACE, so the
        following joints are not shifted into the wrong columns."""
        recorder = _recorder(tmp_path, "fill_shape")

        recorder.add_frame({joint: 0.5 for joint in _JOINTS[:5]}, {joint: 0.6 for joint in _JOINTS[:5]}, task="t")

        assert recorder.frame_count == 1


class TestCountersStartClean:
    def test_a_fresh_recorder_has_no_zero_filled_frames(self, tmp_path):
        recorder = _recorder(tmp_path, "fresh")

        assert recorder.zero_filled_frame_count == 0
        assert recorder._warned_state_mismatch is False
        assert recorder._warned_action_mismatch is False
