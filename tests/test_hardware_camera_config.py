# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Camera spec keys must reach the lerobot config, or be rejected loudly.

``_create_minimal_config`` built camera configs from a hand-picked list of 6 keys
(later 8), discarding everything else with no error. So:

* ``four_cc`` (a typo for ``fourcc``) vanished - and a dropped ``fourcc`` silently
  caps a UVC camera at ~5fps, which presents later as a policy starved of frames
  rather than as a configuration mistake;
* ``backend`` was a REAL ``OpenCVCameraConfig`` field that could never be set;
* ``use_depth`` / ``use_rgb`` were unreachable because ``type="realsense"`` raised
  "Unsupported camera type" even though lerobot ships ``RealSenseCameraConfig``;
* any nonsense key was accepted in silence.

That is the exact opposite of the loud unknown-kwarg rejection the same method
applies to ROBOT kwargs a few lines later, and it contradicts AGENTS.md's "Reject
silently-dropped kwargs".

Construction is now driven by the target dataclass's own fields, so a new lerobot
camera field works with no change here (matching the robot-kwarg contract) while
unknown keys raise.

No camera device is opened: only config dataclasses are constructed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lerobot")

from strands_robots.hardware_robot import _build_camera_config  # noqa: E402

_OPENCV = {"type": "opencv", "index_or_path": "/dev/video0"}


class TestKnownKeysReachTheConfig:
    def test_the_previously_forwarded_keys_still_work(self):
        """No regression on the keys the hand-picked version did forward."""
        config = _build_camera_config(
            "front",
            {**_OPENCV, "fps": 30, "width": 640, "height": 480, "fourcc": "MJPG", "warmup_s": 3},
        )

        assert config.fps == 30
        assert config.width == 640
        assert config.height == 480
        assert config.fourcc == "MJPG"
        assert config.warmup_s == 3

    def test_backend_now_reaches_the_dataclass(self):
        """Regression: a real field that the hand-picked list could never set."""
        from lerobot.cameras.configs import Cv2Backends

        config = _build_camera_config("front", {**_OPENCV, "backend": 200})

        assert config.backend is Cv2Backends.V4L2

    def test_index_or_path_is_preserved(self):
        assert _build_camera_config("front", _OPENCV).index_or_path == "/dev/video0"


class TestUnknownKeysAreRejected:
    @pytest.mark.parametrize("bad_key", ["four_cc", "fourCC", "bogus_key", "with"])
    def test_typos_and_nonsense_raise(self, bad_key):
        with pytest.raises(ValueError, match="Unknown camera key"):
            _build_camera_config("front", {**_OPENCV, bad_key: "x"})

    def test_error_names_the_camera_the_key_and_the_valid_fields(self):
        with pytest.raises(ValueError) as excinfo:
            _build_camera_config("wrist", {**_OPENCV, "four_cc": "MJPG"})

        message = str(excinfo.value)
        assert "'wrist'" in message
        assert "four_cc" in message
        assert "fourcc" in message  # the valid field list guides the fix
        assert "typo" in message

    def test_type_itself_is_not_reported_as_unknown(self):
        """``type`` is strands-level metadata, not a dataclass field."""
        assert _build_camera_config("front", _OPENCV) is not None

    def test_error_message_is_plain_ascii(self):
        """AGENTS.md: user-facing strings are plain ASCII only."""
        with pytest.raises(ValueError) as excinfo:
            _build_camera_config("front", {**_OPENCV, "nope": 1})

        assert str(excinfo.value).isascii()


class TestRealSenseIsReachable:
    def test_realsense_config_is_built(self):
        """Regression: this raised "Unsupported camera type: realsense"."""
        pytest.importorskip("lerobot.cameras.realsense.configuration_realsense")

        config = _build_camera_config("depth", {"type": "realsense", "serial_number_or_name": "123", "use_depth": True})

        assert type(config).__name__ == "RealSenseCameraConfig"
        assert config.use_depth is True

    def test_realsense_rejects_an_opencv_only_key(self):
        """Per-type field sets: ``fourcc`` is not a RealSense field."""
        pytest.importorskip("lerobot.cameras.realsense.configuration_realsense")

        with pytest.raises(ValueError, match="Unknown camera key"):
            _build_camera_config("depth", {"type": "realsense", "serial_number_or_name": "123", "fourcc": "MJPG"})


class TestUnsupportedTypes:
    def test_unknown_type_lists_the_supported_ones(self):
        with pytest.raises(ValueError) as excinfo:
            _build_camera_config("front", {"type": "webcam42", "index_or_path": 0})

        message = str(excinfo.value)
        assert "webcam42" in message
        assert "opencv" in message  # the supported list is named

    def test_missing_required_field_raises_with_context(self):
        """A missing ``index_or_path`` must name the camera, not TypeError."""
        with pytest.raises(ValueError, match="Failed to construct"):
            _build_camera_config("front", {"type": "opencv"})


class TestThroughTheRealConfigBuilder:
    def _cameras(self, cameras):
        from strands_robots.hardware_robot import Robot

        hw = Robot.__new__(Robot)
        hw.tool_name_str = "so101"
        return hw._create_minimal_config("so101_follower", cameras=cameras, port="/dev/null").cameras

    def test_dual_camera_spec_lands_on_the_robot_config(self):
        """The user's live two-camera MJPG setup, end to end."""
        built = self._cameras(
            {
                "wrist": {**_OPENCV, "fps": 30, "width": 640, "height": 480, "fourcc": "MJPG"},
                "front": {
                    "type": "opencv",
                    "index_or_path": "/dev/video2",
                    "fps": 30,
                    "width": 640,
                    "height": 480,
                    "fourcc": "MJPG",
                },
            }
        )

        assert set(built) == {"wrist", "front"}
        assert built["wrist"].fourcc == "MJPG"
        assert built["front"].index_or_path == "/dev/video2"

    def test_a_camera_typo_fails_the_whole_construction(self):
        """Better to fail at construction than to run with a silently dropped key."""
        with pytest.raises(ValueError, match="Unknown camera key"):
            self._cameras({"front": {**_OPENCV, "four_cc": "MJPG"}})
