# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for per-camera config construction on the hardware ``Robot``.

``Robot(..., cameras={"front": {...}})`` describes each camera with a free-form
dict. Those keys are the fields of lerobot's ``OpenCVCameraConfig``, so the
accepted vocabulary must be *derived* from the dataclass rather than
hand-picked. A hand-picked list has two silent failure modes, both pinned here:

    - every field it forgets is **unreachable** -- no caller can set it at all,
      so the camera streams at the default with no signal (``warmup_s`` and
      ``backend`` were both unreachable);
    - every key it does not recognise is **discarded** -- so ``heigth=1080``
      reported success having configured 480p (AGENTS.md > Review Learnings
      (#86) > "Reject silently-dropped kwargs").

The reachability test is deliberately written against
``dataclasses.fields(OpenCVCameraConfig)`` rather than a fixed list of field
names, so a field lerobot adds in a future release fails this test until it is
verified reachable, instead of quietly joining the unreachable set.

No serial/USB hardware is touched: only config dataclasses are constructed.
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("lerobot")

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from strands_robots.hardware_robot import _OPENCV_CAMERA_DEFAULTS, _build_camera_config

from .test_hardware_robot_config import _make_robot

# A non-default, valid value for every field an operator can configure.
# ``index_or_path`` is the one required field and is supplied separately.
#
# Keyed by field name so the coverage assertion below is exact: this map must
# name every declared field, which is what makes "a new lerobot field is
# reachable" a test failure rather than a silent regression.
_PROBE_VALUES: dict[str, object] = {
    "index_or_path": 1,
    "fps": 15,
    "width": 800,
    "height": 600,
    "color_mode": "bgr",
    "rotation": 90,
    "warmup_s": 5,
    "fourcc": "MJPG",
    "backend": 200,  # cv2.CAP_V4L2
}


def _declared_fields() -> dict[str, dataclasses.Field]:
    return {f.name: f for f in dataclasses.fields(OpenCVCameraConfig)}


class TestFieldReachability:
    def test_probe_values_cover_every_declared_field(self):
        """The probe map must name every field, so new lerobot fields surface.

        Without this, a field added by a lerobot release would simply be absent
        from the reachability test below and could go unreachable unnoticed --
        exactly how ``warmup_s`` and ``backend`` were lost.
        """
        assert set(_PROBE_VALUES) == set(_declared_fields())

    @pytest.mark.parametrize("field_name", sorted(_PROBE_VALUES))
    def test_declared_field_is_reachable_from_a_camera_dict(self, field_name: str):
        """Every declared field can be set through the per-camera dict.

        Pre-fix the builder hand-picked six of the nine fields, so ``warmup_s``
        and ``backend`` could not be configured at all: the value was accepted
        and the config carried the default.
        """
        probe = _PROBE_VALUES[field_name]
        cfg = _build_camera_config("front", {"index_or_path": 0, field_name: probe})

        stored = getattr(cfg, field_name)
        # ``color_mode`` / ``rotation`` / ``backend`` are coerced to enums by
        # lerobot's ``__post_init__``; compare on the underlying value.
        assert getattr(stored, "value", stored) == probe

    def test_strands_defaults_apply_and_explicit_values_win(self):
        """The documented 30fps/640x480 defaults survive the fields-driven build.

        lerobot leaves ``fps``/``width``/``height`` as ``None`` ("negotiate with
        the device"); strands_robots pins them so an unconfigured camera has a
        predictable stream. That contract is behaviour, not an implementation
        detail of the old hand-picked list.
        """
        default = _build_camera_config("front", {"index_or_path": 0})
        assert (default.fps, default.width, default.height) == (30, 640, 480)

        explicit = _build_camera_config("front", {"index_or_path": 0, "fps": 60, "width": 1280, "height": 720})
        assert (explicit.fps, explicit.width, explicit.height) == (60, 1280, 720)

    def test_strands_defaults_are_declared_fields(self):
        """A strands default must name a field lerobot still declares.

        If lerobot renames one of these, forwarding the stale key would fail at
        construction time for every camera. Catch the drift here instead.
        """
        assert set(_OPENCV_CAMERA_DEFAULTS) <= set(_declared_fields())


class TestUnknownOptionsRefused:
    def test_typo_is_refused_with_the_closest_field_named(self):
        """A misspelled option raises instead of configuring the default.

        Pre-fix ``heigth=1080`` returned a config at 480p with status success.
        """
        with pytest.raises(ValueError) as excinfo:
            _build_camera_config("front", {"index_or_path": 0, "heigth": 1080})

        message = str(excinfo.value)
        assert "heigth" in message
        assert "'height'" in message  # the suggestion
        assert "'front'" in message  # which camera is at fault

    def test_unrecognisable_option_lists_the_accepted_fields(self):
        """An option with no close match still reports the accepted vocabulary."""
        with pytest.raises(ValueError) as excinfo:
            _build_camera_config("front", {"index_or_path": 0, "resolution": [1920, 1080]})

        message = str(excinfo.value)
        assert "resolution" in message
        for name in _declared_fields():
            assert name in message

    def test_missing_required_option_names_it(self):
        """An omitted required field reports the option, not a bare ``KeyError``.

        Pre-fix this raised ``KeyError: 'index_or_path'`` out of the builder,
        naming neither the camera nor what was required of it.
        """
        with pytest.raises(ValueError, match=r"'front' is missing required option\(s\): \['index_or_path'\]"):
            _build_camera_config("front", {"type": "opencv"})

    def test_non_mapping_camera_config_is_refused(self):
        """A camera entry that is not a mapping reports what was passed.

        Pre-fix this raised ``AttributeError: 'str' object has no attribute
        'get'`` from the first ``config.get(...)``.
        """
        with pytest.raises(ValueError, match=r"Camera 'front' config must be a mapping"):
            _build_camera_config("front", "video0")

    def test_value_rejected_by_lerobot_names_the_camera(self):
        """lerobot's own option validation is reported against the camera name.

        ``fourcc`` must be a 4-character code; lerobot raises with no idea which
        entry of a multi-camera ``cameras`` dict it came from.
        """
        with pytest.raises(ValueError, match=r"for camera 'wrist'"):
            _build_camera_config("wrist", {"index_or_path": 0, "fourcc": "MJP"})


class TestWiredIntoCreateMinimalConfig:
    """The contract holds through the public ``cameras=`` entry point.

    The tests above exercise the builder directly; these pin that
    ``_create_minimal_config`` actually routes through it, so the guard cannot
    be bypassed by the surface an operator/agent really calls.
    """

    def test_previously_unreachable_fields_reach_the_robot_config(self):
        hw = _make_robot()
        cfg = hw._create_minimal_config(
            "so101_follower",
            {"front": {"type": "opencv", "index_or_path": 0, "warmup_s": 4, "backend": 200}},
            port="/dev/ttyACM0",
        )
        cam = cfg.cameras["front"]
        assert cam.warmup_s == 4
        assert cam.backend.value == 200

    def test_typo_in_one_camera_of_several_is_refused(self):
        hw = _make_robot()
        with pytest.raises(ValueError, match=r"Unknown option\(s\) for camera 'wrist'.*fourc"):
            hw._create_minimal_config(
                "so101_follower",
                {
                    "front": {"type": "opencv", "index_or_path": 0},
                    "wrist": {"type": "opencv", "index_or_path": 1, "fourc": "MJPG"},
                },
                port="/dev/ttyACM0",
            )
