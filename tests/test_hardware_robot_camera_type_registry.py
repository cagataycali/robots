# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The camera ``type`` a hardware ``Robot`` accepts is lerobot's own registry.

``Robot(..., cameras={"top": {"type": ..., ...}})`` describes each camera with a
serialized lerobot ``CameraConfig``: ``type`` is draccus' choice discriminator
(``CameraConfig.type`` returns ``get_choice_name(cls)``) and the remaining keys
are fields of the class it selects. So the set of accepted ``type`` values must
be ``CameraConfig.get_known_choices()`` -- the same ``ChoiceRegistry`` lookup
``_create_minimal_config`` already uses for ``robot_type``, and the one
``lerobot.cameras.make_cameras_from_configs`` dispatches on.

Pre-fix the builder resolved ``OpenCVCameraConfig`` unconditionally and refused
every other ``type``, so a RealSense, a ZMQ stream and a Reachy 2 camera were
all unattachable through the factory -- while ``tools/lerobot_camera`` in this
same package opens a RealSense, and the docs' real-mode examples name
``realsense_top``. Every cell here fails on that builder.

The choices are populated lazily, which is the other half of the fix: importing
``lerobot.cameras`` registers nothing at all (pinned below), so a registry
lookup that skips the subpackage walk reports every camera type as unknown.

No camera hardware and no vendor SDK is touched: only config dataclasses are
constructed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("lerobot")

from lerobot.cameras.configs import CameraConfig

from strands_robots.hardware_robot import (
    _CAMERA_STREAM_DEFAULTS,
    _build_camera_config,
    _ensure_lerobot_cameras_registered,
)

from .test_hardware_robot_config import _make_robot

# A minimal valid option set for every camera backend lerobot registers, keyed
# by the ``type`` that selects it. Only the required fields are named; the rest
# come from lerobot's defaults and this package's stream defaults.
#
# Kept as an explicit map, with the coverage assertion below, for the same
# reason ``test_hardware_robot_camera_config._PROBE_VALUES`` is: a backend
# lerobot adds in a future release fails that cell until someone verifies it is
# attachable, instead of quietly joining an unreachable set. A derived value per
# field cannot replace it -- ``Reachy2CameraConfig.__post_init__`` enumerates
# ``name`` and ``image_type``, so only a real pairing constructs.
_MINIMAL_OPTIONS: dict[str, dict[str, object]] = {
    "opencv": {"index_or_path": 0},
    "intelrealsense": {"serial_number_or_name": "819312071961"},
    "zmq": {"server_address": "tcp://127.0.0.1"},
    "reachy2_camera": {"name": "teleop", "image_type": "left"},
}


@pytest.fixture(autouse=True)
def _registered() -> None:
    """Populate the choice registry the way ``Robot()`` does."""
    _ensure_lerobot_cameras_registered()


def _choices() -> list[str]:
    return sorted(CameraConfig.get_known_choices())


class TestTheRegistryIsTheVocabulary:
    def test_importing_lerobot_cameras_registers_nothing(self) -> None:
        """The premise for the walk: the choices are populated lazily.

        ``lerobot.cameras.__init__`` deliberately does not import its backend
        subpackages, and says so in a comment, to keep backend-specific
        dependencies out of every ``import lerobot``. A registry lookup that
        skips :func:`~strands_robots.hardware_robot._ensure_lerobot_cameras_registered`
        therefore answers *every* camera type as unknown -- including
        ``opencv``. Run in a fresh interpreter because registration is a
        process-global import side effect this session has already performed.

        Should lerobot start importing them eagerly, the walk stays correct (it
        is idempotent) and this cell is what says the premise moved.
        """
        probe = (
            "from lerobot.cameras.configs import CameraConfig\n"
            "import lerobot.cameras\n"
            "from lerobot.cameras.utils import make_cameras_from_configs\n"
            "print(sorted(CameraConfig.get_known_choices()))\n"
        )
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "[]"

    def test_the_walk_registers_the_backends_lerobot_ships(self) -> None:
        """The walk finds more than the one backend the builder used to hardcode."""
        assert "opencv" in _choices()
        assert "intelrealsense" in _choices()
        assert len(_choices()) > 1

    def test_every_registered_choice_is_covered_by_this_module(self) -> None:
        """A backend lerobot adds must be verified attachable, not assumed.

        This is the cell that fails when lerobot registers a fifth camera type:
        add it to ``_MINIMAL_OPTIONS`` and the parametrized cells below prove
        the factory can attach it.
        """
        assert sorted(_MINIMAL_OPTIONS) == _choices()

    @pytest.mark.parametrize("cam_type", sorted(_MINIMAL_OPTIONS))
    def test_every_registered_choice_builds_its_own_config_class(self, cam_type: str) -> None:
        """Each ``type`` resolves to the class lerobot registered under it.

        Pre-fix only ``opencv`` reached a config at all; every other cell raised
        ``Unsupported camera type``.
        """
        built = _build_camera_config("top", {"type": cam_type, **_MINIMAL_OPTIONS[cam_type]})

        assert isinstance(built, CameraConfig)
        assert type(built) is CameraConfig.get_choice_class(cam_type)
        # ``type`` round-trips: it is what ``make_cameras_from_configs``
        # dispatches on, so a config whose discriminator disagreed with the
        # requested type would open the wrong device.
        assert built.type == cam_type

    @pytest.mark.parametrize("cam_type", sorted(_MINIMAL_OPTIONS))
    def test_the_stream_defaults_apply_to_every_choice(self, cam_type: str) -> None:
        """``fps``/``width``/``height`` are declared on the ``CameraConfig`` base.

        So this package's documented stream defaults are not an OpenCV
        privilege: an unconfigured RealSense gets the same predictable stream as
        an unconfigured webcam, rather than lerobot's ``None`` ("whatever the
        device negotiates").
        """
        built = _build_camera_config("top", {"type": cam_type, **_MINIMAL_OPTIONS[cam_type]})

        assert (built.fps, built.width, built.height) == (
            _CAMERA_STREAM_DEFAULTS["fps"],
            _CAMERA_STREAM_DEFAULTS["width"],
            _CAMERA_STREAM_DEFAULTS["height"],
        )

    @pytest.mark.parametrize("cam_type", sorted(_MINIMAL_OPTIONS))
    def test_an_explicit_stream_option_still_wins(self, cam_type: str) -> None:
        """The defaults fill gaps; they never override a configured value."""
        built = _build_camera_config(
            "top", {"type": cam_type, **_MINIMAL_OPTIONS[cam_type], "width": 1280, "height": 720}
        )

        assert (built.width, built.height) == (1280, 720)

    def test_a_realsense_is_attachable_through_the_public_cameras_kwarg(self) -> None:
        """The contract holds through the surface an operator really calls.

        ``tools/lerobot_camera`` in this package opens a RealSense and the
        real-mode docs name ``realsense_top``, but pre-fix this raised
        ``Unsupported camera type`` -- the tool could list the device the
        factory refused to attach.
        """
        pytest.importorskip("lerobot.robots.so_follower")
        hw = _make_robot()

        cfg = hw._create_minimal_config(
            "so101_follower",
            {"realsense_top": {"type": "intelrealsense", "serial_number_or_name": "819312071961"}},
            port="/dev/ttyACM0",
        )

        cam = cfg.cameras["realsense_top"]
        assert type(cam).__name__ == "RealSenseCameraConfig"
        assert cam.serial_number_or_name == "819312071961"


class TestAnUnregisteredTypeIsRefusedWithTheRegistrySListing:
    def test_an_unregistered_type_lists_every_registered_one(self) -> None:
        """The refusal survives, and now says what the alternatives are.

        Pre-fix it named only the rejected value, so ``opencv`` -- the one type
        that worked -- was never mentioned.
        """
        with pytest.raises(ValueError) as excinfo:
            _build_camera_config("top", {"type": "thermal"})

        message = str(excinfo.value)
        assert "Unsupported camera type" in message
        assert "'thermal'" in message
        assert "'top'" in message  # which camera is at fault
        for choice in _choices():
            assert choice in message

    def test_the_realsense_spelling_is_answered_with_the_registered_one(self) -> None:
        """``realsense`` is the obvious guess and lerobot registers ``intelrealsense``.

        The two spellings are a real divergence inside this package: the
        ``lerobot_camera`` tool takes ``camera_type="realsense"`` as a tool
        argument, while the factory's ``type`` is draccus' discriminator and so
        must be the registry's name. Naming the registered spelling is what
        keeps that divergence from being a dead end -- the tool's own refusal
        test makes the same argument about not sending a caller "looking for a
        spelling that does not exist".
        """
        with pytest.raises(ValueError) as excinfo:
            _build_camera_config("top", {"type": "realsense", "serial_number_or_name": "123"})

        assert "Did you mean 'intelrealsense'?" in str(excinfo.value)

    def test_an_unhashable_type_is_refused_not_a_traceback(self) -> None:
        """A list can never be a registry key, so it is the same refusal.

        The lookup is a dict subscript; without this the ``TypeError:
        unhashable type`` escapes from inside draccus.
        """
        with pytest.raises(ValueError, match="Unsupported camera type"):
            _build_camera_config("top", {"type": ["intelrealsense"]})


class TestTheOptionVocabularyFollowsTheResolvedClass:
    def test_an_opencv_option_on_a_realsense_lists_realsenses_fields(self) -> None:
        """The accepted vocabulary is the resolved class's, not OpenCV's.

        ``index_or_path`` identifies a webcam; a RealSense is identified by
        ``serial_number_or_name``. Listing OpenCV's fields here would send the
        caller after an option this config does not declare.
        """
        with pytest.raises(ValueError) as excinfo:
            _build_camera_config("top", {"type": "intelrealsense", "index_or_path": 0})

        message = str(excinfo.value)
        assert "index_or_path" in message  # the offending key
        assert "RealSenseCameraConfig accepts" in message
        assert "serial_number_or_name" in message
        assert "'fourcc'" not in message  # an OpenCV-only field

    def test_a_realsense_only_field_is_reachable(self) -> None:
        """A field no OpenCV config declares can actually be configured.

        Pre-fix ``use_depth`` was unreachable through ``Robot()``: the only
        config class the builder would construct does not declare it.
        """
        built = _build_camera_config(
            "top", {"type": "intelrealsense", "serial_number_or_name": "123", "use_depth": True}
        )

        assert built.use_depth is True

    def test_a_missing_required_option_names_the_resolved_class(self) -> None:
        """Each backend has its own required field, and the refusal says which."""
        with pytest.raises(ValueError) as excinfo:
            _build_camera_config("top", {"type": "zmq"})

        message = str(excinfo.value)
        assert "missing required option(s): ['server_address']" in message
        assert "ZMQCameraConfig accepts" in message

    def test_a_value_lerobot_refuses_names_the_camera_and_the_class(self) -> None:
        """lerobot's own ``__post_init__`` validation is reported per camera.

        ``ZMQCameraConfig`` refuses a port outside 1..65535 with no idea which
        entry of a multi-camera ``cameras`` dict it came from.
        """
        with pytest.raises(ValueError) as excinfo:
            _build_camera_config("wrist", {"type": "zmq", "server_address": "tcp://127.0.0.1", "port": 0})

        message = str(excinfo.value)
        assert "Failed to construct ZMQCameraConfig for camera 'wrist'" in message
        assert "port" in message
