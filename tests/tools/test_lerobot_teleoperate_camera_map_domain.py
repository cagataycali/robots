"""``build_lerobot_command`` must refuse a camera map the argv would not carry.

``robot_cameras`` is the one argument this tool renders into the lerobot argv as
*structure* rather than as a value: it becomes the nested-dict literal draccus
parses for ``--robot.cameras``, so a name or a value carrying a delimiter changes
the shape of that dict instead of the number in it. Nothing checked it. Measured
on ``e588be1``, the rendered flag for each request:

| request | rendered | consequence |
| --- | --- | --- |
| ``{"front": {"index": 4}}`` | ``index_or_path: 0`` | camera **0** recorded under the name ``front`` |
| ``{"front": {"framerate": 60}}`` | ``fps: 30`` | the rate asked for is not the rate recorded |
| ``{"front": {"resolution": "1920x1080"}}`` | ``640x480`` | ditto for geometry |
| ``{"front": {"fps": 0}}`` | ``fps: 0`` | the same quantity ``--dataset.fps 0`` is refused |
| ``{"front": {"width": -640}}`` | ``width: -640`` | ditto |
| ``{"front": {"type": "realsense"}}`` | ``type: realsense`` | not a registered backend (lerobot's is ``intelrealsense``); the subprocess dies in its log |
| ``{"top": {"type": "intelrealsense", "serial_number_or_name": "0123"}}`` | ``serial_number_or_name: 0123`` | YAML reads ``0123`` as octal: the serial arrives as ``"83"`` |
| ``{"front,wrist": {}}`` | one entry parsed as two | draccus cannot parse it |
| ``{"front": {"index_or_path": "0, wrist: {type: opencv, index_or_path: 5"}}`` | a **second camera** | the argv describes a set the call never named |
| ``{"front": "opencv"}`` | ``AttributeError`` | ``"Command build failed: 'str' object has no attribute 'get'"`` names nothing |

The first three rows are the silent ones, and they are the reason an unknown
option is a refusal rather than a default: ``_build_camera_arg`` renders the
default for every option an entry does not name, so a misspelling is not dropped
- it records an episode from a device nobody asked for, under the caller's own
camera name, and reports ``status="success"``. The last rows are the failure
this module's numeric table already exists to prevent: the map goes onto the
command line of a subprocess started with ``start_new_session=True``, which is
not a channel the call can read a failure back from.

Which ``type`` values exist and which options each admits is lerobot's
``CameraConfig`` registry's to say, read through the one owner
(``hardware_robot._camera_option_vocabulary``) rather than a list copied here:
a copy admitted the ``realsense`` spelling and refused the
``serial_number_or_name`` a RealSense is identified by. A string value is
quoted at the render so draccus reads it back as the string it was, which is
the only closure for the last rows: a blocklist of structure characters misses
``[``, ``]``, ``?``, a leading ``*`` or ``&``, and every re-typing YAML does
(``0123``, ``yes``, ``~``, ``1e3``).

The geometry rows are checked with the guards ``lerobot_camera`` already reads
``width`` / ``height`` / ``fps`` with, so a frame size or rate that tool refuses
cannot reach a recording through this one either.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("psutil")
# The vocabulary is lerobot's camera registry, so every cell that judges a map
# needs it; the sibling registry test gates on the same module.
pytest.importorskip("lerobot")

import strands_robots.tools.lerobot_teleoperate as tele_mod  # noqa: E402
from strands_robots.tools import _process_stop  # noqa: E402

build_lerobot_command = tele_mod.build_lerobot_command
lerobot_teleoperate = tele_mod.lerobot_teleoperate


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Keep the module-level session store inside the test's temp dir."""
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(_process_stop, "SESSION_DIR", session_dir)
    return session_dir


def _teleop(cameras: Any, **overrides: Any) -> list[str]:
    """A ``lerobot-teleoperate`` argv carrying ``cameras``."""
    kwargs: dict[str, Any] = {
        "action": "start",
        "robot_type": "so101_follower",
        "robot_port": "/dev/ttyACM1",
        "teleop_type": "so101_leader",
        "teleop_port": "/dev/ttyACM0",
        "robot_cameras": cameras,
    }
    kwargs.update(overrides)
    return build_lerobot_command(**kwargs)


def _cameras_flag(argv: list[str]) -> str | None:
    """The ``--robot.cameras=`` token, or ``None`` when no camera is emitted."""
    return next((arg for arg in argv if arg.startswith("--robot.cameras=")), None)


# Every request the argv could not have carried as asked, with the substring the
# refusal must name so a caller can act on it. Table-driven because the point is
# that one rule covers the whole map, not that each cell has its own guard.
UNUSABLE = [
    pytest.param({"front": {"index": 4}}, "'index'", id="misspelled-index"),
    pytest.param({"front": {"framerate": 60}}, "'framerate'", id="misspelled-fps"),
    pytest.param({"front": {"resolution": "1920x1080"}}, "'resolution'", id="resolution"),
    pytest.param({"front": {"fps": 0}}, "fps", id="zero-rate"),
    pytest.param({"front": {"width": -640}}, "width", id="negative-width"),
    pytest.param({"front": {"height": 2.5}}, "height", id="fractional-height"),
    pytest.param({"front": {"fps": True}}, "fps", id="bool-rate"),
    pytest.param({"front,wrist": {}}, "bare token", id="comma-in-name"),
    pytest.param({"front wrist": {}}, "bare token", id="space-in-name"),
    pytest.param({"front\n--robot.port=/dev/x": {}}, "bare token", id="newline-in-name"),
    pytest.param({"": {}}, "non-empty string", id="empty-name"),
    pytest.param({7: {}}, "non-empty string", id="non-str-name"),
    pytest.param({"front": {"type": "opencv}"}}, "Unsupported camera type", id="brace-in-type"),
    pytest.param({"front": {"type": 0}}, "Unsupported camera type", id="non-str-type"),
    pytest.param({"front": {"type": ["opencv"]}}, "Unsupported camera type", id="unhashable-type"),
    # The spelling the tool's own schema once suggested; lerobot registers
    # ``intelrealsense``, and the refusal must say so rather than leave the
    # caller hunting for one that does not exist.
    pytest.param({"top": {"type": "realsense"}}, "Did you mean 'intelrealsense'?", id="realsense-spelling"),
    # The option set is the resolved class's: a RealSense is identified by
    # ``serial_number_or_name``, and the hint names it.
    pytest.param(
        {"top": {"type": "intelrealsense", "index_or_path": 0}},
        "serial_number_or_name",
        id="opencv-option-on-a-realsense",
    ),
    pytest.param({"top": {"type": "intelrealsense", "serial_number_or_name": ""}}, "is empty", id="empty-serial"),
    pytest.param({"front": {"index_or_path": -1}}, "index_or_path", id="negative-index"),
    pytest.param({"front": {"index_or_path": ""}}, "index_or_path", id="empty-device"),
    pytest.param({"front": "opencv"}, "must be a mapping", id="entry-is-not-a-mapping"),
    pytest.param(["front"], "must be a mapping", id="map-is-not-a-mapping"),
    pytest.param("front", "must be a mapping", id="map-is-a-string"),
]

# Maps a run can honour, with the flag each must render. The control that makes
# the refusals above a narrowing rather than a blanket one. A string value is
# single-quoted, the YAML form draccus reads back verbatim (see the module
# docstring); a number renders bare, as the ``int`` the field declares.
USABLE = [
    pytest.param(
        {"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}},
        "--robot.cameras={'front': {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}",
        id="fully-specified",
    ),
    pytest.param(
        {"front": {"fps": 60}},
        "--robot.cameras={'front': {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 60}}",
        id="partially-specified-defaults-the-rest",
    ),
    pytest.param(
        {"wrist-1": {"index_or_path": "/dev/video0", "type": "opencv"}},
        "--robot.cameras={'wrist-1': {type: opencv, index_or_path: '/dev/video0', width: 640, height: 480, fps: 30}}",
        id="device-path-and-a-hyphenated-name",
    ),
    pytest.param(
        {"front": {}, "wrist": {"index_or_path": 2}},
        "--robot.cameras={'front': {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, "
        "'wrist': {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}",
        id="two-cameras",
    ),
    # The options are the resolved class's, so a RealSense renders the field
    # that identifies it and none of OpenCV's; the geometry defaults still apply.
    pytest.param(
        {"top": {"type": "intelrealsense", "serial_number_or_name": "0123", "use_depth": True}},
        "--robot.cameras={'top': {type: intelrealsense, width: 640, height: 480, fps: 30, "
        "serial_number_or_name: '0123', use_depth: True}}",
        id="realsense-by-serial",
    ),
    # Quoted, a path is carried as the path it is: unquoted, ``[`` fails the
    # flow-mapping parse and ``, wrist: {...`` describes a second camera.
    pytest.param(
        {"front": {"index_or_path": "/dev/v4l/by-id/usb-046d_C922[1]", "fourcc": "MJPG"}},
        "--robot.cameras={'front': {type: opencv, index_or_path: '/dev/v4l/by-id/usb-046d_C922[1]', "
        "width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}",
        id="a-bracket-in-a-device-path",
    ),
    pytest.param(
        {"front": {"index_or_path": "0, wrist: {type: opencv, index_or_path: 5"}},
        "--robot.cameras={'front': {type: opencv, index_or_path: '0, wrist: {type: opencv, index_or_path: 5', "
        "width: 640, height: 480, fps: 30}}",
        id="delimiters-in-a-value-are-a-value",
    ),
    pytest.param(
        {"front": {"index_or_path": "/dev/vi'deo"}},
        "--robot.cameras={'front': {type: opencv, index_or_path: '/dev/vi''deo', width: 640, height: 480, fps: 30}}",
        id="a-quote-in-a-device-path",
    ),
]


class TestACameraMapTheArgvCannotCarryIsRefusedBeforeIt:
    """The whole map is judged before a single flag is rendered."""

    @pytest.mark.parametrize(
        ("cameras", "named"), [(p.values[0], p.values[1]) for p in UNUSABLE], ids=[p.id for p in UNUSABLE]
    )
    def test_an_unusable_map_is_refused_and_the_refusal_names_it(self, cameras: Any, named: str) -> None:
        with pytest.raises(ValueError) as excinfo:
            _teleop(cameras)
        text = str(excinfo.value)
        assert "robot_cameras" in text, text
        assert named in text, text

    @pytest.mark.parametrize(
        ("cameras", "expected"), [(p.values[0], p.values[1]) for p in USABLE], ids=[p.id for p in USABLE]
    )
    def test_a_usable_map_still_renders_exactly_as_before(self, cameras: Any, expected: str) -> None:
        assert _cameras_flag(_teleop(cameras)) == expected

    def test_no_cameras_emits_no_flag(self) -> None:
        """The absent and the empty map are both "record no camera", not errors."""
        assert _cameras_flag(_teleop(None)) is None
        assert _cameras_flag(_teleop({})) is None

    @pytest.mark.parametrize("value", [30, 30.0, np.int64(30)])
    def test_an_integral_rate_is_honoured(self, value: Any) -> None:
        """A rate read from a config or promoted by NumPy is a usable rate."""
        assert "fps: 30}" in str(_cameras_flag(_teleop({"front": {"fps": value}})))

    @pytest.mark.parametrize("value", [4, 4.0, np.int64(4), np.float64(4.0)])
    def test_an_integral_index_is_rendered_as_the_whole_number(self, value: Any) -> None:
        """An index the guard accepts is emitted as the ``int`` lerobot decodes.

        draccus decodes ``index_or_path: 4`` as ``int`` and refuses ``4.0`` as
        both an ``int`` and a ``Path``, so an integral real that passed the
        guard must not reach the argv in its own spelling.
        """
        assert "index_or_path: 4," in str(_cameras_flag(_teleop({"front": {"index_or_path": value}})))

    @pytest.mark.parametrize("name", ["yes", "no", "null", "true", "false", "on", "off", "Yes", "TRUE", "Off"])
    def test_a_yaml_boolean_or_null_camera_name_round_trips_as_the_string(self, name: str) -> None:
        """YAML 1.1 re-types bare ``yes``/``no``/``null`` as booleans/null.

        A camera named ``yes`` must appear in the argv as the string ``'yes'``
        (single-quoted), not as the boolean ``True`` that an unquoted bare
        ``yes:`` resolves to under draccus / pyyaml. Measured against
        lerobot 0.6.1: ``{yes: {...}}`` decoded to key ``'True'`` while
        ``{'yes': {...}}`` decoded to key ``'yes'``.
        """
        flag = _cameras_flag(_teleop({name: {}}))
        assert flag is not None
        assert f"'{name}':" in flag, f"camera name {name!r} must be quoted in {flag}"


class TestEveryModeThatEmitsTheMapChecksIt:
    """The map is on the argv of all four modes, so the rule cannot be per-mode."""

    @pytest.mark.parametrize("mode", ["teleoperate", "record", "replay", "dagger"])
    def test_the_refusal_reaches_each_mode(self, mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
        kwargs: dict[str, Any] = {
            "action": "start",
            "robot_type": "so101_follower",
            "robot_port": "/dev/ttyACM1",
            "teleop_type": "so101_leader",
            "teleop_port": "/dev/ttyACM0",
            "robot_cameras": {"front": {"index": 4}},
        }
        if mode in ("record", "replay", "dagger"):
            kwargs["dataset_repo_id"] = "user/pick"
            kwargs["dataset_single_task"] = "pick the cube"
        if mode == "replay":
            kwargs["action"] = "replay"
        if mode == "dagger":
            kwargs["action"] = "dagger"
            kwargs["policy_path"] = "lerobot/act_so101"
        with pytest.raises(ValueError, match="index"):
            build_lerobot_command(**kwargs)


class TestTheToolAnswersTheCallerInsteadOfASessionLog:
    """A refused map must leave no session and start no process."""

    def test_the_tool_reports_the_refusal_without_starting_a_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _never(*args: Any, **kwargs: Any):
            raise AssertionError("subprocess.Popen must not be reached for a refused call")

        monkeypatch.setattr(tele_mod.subprocess, "Popen", _never)
        result = lerobot_teleoperate(
            action="start",
            robot_type="so101_follower",
            robot_port="/dev/ttyACM1",
            teleop_type="so101_leader",
            teleop_port="/dev/ttyACM0",
            robot_cameras={"front": {"index": 4}},
            session_name="refused",
        )
        assert result["status"] == "error"
        text = "\n".join(item.get("text", "") for item in result["content"] if "text" in item)
        assert "robot_cameras" in text and "index" in text, text
        assert tele_mod.SessionManager().get_session("refused") is None

    def test_a_non_mapping_entry_no_longer_answers_with_an_attribute_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-fix: ``'str' object has no attribute 'get'``, which names nothing."""
        monkeypatch.setattr(
            tele_mod.subprocess,
            "Popen",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not launch")),
        )
        result = lerobot_teleoperate(
            action="start",
            robot_type="so101_follower",
            robot_port="/dev/ttyACM1",
            teleop_type="so101_leader",
            teleop_port="/dev/ttyACM0",
            robot_cameras={"front": "opencv"},
            session_name="refused",
        )
        text = "\n".join(item.get("text", "") for item in result["content"] if "text" in item)
        assert result["status"] == "error"
        assert "has no attribute" not in text, text
        assert "robot_cameras['front']" in text, text


class TestTheOptionSetIsTheRegistrys:
    """The vocabulary is lerobot's camera registry, read through its one owner.

    A list copied here would admit a ``type`` the registry does not know and
    refuse a field a backend declares, which is the defect this file closes; so
    the population is derived from the registry rather than restated.
    """

    def test_every_registered_backend_renders_and_every_declared_field_is_admitted(self) -> None:
        import dataclasses

        from lerobot.cameras.configs import CameraConfig

        from strands_robots.hardware_robot import _ensure_lerobot_cameras_registered

        _ensure_lerobot_cameras_registered()
        choices = sorted(CameraConfig.get_known_choices())
        assert {"opencv", "intelrealsense"} <= set(choices), choices
        numeric = {name for name, _ in tele_mod._CAMERA_RENDER_DEFAULTS}
        for choice in choices:
            fields = [f.name for f in dataclasses.fields(CameraConfig.get_choice_class(choice))]
            # Every field the class declares is an admitted option. The four
            # with a numeric domain take a whole number; every other one takes
            # a string, which the render must quote whatever the field's type.
            entry = {"type": choice, **{name: (7 if name in numeric else "x") for name in fields}}
            flag = str(_cameras_flag(_teleop({"cam": entry})))
            assert flag.startswith(f"--robot.cameras={{cam: {{type: {choice}, "), flag
            for name in fields:
                assert f"{name}: {7 if name in numeric else chr(39) + 'x' + chr(39)}" in flag, (choice, name, flag)

    def test_a_field_the_backend_does_not_declare_is_refused_with_the_backends_fields(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _teleop({"top": {"type": "intelrealsense", "fourcc": "MJPG"}})
        text = str(excinfo.value)
        assert "RealSenseCameraConfig accepts" in text, text
        assert "'fourcc'" not in text.split("accepts")[1], text  # an OpenCV-only field is not offered

    def test_the_defaults_render_only_where_the_backend_declares_them(self) -> None:
        """``index_or_path`` names an OpenCV device; a RealSense has no such field."""
        flag = str(_cameras_flag(_teleop({"top": {"type": "intelrealsense", "serial_number_or_name": "1"}})))
        assert "index_or_path" not in flag, flag
        assert "width: 640, height: 480, fps: 30" in flag, flag

    def test_the_geometry_options_share_the_recorders_domain(self) -> None:
        """The same guard ``lerobot_camera`` reads pixels and frames with."""
        from strands_robots.utils import positive_whole_number_error

        assert {name for name, _ in tele_mod._CAMERA_GEOMETRY_DOMAINS} == {"width", "height", "fps"}
        assert {check for _, check in tele_mod._CAMERA_GEOMETRY_DOMAINS} == {positive_whole_number_error}
