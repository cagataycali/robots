"""``Robot("so101", mode="real", cameras={...})`` on the native driver opens, reads and closes them.

Until the native drivers became the hardware default (2026-10-01) the Feetech
driver discarded ``cameras=`` (``del cameras``) and the factory refused a
non-empty dict for it, so a caller who wanted a wrist camera on an SO-101 had
to take the lerobot path. Now the bare call builds this driver, and the
dashboard's spawn sends ``cameras={"main": {"index_or_path": 1, "fps": 30,
"width": 640, "height": 480}}`` with it, so the driver has to own the cameras
the way :mod:`strands_robots.drivers.base` says an opted-in driver does: open
them in ``connect_eagerly`` after the bus (a camera that fails costs the camera,
not the arm), hand their frames to the mesh, close them in ``cleanup``.

Measured before the change, on the branch's first commit::

    >>> Robot("so101", mode="real", port="/dev/null", cameras={"main": {"index_or_path": 1}})
    ValueError: FeetechDriver does not open cameras, so cameras= cannot be honored for 'so101'.

No hardware here: the bus is the twin-free fake the seam tests use (a port that
never opens is enough, because ``connect_eagerly`` reports the bus and the
cameras separately) and the camera class is swapped for a recorder, so the
cells grade the ownership contract and not OpenCV.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

import strands_robots.drivers.cameras as cameras_mod
from strands_robots import Robot
from strands_robots.drivers.cameras import (
    CameraSpec,
    camera_entry_error,
    camera_names,
    camera_read_source,
    camera_specs,
)
from strands_robots.drivers.feetech.driver import FeetechDriver

_CAMERAS = {"main": {"index_or_path": 1, "fps": 30, "width": 640, "height": 480}}


class _RecordingCamera:
    """Stands in for :class:`OpenCVCamera`: records opens/reads/closes, never touches a device."""

    fail_open: set[str] = set()
    log: list[tuple[str, str]] = []

    def __init__(self, spec: CameraSpec) -> None:
        self.spec = spec
        self._open = False

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        self.log.append(("open", self.spec.name))
        if self.spec.name in self.fail_open:
            raise OSError(f"camera {self.spec.name!r}: could not open {self.spec.index_or_path!r}")
        self._open = True

    def read(self) -> Any:
        if not self._open:
            raise RuntimeError(f"camera {self.spec.name!r}: not open")
        self.log.append(("read", self.spec.name))
        frame = np.zeros((self.spec.height or 4, self.spec.width or 4, 3), dtype=np.uint8)
        frame[..., 0] = 200  # red-leaning, so a reader can tell it from a blank
        return frame

    def close(self) -> None:
        self.log.append(("close", self.spec.name))
        self._open = False

    def describe(self) -> dict[str, Any]:
        return {"index_or_path": self.spec.index_or_path, "open": self._open}


@pytest.fixture(autouse=True)
def _fake_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cameras_mod, "OpenCVCamera", _RecordingCamera)
    _RecordingCamera.fail_open = set()
    _RecordingCamera.log = []


def _driver(**kwargs: Any) -> FeetechDriver:
    robot: Any = Robot("so101", mode="real", port="/dev/null", cameras=_CAMERAS, **kwargs)
    assert isinstance(robot, FeetechDriver), f"premise: the bare so101 call builds the native driver, got {type(robot)}"
    return robot


class TestTheFactoryHandsTheCamerasToTheDriver:
    def test_the_driver_declares_it_reads_cameras(self) -> None:
        """The opt-in :mod:`strands_robots.drivers.base` names, on the class."""
        assert FeetechDriver.reads_cameras is True

    def test_a_cameras_dict_is_accepted_on_the_bare_call(self) -> None:
        driver = _driver()
        assert camera_names(driver) == ["main"]
        assert camera_read_source(driver) is driver

    def test_nothing_opens_at_construction(self) -> None:
        """The arm opens first, on connect; a camera opened in the constructor would outlive a refused bus."""
        _driver()
        assert _RecordingCamera.log == []

    def test_an_unknown_camera_option_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="camera 'main': unknown option\\(s\\) \\['fsp'\\]"):
            Robot("so101", mode="real", port="/dev/null", cameras={"main": {"index_or_path": 1, "fsp": 30}})

    def test_a_non_opencv_type_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="type must be 'opencv'"):
            Robot("so101", mode="real", port="/dev/null", cameras={"main": {"type": "realsense", "index_or_path": 1}})


class TestConnectOpensThemAfterTheBus:
    def test_connect_reports_the_bus_and_opens_the_cameras(self) -> None:
        """``/dev/null`` is not a servo bus, so the bus refuses; the cameras open regardless."""
        driver = _driver()
        reason = driver.connect_eagerly()
        assert reason is not None, "premise: /dev/null is not a servo bus"
        assert not driver.is_connected
        assert ("open", "main") in _RecordingCamera.log
        assert driver.cameras["main"].is_open

    def test_a_camera_that_fails_costs_the_camera_not_the_arm(self) -> None:
        _RecordingCamera.fail_open = {"main"}
        driver = _driver()
        driver.connect_eagerly()
        assert "main" not in driver.cameras, "a camera that did not open is not offered as one that did"
        assert "main" in driver.camera_failures
        assert "could not open" in driver.camera_failures["main"]

    def test_status_reports_the_cameras_and_their_failures(self) -> None:
        _RecordingCamera.fail_open = {"main"}
        driver = _driver()
        driver.connect_eagerly()
        payload = asyncio.run(driver.get_status())["content"][0]["json"]
        assert payload["cameras"] == {}
        assert "main" in payload["camera_failures"]

    def test_cleanup_closes_what_connect_opened(self) -> None:
        driver = _driver()
        driver.connect_eagerly()
        driver.cleanup()
        assert ("close", "main") in _RecordingCamera.log
        assert driver.cameras == {}


class TestTheMeshReadsTheNativeShape:
    def test_frames_come_back_rgb_under_the_camera_name(self) -> None:
        driver = _driver()
        driver.connect_eagerly()
        frames = driver.camera_frames()
        assert set(frames) == {"main"}
        assert frames["main"].shape == (480, 640, 3)

    def test_the_seam_resolves_both_shapes(self) -> None:
        """A lerobot wrapper keeps cameras on the inner device; the native driver on itself."""

        class _Inner:
            cameras: dict[str, object] = {"wrist": object()}

            class config:
                cameras: dict[str, dict[str, object]] = {"wrist": {}}

        class _Wrapper:
            robot = _Inner()

        wrapper = _Wrapper()
        assert camera_read_source(wrapper) is wrapper.robot
        assert camera_names(wrapper) == ["wrist"]
        assert camera_read_source(object()) is None
        assert camera_names(object()) == []


class TestTheEntryGrader:
    @pytest.mark.parametrize(
        ("config", "fragment"),
        [
            ({"index_or_path": -1}, "non-negative"),
            ({"index_or_path": ""}, "non-empty path"),
            ({"index_or_path": True}, "device index"),
            ({"index_or_path": 0, "width": 0}, "width must be a positive integer"),
            ({"index_or_path": 0, "fps": float("nan")}, "fps must be a positive number"),
            ({}, "index_or_path is required"),
            ("not a mapping", "expected a mapping"),
        ],
    )
    def test_bad_entries_are_refused_with_the_camera_named(self, config: Any, fragment: str) -> None:
        reason = camera_entry_error("main", config)
        assert reason is not None and "main" in reason and fragment in reason, reason

    def test_an_int_beyond_float_range_is_graded_not_crashed_on(self) -> None:
        # The NaN clause is scoped to floats: ``math.isnan(10**400)`` raises
        # OverflowError, and the grader's contract is a refusal string or None.
        assert camera_entry_error("main", {"index_or_path": 0, "fps": 10**400}) is None

    def test_good_entries_become_specs(self) -> None:
        specs = camera_specs(
            {
                "main": {"index_or_path": 1, "fps": 30, "width": 640, "height": 480},
                "side": {"index_or_path": "/dev/video2"},
            }
        )
        assert specs["main"] == CameraSpec("main", 1, 640, 480, 30.0)
        assert specs["side"] == CameraSpec("side", "/dev/video2")
        assert camera_specs(None) == {}


class TestTheMeshPublishesTheNativeDriversCameras:
    """The two mesh seams that read cameras resolve the native shape as well as the wrapper's."""

    @staticmethod
    def _mesh(robot: Any) -> Any:
        from strands_robots.mesh.core import Mesh

        mesh = Mesh.__new__(Mesh)
        mesh.robot = robot
        mesh.peer_id = "so101-bench"
        mesh.peer_type = "so101"
        mesh._running = True
        return mesh

    @staticmethod
    def _presence(robot: Any) -> dict[str, Any]:
        """A presence payload from a real ``Mesh`` (never started, so no session)."""
        from strands_robots.mesh.core import Mesh

        return Mesh(robot, peer_id="so101-bench", peer_type="robot")._build_presence()

    def test_presence_names_the_cameras_and_the_connection(self) -> None:
        driver = _driver()
        driver.connect_eagerly()
        payload = self._presence(driver)
        assert payload["cameras"] == ["main"]
        assert payload["connected"] is False, "/dev/null never opened, and presence says so for a native driver too"
        assert "camera_failures" not in payload

    def test_presence_carries_a_camera_that_did_not_open(self) -> None:
        _RecordingCamera.fail_open = {"main"}
        driver = _driver()
        driver.connect_eagerly()
        payload = self._presence(driver)
        assert payload["cameras"] == ["main"], "a configured camera is announced even while it is down"
        assert "could not open" in payload["camera_failures"]["main"]

    def test_the_camera_tick_publishes_the_drivers_frames(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _driver()
        driver.connect_eagerly()
        mesh = self._mesh(driver)
        published: list[tuple[dict[str, Any], list[str]]] = []
        monkeypatch.setattr(mesh, "_encode_and_publish_frames", lambda obs, names: published.append((obs, names)))
        monkeypatch.setattr(
            mesh, "_publish_sim_cameras", lambda: pytest.fail("a native driver with cameras is not a sim peer")
        )
        monkeypatch.setattr("strands_robots.mesh._zenoh_config._bool_env", lambda *a, **k: False)
        mesh._publish_cameras_once()
        assert len(published) == 1
        obs, names = published[0]
        assert names == ["main"]
        assert obs["main"].shape == (480, 640, 3)

    def test_the_lerobot_wrapper_path_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The inner device's ``get_observation`` is still the first read, under the bus lock."""
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        class _Config:
            cameras: dict[str, dict[str, object]] = {"wrist": {}}

        class _Inner:
            is_connected = True
            config = _Config()
            cameras: dict[str, object] = {"wrist": object()}

            def get_observation(self) -> dict[str, Any]:
                return {"wrist": frame, "shoulder_pan.pos": 1.0}

        class _Wrapper:
            robot = _Inner()

        mesh = self._mesh(_Wrapper())
        published: list[tuple[dict[str, Any], list[str]]] = []
        monkeypatch.setattr(mesh, "_encode_and_publish_frames", lambda obs, names: published.append((obs, names)))
        monkeypatch.setattr("strands_robots.mesh._zenoh_config._bool_env", lambda *a, **k: False)
        mesh._publish_cameras_once()
        assert published == [({"wrist": frame, "shoulder_pan.pos": 1.0}, ["wrist"])]
