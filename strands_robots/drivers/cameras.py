"""Cameras for a native driver that declares ``reads_cameras = True``.

The factory hands a driver the caller's ``cameras=`` dict verbatim, so a driver
that opts in owns opening, reading and closing the devices in it (see
:mod:`strands_robots.drivers.base`). This module is that ownership, written
once: :func:`camera_specs` grades the dict, :class:`OpenCVCamera` is one device,
and :func:`open_cameras` opens a set so that a camera which fails to open costs
that camera and not the arm.

The accepted entry shape is the one the dashboard sends and lerobot's
``OpenCVCameraConfig`` declares, ``{name: {"index_or_path", "fps", "width",
"height"}}`` plus an optional ``type`` that must spell ``"opencv"``. A key
outside that set is refused by name rather than dropped, for the reason
:func:`strands_robots.hardware_robot._camera_option_vocabulary` gives: a
silently discarded option reports success while the camera streams at its
default. Frames are returned RGB, which is what the mesh camera publisher and
lerobot's own cameras hand out, so one encoder serves both driver families.

OpenCV ships with the package (``opencv-python-headless`` is a core
dependency), so no extra is named here; the import still goes through
:func:`~strands_robots.utils.require_optional` so a broken install is reported
with the line that repairs it rather than as a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from strands_robots.utils import refusal_repr, require_optional

logger = logging.getLogger(__name__)

#: The per-camera options a native driver reads. ``type`` is accepted as well,
#: and only as ``"opencv"``: a RealSense entry is refused here by name so the
#: caller learns this driver has no RealSense path, instead of an OpenCV grab on
#: an index that happens to exist.
CAMERA_KEYS: tuple[str, ...] = ("index_or_path", "fps", "width", "height")

#: The one backend this module opens.
CAMERA_TYPE = "opencv"

_TYPE_KEY = "type"

_OPENCV_PURPOSE = "native driver cameras"


@dataclass(frozen=True)
class CameraSpec:
    """One graded camera entry: where to open it and what to ask of it.

    Attributes:
        name: The key the caller registered the camera under; the name its
            frames publish under and every refusal quotes.
        index_or_path: An OpenCV device index or a device path/URL.
        width: Requested frame width in pixels, or ``None`` for the device's.
        height: Requested frame height in pixels, or ``None`` for the device's.
        fps: Requested frame rate, or ``None`` for the device's.
    """

    name: str
    index_or_path: int | str
    width: int | None = None
    height: int | None = None
    fps: float | None = None


def camera_entry_error(name: object, config: object) -> str | None:
    """Report why ``config`` is not a camera entry, or ``None`` when it is one.

    Args:
        name: The camera's key in the ``cameras=`` dict.
        config: The per-camera options the caller gave under that key.

    Returns:
        A reason naming the camera and the offending key or value, or ``None``.
    """
    if not isinstance(name, str) or not name.strip():
        return f"cameras: every key must be a non-empty camera name, got {refusal_repr(name)}"
    if not isinstance(config, Mapping):
        return f"camera {name!r}: expected a mapping of options, got {refusal_repr(config)}"
    unknown = sorted((str(key) for key in config if key not in CAMERA_KEYS and key != _TYPE_KEY), key=repr)
    if unknown:
        return (
            f"camera {name!r}: unknown option(s) {unknown}; a native driver reads "
            f"{list(CAMERA_KEYS)} (and {_TYPE_KEY!r}, which must be {CAMERA_TYPE!r})"
        )
    cam_type = config.get(_TYPE_KEY, CAMERA_TYPE)
    if cam_type != CAMERA_TYPE:
        return f"camera {name!r}: type must be {CAMERA_TYPE!r} on a native driver, got {refusal_repr(cam_type)}"
    if "index_or_path" not in config:
        return f"camera {name!r}: index_or_path is required (an OpenCV device index or a device path)"
    source = config["index_or_path"]
    if (
        isinstance(source, bool)
        or not isinstance(source, int | str)
        or (isinstance(source, str) and not source.strip())
    ):
        return f"camera {name!r}: index_or_path must be a device index or a non-empty path, got {refusal_repr(source)}"
    if isinstance(source, int) and source < 0:
        return f"camera {name!r}: index_or_path must be a non-negative device index, got {source}"
    for key in ("width", "height"):
        value = config.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return f"camera {name!r}: {key} must be a positive integer, got {refusal_repr(value)}"
    fps = config.get("fps")
    if fps is not None and (
        isinstance(fps, bool)
        or not isinstance(fps, int | float)
        or (isinstance(fps, float) and math.isnan(fps))
        or not fps > 0
    ):
        return f"camera {name!r}: fps must be a positive number, got {refusal_repr(fps)}"
    return None


def camera_specs(cameras: Mapping[Any, Any] | None) -> dict[str, CameraSpec]:
    """Grade a ``cameras=`` dict into specs, refusing the first bad entry by name.

    Args:
        cameras: The caller's dict, ``None`` or empty for no cameras.

    Returns:
        Specs keyed by camera name, in the caller's order.

    Raises:
        ValueError: With :func:`camera_entry_error`'s reason for the first
            entry that is not a camera entry, or when ``cameras`` is not a
            mapping at all.
    """
    if cameras is None:
        return {}
    if not isinstance(cameras, Mapping):
        raise ValueError(f"cameras: expected a mapping of camera name -> options, got {refusal_repr(cameras)}")
    specs: dict[str, CameraSpec] = {}
    for name, config in cameras.items():
        if (reason := camera_entry_error(name, config)) is not None:
            raise ValueError(reason)
        fps = config.get("fps")
        specs[str(name)] = CameraSpec(
            name=str(name),
            index_or_path=config["index_or_path"],
            width=config.get("width"),
            height=config.get("height"),
            fps=float(fps) if fps is not None else None,
        )
    return specs


class OpenCVCamera:
    """One camera behind ``cv2.VideoCapture``, returning RGB frames.

    Nothing opens in the constructor: :meth:`open` is the call that touches the
    device, so a spec can be built and reported before any hardware is reached,
    and a driver can open its arm first and its cameras after.
    """

    def __init__(self, spec: CameraSpec) -> None:
        self.spec = spec
        self._capture: Any | None = None
        self._cv2: Any | None = None

    @property
    def name(self) -> str:
        """The camera's key in the caller's dict."""
        return self.spec.name

    @property
    def is_open(self) -> bool:
        """Whether the device is open and answering."""
        capture = self._capture
        return capture is not None and bool(capture.isOpened())

    def open(self) -> None:
        """Open the device and apply the requested size and rate.

        Raises:
            OSError: When the device does not open or its first frame does not
                arrive; the message names the camera and its source so a
                dashboard can print it next to the arm that did connect.
        """
        if self.is_open:
            return
        cv2: Any = require_optional("cv2", pip_install="opencv-python-headless", purpose=_OPENCV_PURPOSE)
        self._cv2 = cv2
        capture = cv2.VideoCapture(self.spec.index_or_path)
        if not capture.isOpened():
            capture.release()
            raise OSError(
                f"camera {self.spec.name!r}: could not open {self.spec.index_or_path!r}; "
                "check the index with `strands_robots dashboard` > Devices, or the path"
            )
        if self.spec.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
        if self.spec.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
        if self.spec.fps is not None:
            capture.set(cv2.CAP_PROP_FPS, self.spec.fps)
        ok, _frame = capture.read()
        if not ok:
            capture.release()
            raise OSError(
                f"camera {self.spec.name!r}: opened {self.spec.index_or_path!r} but no frame arrived; "
                "another process may hold the device"
            )
        self._capture = capture

    def read(self) -> Any:
        """Return the newest frame as an RGB array.

        Raises:
            RuntimeError: When the camera is not open or a frame did not arrive.
        """
        capture = self._capture
        if capture is None or not capture.isOpened():
            raise RuntimeError(f"camera {self.spec.name!r}: not open; call open() first")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"camera {self.spec.name!r}: frame read failed on {self.spec.index_or_path!r}")
        cv2 = self._cv2
        if cv2 is not None and getattr(frame, "ndim", 0) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def close(self) -> None:
        """Release the device. Safe when never opened or already closed."""
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def describe(self) -> dict[str, Any]:
        """The camera as a status row: its spec and whether it is open."""
        return {
            "index_or_path": self.spec.index_or_path,
            "width": self.spec.width,
            "height": self.spec.height,
            "fps": self.spec.fps,
            "open": self.is_open,
        }


def open_cameras(specs: Mapping[str, CameraSpec]) -> tuple[dict[str, OpenCVCamera], dict[str, str]]:
    """Open every camera in ``specs``, keeping the ones that failed as reasons.

    A camera that does not open costs that camera, not the arm: the caller gets
    the cameras that did open and a reason per camera that did not, and decides
    what to report. Nothing raises here, so a driver's connect can open the bus
    first and never lose it to a camera.

    Args:
        specs: Graded camera specs, from :func:`camera_specs`.

    Returns:
        ``(opened, failed)``: open cameras by name, and failure reasons by name.
    """
    opened: dict[str, OpenCVCamera] = {}
    failed: dict[str, str] = {}
    for name, spec in specs.items():
        camera = OpenCVCamera(spec)
        try:
            camera.open()
        except (OSError, ImportError, RuntimeError) as exc:
            failed[name] = str(exc)
            logger.warning("camera %r did not open: %s", name, exc)
            continue
        opened[name] = camera
    return opened, failed


def camera_frames(cameras: Mapping[str, OpenCVCamera]) -> dict[str, Any]:
    """Read one frame from every open camera; a camera that fails is omitted.

    Args:
        cameras: Open cameras by name.

    Returns:
        RGB frames by camera name. A camera whose read fails is left out and
        logged at debug level, as the mesh publisher does for lerobot cameras,
        so one dead camera does not blank the others.
    """
    frames: dict[str, Any] = {}
    for name, camera in cameras.items():
        try:
            frames[name] = camera.read()
        except RuntimeError as exc:
            logger.debug("camera %r: %s", name, exc)
    return frames


def camera_read_source(robot: Any) -> Any | None:
    """The object whose ``cameras`` dict holds ``robot``'s camera devices, or ``None``.

    The camera twin of :func:`strands_robots.bus_access.joint_read_source`. A
    lerobot wrapper keeps its cameras on the inner device (``robot.robot``,
    whose ``cameras`` dict and ``config.cameras`` the mesh already reads); a
    native driver that declares ``reads_cameras`` keeps them on itself, as
    ``cameras`` (the open devices) and ``camera_specs`` (what was configured,
    open or not - so a camera that failed to open is still a name the robot
    announces, next to the failure). An inner device is preferred whenever one
    is present, so a wrapper is never read in place of the device it wraps.

    Args:
        robot: A lerobot wrapper, a native driver, or anything shaped like one.

    Returns:
        The device carrying a non-empty ``cameras`` dict, or ``None`` when
        ``robot`` has no cameras to read, which callers treat as "no cameras"
        and not as a failure.
    """
    device = getattr(robot, "robot", None)
    if device is None:
        device = robot
    for shape in (
        getattr(device, "cameras", None),
        getattr(getattr(device, "config", None), "cameras", None),
        getattr(device, "camera_specs", None),
    ):
        if isinstance(shape, Mapping) and shape:
            return device
    return None


def camera_names(robot: Any) -> list[str]:
    """The camera names ``robot`` publishes, from whichever shape carries them.

    Args:
        robot: A lerobot wrapper, a native driver, or anything shaped like one.

    Returns:
        Camera names in declaration order; empty when there are none.
    """
    device = camera_read_source(robot)
    return [] if device is None else device_camera_names(device)


def device_camera_names(device: Any) -> list[str]:
    """The camera names on an already-resolved camera device.

    Args:
        device: What :func:`camera_read_source` returned - read as it is, never
            unwrapped again, so a device that happens to answer ``robot`` is not
            mistaken for a wrapper.

    Returns:
        Camera names in declaration order; empty when there are none.
    """
    for shape in (
        getattr(getattr(device, "config", None), "cameras", None),
        getattr(device, "camera_specs", None),
        getattr(device, "cameras", None),
    ):
        if isinstance(shape, Mapping) and shape:
            return [str(name) for name in shape]
    return []
