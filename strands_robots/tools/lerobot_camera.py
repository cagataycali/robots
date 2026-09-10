#!/usr/bin/env python3
"""
LeRobot-based camera tool for Strands agents.
Leverages LeRobot's OpenCV and RealSense camera classes for professional camera management.

Every span this tool measures - a connect time, a per-frame capture time, a
recording's achieved duration - is a duration, so it is measured on
``time.monotonic()`` and its base carries that clock in its name
(``..._started_mono``). ``time.time()`` is not a clock but the current opinion
about the date, and an NTP correction, a ``date -s`` or a resume from suspend
landing inside one of these windows subtracts the step from the span. The
performance test then reads that span as a verdict about the camera (``Est.
FPS``, ``Fast``/``Slow``, ``Good``/``Slow``), so a corrected clock is reported
as a device measurement. The absolute stamps this tool writes - a filename's
date, a report's ``Timestamp`` line - are the other half of that boundary and
stay on ``datetime.now()``.

RealSense support needs the Intel SDK (``pyrealsense2``) in addition to lerobot,
which is what ``REALSENSE_AVAILABLE`` reports; importing lerobot's RealSense
camera classes does not establish it, because lerobot requires the SDK at its
call sites rather than at import. Every surface that reports the SDK absent
names the same install, :data:`REALSENSE_SDK_ABSENT`.
"""

import importlib.util
import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import numpy as np

try:
    import cv2
    from lerobot.cameras.camera import Camera
    from lerobot.cameras.opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import ColorMode, Cv2Rotation, OpenCVCameraConfig

    # Try to import RealSense camera if available
    try:
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        # These modules import whether or not the Intel SDK is installed: lerobot
        # binds ``pyrealsense2`` to None behind an availability flag and requires
        # it at the call sites instead. So the import succeeding says the camera
        # classes exist, not that a RealSense camera can be opened - that is what
        # the SDK decides, and it is the question every use of this flag asks. It
        # is probed under the import name, which is the one both the
        # ``pyrealsense2`` and the ``pyrealsense2-macosx`` distribution provide.
        REALSENSE_AVAILABLE = importlib.util.find_spec("pyrealsense2") is not None
    except ImportError:
        REALSENSE_AVAILABLE = False
        RealSenseCamera = None
        RealSenseCameraConfig = None

except ImportError as e:
    raise ImportError(f"LeRobot camera modules not available: {e}")

from strands import tool

from strands_robots.tools._path_validation import resolve_output_path, validate_save_path
from strands_robots.utils import (
    boolean_flag_error,
    positive_finite_number_error,
    positive_whole_number_error,
)

# The one remedy for an absent RealSense SDK, so every surface that reports it
# reports the same install. It names lerobot's ``intelrealsense`` extra rather
# than the ``pyrealsense2`` distribution because the extra is what carries the
# per-platform split - on macOS the wheel ships as ``pyrealsense2-macosx``, so a
# bare ``pip install pyrealsense2`` there installs nothing that can be imported.
REALSENSE_SDK_ABSENT = (
    "The Intel RealSense SDK (pyrealsense2) is not installed, so RealSense "
    "cameras cannot be opened. Install with: pip install 'lerobot[intelrealsense]'"
)

logger = logging.getLogger(__name__)


def _frame_to_image_content(frame: np.ndarray, format: str = "jpg") -> dict[str, Any]:
    """Convert a numpy frame to image content format for Converse API."""
    try:
        # Convert RGB to BGR for OpenCV encoding
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            bgr_frame = frame

        # Encode frame to specified format
        if format.lower() in ["jpg", "jpeg"]:
            success, encoded_img = cv2.imencode(".jpg", bgr_frame)
            image_format = "jpeg"
        elif format.lower() == "png":
            success, encoded_img = cv2.imencode(".png", bgr_frame)
            image_format = "png"
        else:
            success, encoded_img = cv2.imencode(".jpg", bgr_frame)  # Default to JPEG
            image_format = "jpeg"

        if not success:
            raise ValueError("Failed to encode frame")

        # Convert to bytes
        image_bytes = encoded_img.tobytes()

        return {"image": {"format": image_format, "source": {"bytes": image_bytes}}}

    except Exception as e:
        logger.error(f"Failed to convert frame to image content: {e}")
        return {"text": f"Failed to encode image: {str(e)}"}


# Which numeric options each action actually consumes. Every action that opens a
# camera is configured with the caller's geometry, so width/height/fps are
# effective for all of them; each duration knob drives exactly one loop, and
# "discover"/"list" open no camera with caller-supplied geometry at all. Every
# handler that selects the asynchronous read hands the caller's timeout_ms to it,
# so each of those actions carries that row.
#: Accepted ``color_mode`` spellings, each mapped to the LeRobot enum member it
#: selects. This dict is the single owner of that vocabulary: both
#: :func:`_create_camera`, which resolves a spelling to an enum, and
#: :func:`_vocabulary_option_error`, which refuses a spelling that resolves to
#: nothing, read it, so the accepted set and the enforced set cannot diverge.
_COLOR_MODES: dict[str, ColorMode] = {
    "RGB": ColorMode.RGB,
    "BGR": ColorMode.BGR,
}

#: Accepted ``rotation`` spellings, mapped to the LeRobot enum member each
#: selects. Same single-owner contract as :data:`_COLOR_MODES`.
_ROTATIONS: dict[str, Cv2Rotation] = {
    "NO_ROTATION": Cv2Rotation.NO_ROTATION,
    "ROTATE_90": Cv2Rotation.ROTATE_90,
    "ROTATE_180": Cv2Rotation.ROTATE_180,
    "ROTATE_270": Cv2Rotation.ROTATE_270,
}

_ACTION_NUMERIC_OPTIONS: dict[str, tuple[str, ...]] = {
    "capture": ("width", "height", "fps", "timeout_ms"),
    "capture_batch": ("width", "height", "fps", "timeout_ms"),
    "record": ("width", "height", "fps", "capture_duration", "timeout_ms"),
    "preview": ("width", "height", "fps", "preview_duration", "timeout_ms"),
    "test": ("width", "height", "fps", "timeout_ms"),
    "configure": ("width", "height", "fps"),
}


_ACTION_POSTURE_FLAGS: dict[str, tuple[str, ...]] = {
    "capture": ("async_mode", "warmup"),
    "capture_batch": ("async_mode", "warmup"),
    "record": ("async_mode", "warmup"),
    "preview": ("async_mode", "warmup"),
    "test": ("async_mode", "warmup"),
    "configure": ("save_config", "warmup"),
}


def _posture_flag_error(action: str, *, async_mode: Any, warmup: Any, save_config: Any) -> str | None:
    """Error text for the first posture flag ``action`` consumes but cannot read.

    These three select a *posture* rather than scaling a quantity, and each was
    read by truthiness - so every non-empty string, the spellings a caller
    reaches for when opting out included, selected the affirmative posture.
    Measured on ``eecaa80`` against a recording camera stand-in:

    * ``save_config="false"`` wrote the configuration file, so a caller who
      spelled the opt-out got a durable artifact on disk under
      ``status="success"``;
    * ``warmup="false"`` was handed to ``Camera.connect`` as the string, was
      persisted into that file as ``"warmup": "false"`` - a string in the one
      field of that document declared a boolean, beside the integer geometry and
      rate that are validated - and was reported on the line above it as
      ``Warmup: on``, so the file, the report and the caller disagree three ways
      about one posture;
    * ``async_mode="false"`` selected the asynchronous read path, which the
      plain boolean ``False`` does not: one ``async_read`` and no synchronous
      read, against zero and one.

    It runs ahead of :func:`_numeric_option_error` because that guard's
    ``timeout_ms`` row is *gated* on this flag - the synchronous read consumes no
    budget, so an option no handler reads is not refused. Reading the gate by
    truthiness switched the row off from outside its own table: ``async_mode=0``
    with ``timeout_ms=-5`` was answered ``status="success"``, an unusable budget
    accepted because a falsy value that is not a declared spelling of *off*
    discarded the row. Ordered the other way the refusal also names the wrong
    parameter - ``async_mode="false", timeout_ms=-5`` reported ``capture:
    timeout_ms must be > 0``, sending the caller to correct a budget whose only
    problem was the flag that selected it.

    Keyed by action, and holding only the flags each handler is actually passed,
    for the reason the numeric table is: ``discover`` and ``list`` take none of
    the three, so a value neither consults is not refused. The domain itself
    belongs to neither surface, so it delegates to
    :func:`~strands_robots.utils.boolean_flag_error` - the one owner the sibling
    ``lerobot_train`` builder already consults - exactly as the numeric rows
    delegate their spans and counts. What stays here is the
    roster and the report order, which puts ``save_config`` ahead of ``warmup``
    so the flag that writes a file is named before the one that only opens a
    camera.

    Args:
        action: The requested action; decides which flags are effective.
        async_mode: Whether the asynchronous read path is selected, as supplied.
        warmup: Whether the camera is warmed on connection, as supplied.
        save_config: Whether the configuration is written to a file, as supplied.

    Returns:
        An error message naming the action and the flag, or ``None`` when every
        flag this action reads is a boolean.
    """
    supplied = {"async_mode": async_mode, "warmup": warmup, "save_config": save_config}
    for param in _ACTION_POSTURE_FLAGS.get(action, ()):
        if error := boolean_flag_error(supplied[param], param, action):
            return error
    return None


def _numeric_option_error(
    action: str,
    *,
    width: Any,
    height: Any,
    fps: Any,
    capture_duration: Any,
    preview_duration: Any,
    timeout_ms: Any,
    async_mode: Any,
) -> str | None:
    """Error text for the first numeric option ``action`` consumes but cannot honor.

    Every value here is agent-supplied, so each is checked against the shared
    domain for its kind before a camera is opened: ``width`` / ``height`` / ``fps``
    count pixels and frames
    (:func:`~strands_robots.utils.positive_whole_number_error`, which already owns
    the recorders' geometry and rate), while the two durations and ``timeout_ms``
    are continuous spans of time
    (:func:`~strands_robots.utils.positive_finite_number_error`). Reusing those
    helpers is what keeps this tool from accepting a frame size or a rate that the
    plain-MP4 recorders refuse.

    Validating here rather than letting the camera driver object is what makes the
    refusal a property of the request instead of the device: the driver compares
    the rate it was asked for against the rate the attached camera reports, so its
    complaint names an ``actual_fps`` and is only raised once the device has been
    opened and reconfigured. A rate of ``0`` is impossible on every camera, and the
    tool has its own stake in these values regardless of the device - ``fps`` is
    written into the MP4 container as its timebase and is the divisor of the
    preview's frame period.

    ``timeout_ms`` is only effective under ``async_mode``: the synchronous read
    takes no timeout, so a value it never consumes is not refused. That makes the
    flag a gate on this table, so it is a boolean by the time it is read here -
    :func:`_posture_flag_error` runs first, and a truthy spelling of *off* can no
    longer switch the row off from outside the table.

    **The frame count is a product, so one factor's sign does not decide whether a
    frame can be captured.** ``positive_finite_number_error`` reads
    ``capture_duration`` alone, and at the default ``fps=30`` every span below
    ``0.0333`` is positive, finite, and makes the recording loop's bound
    ``int(fps * capture_duration) == 0`` - the loop body never runs, and the tool
    reports ``status="success"`` with ``Frames: 0`` while its ``Saved:`` line names
    the same 258-byte MP4 that no decoder will open. Which side of the line a span
    falls on is not a property of the span, so the rate is read with it. Refused
    rather than floored to one frame: a recording that cannot be honored as asked
    is a caller error, not a value to silently substitute. ``preview_duration`` is
    deliberately not paired with ``fps`` this way - the preview is bounded by a
    ``time.monotonic()`` deadline whose first iteration always runs, so a short
    preview displays a frame rather than none.

    Args:
        action: The requested action; decides which options are effective.
        width: Frame width in pixels, as supplied.
        height: Frame height in pixels, as supplied.
        fps: Frame rate, as supplied.
        capture_duration: Recording span in seconds, as supplied.
        preview_duration: Preview span in seconds, as supplied.
        timeout_ms: Asynchronous read budget in milliseconds, as supplied.
        async_mode: Whether the asynchronous read path is selected.

    Returns:
        An error message naming the action and the option, or ``None`` when every
        option this action reads is usable.
    """
    consumed = set(_ACTION_NUMERIC_OPTIONS.get(action, ()))
    if not async_mode:
        consumed.discard("timeout_ms")
    for param, value, check in (
        ("width", width, positive_whole_number_error),
        ("height", height, positive_whole_number_error),
        ("fps", fps, positive_whole_number_error),
        ("capture_duration", capture_duration, positive_finite_number_error),
        ("preview_duration", preview_duration, positive_finite_number_error),
        ("timeout_ms", timeout_ms, positive_finite_number_error),
    ):
        if param in consumed:
            error = check(value, param, action)
            if error:
                return error
    # Reaching here means every option this action consumes is individually
    # usable, which is what lets the pair be judged together. Keyed on the two
    # names being consumed rather than on ``action == "record"``, so an action
    # that later gains a capture span is covered without a second edit.
    # Compared as floats rather than through the consumer's ``int(...)``: both
    # factors are bounded only by the float64 range, so their product can be
    # ``inf``, and ``int(inf)`` raises out of the guard that exists so a frame
    # count never raises. For a non-negative product the tests are equivalent -
    # ``int(p) < 1`` iff ``p < 1``.
    if {"fps", "capture_duration"} <= consumed and float(fps) * float(capture_duration) < 1.0:
        return (
            f"{action}: capture_duration={float(capture_duration):g} at fps={int(fps)} records 0 frames, "
            f"so the file would contain no video. Raise capture_duration to at least "
            f"{1.0 / float(fps):g}, or lower fps."
        )
    return None


# Which enumerated vocabularies each action actually consumes. Every action that
# opens a camera hands the caller's ``color_mode`` and ``rotation`` to
# :func:`_create_camera`, so each carries both rows; "discover" and "list" probe
# devices without applying either.
_ACTION_VOCABULARY_OPTIONS: dict[str, tuple[str, ...]] = {
    "capture": ("color_mode", "rotation"),
    "capture_batch": ("color_mode", "rotation"),
    "record": ("color_mode", "rotation"),
    "preview": ("color_mode", "rotation"),
    "test": ("color_mode", "rotation"),
    "configure": ("color_mode", "rotation"),
}


def _vocabulary_option_error(action: str, *, color_mode: Any, rotation: Any) -> str | None:
    """Error text for the first enumerated option ``action`` reads but cannot honor.

    The enum-valued counterpart to :func:`_numeric_option_error`, and it exists for
    the same reason that one records: every value here is agent-supplied, and the
    check belongs in front of the camera rather than behind it. Unlike a frame rate,
    though, neither of these is ever refused downstream - each names a closed
    vocabulary that :func:`_create_camera` resolves by lookup, and an unrecognised
    spelling resolved to a *plausible neighbour* instead of failing: anything other
    than ``RGB`` selected ``BGR``, and any rotation the map did not name selected
    ``NO_ROTATION``. Both substitutions reported success.

    That is what makes the substitution expensive rather than merely wrong. A
    ``color_mode`` the driver reads as ``BGR`` is delivered in that channel order,
    and the capture path converts the frame it is handed with ``COLOR_RGB2BGR``
    unconditionally, so the saved image has its red and blue channels transposed -
    a photograph of a red object is written blue - while the result says
    "Image Capture Success". A ``rotation`` that resolves to ``NO_ROTATION`` is the
    quieter half: the frame is simply not rotated, so the caller receives a correct
    image of the wrong orientation. Neither is visible in the tool result, and a
    trailing space or a transposed pair of letters is enough to reach either.

    Both spellings are compared case-insensitively, matching how
    :func:`_create_camera` resolves them, so every value that worked before still
    works: this refuses only the spellings that used to be silently replaced.

    Only the options ``action`` actually reads are checked, so a caller is never
    refused for a value the requested action ignores.

    Args:
        action: The requested action; decides which options are effective.
        color_mode: Channel order selector, as supplied.
        rotation: Frame rotation selector, as supplied.

    Returns:
        An error message naming the action, the option and the accepted
        vocabulary, or ``None`` when every option this action reads is usable.
    """
    consumed = _ACTION_VOCABULARY_OPTIONS.get(action, ())
    for param, value, vocabulary in (
        ("color_mode", color_mode, _COLOR_MODES),
        ("rotation", rotation, _ROTATIONS),
    ):
        if param not in consumed:
            continue
        # Total over any input: a non-string is refused here rather than reaching
        # ``.upper()``, where it would raise past the tool's structured result.
        if not isinstance(value, str) or value.upper() not in vocabulary:
            accepted = ", ".join(vocabulary)
            return (
                f"{action}: {param} must be one of {accepted} (compared "
                f"case-insensitively), got {value!r}. An unrecognised value used to "
                f"select a different one silently."
            )
    return None


# The cameras ``capture_batch`` reads when the caller names none. Held in one
# place so the documented default and the resolution below cannot drift apart.
_DEFAULT_BATCH_CAMERA_IDS: tuple[int | str, ...] = (0, "/dev/video4")


def _camera_ids_error(camera_ids: Any) -> str | None:
    """Error text when ``camera_ids`` is not a usable selection of cameras.

    ``camera_ids`` SELECTS the cameras one ``capture_batch`` call opens, and a
    selection is read by membership, never by truthiness: ``None`` is the one
    spelling of "the default robot cameras", so it is the caller's to skip and
    never reaches here. Every other value is graded.

    Read by truthiness, ``[]`` took the same branch as ``None`` and was widened
    to the two default cameras, so a caller who selected no camera - which is
    what a filter that matched nothing produces - had two devices opened and two
    files written, under ``status="success"`` and a "2/2 cameras" summary
    quoting a count the caller never asked for. The empty selection is refused
    rather than widened, which is the verdict the shared name-list domain
    (:func:`strands_robots.utils.name_list_error`) reserves for the caller. That
    domain is not reused here because a camera id is legitimately an ``int``
    index as well as a device-path string, and it accepts names only.

    The other shapes fail the same way that domain describes. A bare string is
    iterable per character, so ``"/dev/video4"`` opened eleven one-character
    cameras on eleven threads and reported eleven verdicts about devices the
    caller never named, instead of one about the parameter. A ``Mapping`` is
    iterable over its keys, so its values were discarded. A repeated id opens
    one device twice concurrently and writes two files for it. A one-shot
    iterator is consumed by the ``len()`` that sizes the thread pool before the
    loop that submits work reads it. A ``bool`` is an ``int`` subclass, so
    ``True`` selected camera index 1 without the caller writing a 1.

    Every refusal here precedes the save directory being created, the thread
    pool being built and any camera being opened, so a refused selection has no
    partial effect to undo.

    Args:
        camera_ids: The caller-supplied selection, anything but ``None``.

    Returns:
        An error message naming the shape and the accepted one, or ``None`` when
        the selection can be honored as written.
    """
    prefix = "capture_batch: camera_ids"
    accepted = (
        "a list of distinct camera ids, each an int index or a device path string; "
        "omit it (None) for the default robot cameras"
    )
    if isinstance(camera_ids, str):
        return (
            f"{prefix} must be {accepted}, not a single string ({camera_ids!r}). A "
            f"string is read one camera per character, so pass [{camera_ids!r}] to "
            f"name one camera."
        )
    if isinstance(camera_ids, bytes):
        return f"{prefix} must be {accepted}, not bytes ({camera_ids!r})."
    if isinstance(camera_ids, Mapping):
        return f"{prefix} must be {accepted}, not a mapping - its values would be discarded."
    if not isinstance(camera_ids, Sequence):
        return (
            f"{prefix} must be {accepted}, got {type(camera_ids).__name__}. A one-shot "
            f"iterator is consumed before the cameras are opened."
        )
    ids = list(camera_ids)
    if not ids:
        return (
            f"{prefix}=[] selects no camera, so there is nothing to capture. Omit "
            f"camera_ids to select the default robot cameras "
            f"{list(_DEFAULT_BATCH_CAMERA_IDS)!r}, or name the cameras to capture from."
        )
    for i, cam_id in enumerate(ids):
        if isinstance(cam_id, bool) or not isinstance(cam_id, int | str):
            return (
                f"{prefix}[{i}] must be an int index or a device path string, got {cam_id!r} ({type(cam_id).__name__})."
            )
        if isinstance(cam_id, str) and not cam_id.strip():
            return f"{prefix}[{i}] is blank ({cam_id!r}); a camera path must name a device."
    seen: set[int | str] = set()
    for cam_id in ids:
        if cam_id in seen:
            return (
                f"{prefix} names {cam_id!r} more than once ({ids!r}); each camera is "
                f"opened once per batch, so name each id once."
            )
        seen.add(cam_id)
    return None


@tool
def lerobot_camera(
    action: str = "list",
    camera_type: str = "opencv",
    camera_id: int | str | None = None,
    save_path: str = "./lerobot_captures",
    filename: str | None = None,
    camera_ids: list[int | str] | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    color_mode: str = "RGB",
    rotation: str = "NO_ROTATION",
    format: str = "jpg",
    capture_duration: float = 5.0,
    preview_duration: float = 10.0,
    async_mode: bool = False,
    timeout_ms: float = 1000,
    warmup: bool = True,
    save_config: bool = False,
) -> dict[str, Any]:
    """Advanced LeRobot-based camera tool for professional camera management.

    Args:
        action: Action to perform
            - "discover": Discover all available cameras (OpenCV + RealSense)
            - "list": List camera details and configurations
            - "capture": Capture single image from camera
            - "capture_batch": Capture from multiple cameras simultaneously
            - "record": Record video sequence from camera
            - "preview": Show live preview from camera
            - "test": Test camera functionality and performance
            - "configure": Configure camera settings and save
        camera_type: Camera type ("opencv" or "realsense"). "realsense" needs
            the Intel SDK installed on top of lerobot; without it the action is
            refused naming that install, rather than reported as unsupported.
        camera_id: Camera device ID (int for index, str for path like "/dev/video0")
        save_path: Directory to save captured images/videos
        filename: Custom filename (without extension). Resolved inside
            save_path; a value naming a location outside it is refused rather
            than written there.
        camera_ids: Cameras to capture from in one capture_batch call - a list
            of distinct ids, each an int index or a device path string. Omit it
            for the default robot cameras. An empty list selects no camera and
            is refused rather than widened to those defaults; a single id passed
            as a bare string is refused rather than read one camera per character.
        width: Frame width in pixels (a positive whole number)
        height: Frame height in pixels (a positive whole number)
        fps: Frames per second (a positive whole number)
        color_mode: Color mode; one of "RGB" or "BGR", compared case-insensitively.
            Any other value is refused rather than silently read as "BGR".
        rotation: Image rotation; one of "NO_ROTATION", "ROTATE_90", "ROTATE_180"
            or "ROTATE_270", compared case-insensitively. Any other value is
            refused rather than silently read as "NO_ROTATION".
        format: Image format ("jpg", "png", "bmp"). Becomes the saved file's
            extension, so like filename it is resolved inside save_path and
            refused if it names a location outside it.
        capture_duration: Duration for video recording (positive seconds)
        preview_duration: Duration for preview display (positive seconds)
        async_mode: Use async reading for better performance. A boolean; it
            selects a read path rather than scaling one, so any other value is
            refused rather than read as its opposite.
        timeout_ms: Timeout for async operations (positive milliseconds; read only when async_mode is on)
        warmup: Enable camera warmup on connection. A boolean, refused rather
            than read by truthiness - it is also recorded in the saved
            configuration, so a non-boolean would persist there.
        save_config: Save camera configuration to file. A boolean, refused
            rather than read by truthiness: it writes a file, so a truthy
            spelling of off would leave one behind.

    Returns:
        Dict containing status and detailed camera operation results
    """

    try:
        # Ahead of the numeric guard: that guard's ``timeout_ms`` row is gated on
        # ``async_mode``, so the gate is checked before it decides a row.
        posture_error = _posture_flag_error(action, async_mode=async_mode, warmup=warmup, save_config=save_config)
        if posture_error:
            return {"status": "error", "content": [{"text": posture_error}]}
        numeric_error = _numeric_option_error(
            action,
            width=width,
            height=height,
            fps=fps,
            capture_duration=capture_duration,
            preview_duration=preview_duration,
            timeout_ms=timeout_ms,
            async_mode=async_mode,
        )
        if numeric_error:
            return {"status": "error", "content": [{"text": numeric_error}]}
        vocabulary_error = _vocabulary_option_error(action, color_mode=color_mode, rotation=rotation)
        if vocabulary_error:
            return {"status": "error", "content": [{"text": vocabulary_error}]}
        if action in _ACTION_NUMERIC_OPTIONS:
            # Accepted above as integral values; coerce so the camera config's
            # declared int fields, the VideoWriter frame size and the progress
            # modulo each receive a true int rather than an integral float.
            width, height, fps = int(width), int(height), int(fps)

        if action == "discover":
            return _discover_cameras()
        elif action == "list":
            return _list_camera_details(camera_type, camera_id)
        elif action == "capture":
            if camera_id is None:
                return {
                    "status": "error",
                    "content": [{"text": "camera_id required for capture action"}],
                }
            return _capture_single_image(
                camera_type,
                camera_id,
                save_path,
                filename or "",
                width,
                height,
                fps,
                color_mode,
                rotation,
                format,
                async_mode,
                timeout_ms,
                warmup,
            )
        elif action == "capture_batch":
            # Read ``is None``: camera_ids selects a SUBSET of the cameras, so
            # only the absent spelling means the default robot cameras. An empty
            # selection, a bare string, a mapping or a repeat is refused here,
            # before the save directory, the thread pool or any camera exists.
            if camera_ids is None:
                camera_ids = list(_DEFAULT_BATCH_CAMERA_IDS)
            elif selection_error := _camera_ids_error(camera_ids):
                return {"status": "error", "content": [{"text": selection_error}]}
            return _capture_batch_images(
                camera_type,
                camera_ids,
                save_path,
                filename or "",
                width,
                height,
                fps,
                color_mode,
                rotation,
                format,
                async_mode,
                timeout_ms,
                warmup,
            )
        elif action == "record":
            if camera_id is None:
                return {
                    "status": "error",
                    "content": [{"text": "camera_id required for record action"}],
                }
            return _record_video_sequence(
                camera_type,
                camera_id,
                save_path,
                filename or "",
                width,
                height,
                fps,
                color_mode,
                rotation,
                capture_duration,
                async_mode,
                timeout_ms,
                warmup,
            )
        elif action == "preview":
            if camera_id is None:
                return {
                    "status": "error",
                    "content": [{"text": "camera_id required for preview action"}],
                }
            return _preview_camera_live(
                camera_type,
                camera_id,
                width,
                height,
                fps,
                color_mode,
                rotation,
                preview_duration,
                async_mode,
                timeout_ms,
                warmup,
            )
        elif action == "test":
            if camera_id is None:
                return {
                    "status": "error",
                    "content": [{"text": "camera_id required for test action"}],
                }
            return _test_camera_performance(
                camera_type,
                camera_id,
                width,
                height,
                fps,
                color_mode,
                rotation,
                async_mode,
                timeout_ms,
                warmup,
            )
        elif action == "configure":
            if camera_id is None:
                return {
                    "status": "error",
                    "content": [{"text": "camera_id required for configure action"}],
                }
            return _configure_camera_settings(
                camera_type,
                camera_id,
                width,
                height,
                fps,
                color_mode,
                rotation,
                save_path,
                save_config,
                warmup,
            )
        else:
            return {
                "status": "error",
                "content": [{"text": f"Unknown action: {action}"}],
            }

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Camera operation failed: {str(e)}"}],
        }


def _discover_cameras() -> dict[str, Any]:
    """Discover all available cameras using LeRobot's detection methods."""
    try:
        # Discover OpenCV cameras
        opencv_cameras = OpenCVCamera.find_cameras()

        # Discover RealSense cameras if available
        realsense_cameras = []
        if REALSENSE_AVAILABLE:
            try:
                realsense_cameras = RealSenseCamera.find_cameras()
            except Exception as e:
                logger.warning(f"RealSense camera discovery failed: {e}")

        total_cameras = len(opencv_cameras) + len(realsense_cameras)

        # Format discovery results
        discovery_info = []
        discovery_info.append(" **Camera Discovery Results**\n")

        if opencv_cameras:
            discovery_info.append(" **OpenCV Cameras:**")
            for i, cam in enumerate(opencv_cameras):
                profile = cam.get("default_stream_profile", {})
                discovery_info.append(
                    f"  - **{cam.get('name', 'Unknown')}**\n"
                    f"    - ID: `{cam.get('id', 'N/A')}`\n"
                    f"    - Backend: {cam.get('backend_api', 'N/A')}\n"
                    f"    - Resolution: {profile.get('width', '?')}x{profile.get('height', '?')}\n"
                    f"    - FPS: {profile.get('fps', '?')}\n"
                    f"    - Format: {profile.get('format', '?')}"
                )
            discovery_info.append("")

        if realsense_cameras:
            discovery_info.append(" **RealSense Cameras:**")
            for i, cam in enumerate(realsense_cameras):
                discovery_info.append(
                    f"  - **{cam.get('name', 'Unknown')}**\n"
                    f"    - Serial: `{cam.get('serial_number', 'N/A')}`\n"
                    f"    - Type: {cam.get('type', 'N/A')}"
                )
            discovery_info.append("")

        if total_cameras == 0:
            discovery_info.append(" **No cameras detected**")
        else:
            discovery_info.append(f"**Total: {total_cameras} cameras found**")
            discovery_info.append(f"   - OpenCV: {len(opencv_cameras)}")
            discovery_info.append(f"   - RealSense: {len(realsense_cameras)}")

        return {"status": "success", "content": [{"text": "\n".join(discovery_info)}]}

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Camera discovery failed: {str(e)}"}],
        }


def _list_camera_details(camera_type: str, camera_id: int | str | None = None) -> dict[str, Any]:
    """List detailed camera information and configurations."""
    try:
        details = []
        details.append(" **Camera Configuration Details**\n")

        if camera_type.lower() == "opencv":
            details.append(" **OpenCV Camera System:**")
            details.append(f"   - Backend: {_get_opencv_backend_name()}")
            details.append(f"   - Version: {cv2.__version__}")
            details.append("   - Available color modes: RGB, BGR")
            details.append("   - Supported rotations: 0, 90, 180, 270 degrees")
            details.append("   - Async reading:  Supported")
            details.append("")

            if camera_id is not None:
                try:
                    config = OpenCVCameraConfig(index_or_path=camera_id, fps=30, width=640, height=480)
                    camera = OpenCVCamera(config)
                    camera.connect(warmup=False)

                    details.append(f"**Camera {camera_id} Details:**")
                    details.append("   - Connection:  Success")
                    details.append(f"   - Actual FPS: {camera.fps}")
                    details.append(f"   - Resolution: {camera.width}x{camera.height}")
                    details.append(f"   - Color Mode: {camera.color_mode.value}")

                    camera.disconnect()

                except Exception as e:
                    details.append(f"**Camera {camera_id} Details:**")
                    details.append(f"   - Connection:  Failed ({str(e)})")

        elif camera_type.lower() == "realsense" and REALSENSE_AVAILABLE:
            details.append(" **RealSense Camera System:**")
            details.append("   - SDK Available:  Yes")
            details.append("   - Depth Support:  Yes")
            details.append("   - Multiple streams: Color, Depth, Infrared")
            details.append("   - Advanced features: Post-processing, alignment")

        else:
            if not REALSENSE_AVAILABLE and camera_type.lower() == "realsense":
                details.append(" **RealSense Camera System:**")
                details.append("   - SDK Available:  Not installed")
                details.append(f"   - {REALSENSE_SDK_ABSENT}")
            else:
                details.append(f"**Unknown camera type: {camera_type}**")

        return {"status": "success", "content": [{"text": "\n".join(details)}]}

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Camera details failed: {str(e)}"}],
        }


def _capture_single_image(
    camera_type: str,
    camera_id: int | str,
    save_path: str,
    filename: str,
    width: int,
    height: int,
    fps: int,
    color_mode: str,
    rotation: str,
    format: str,
    async_mode: bool,
    timeout_ms: float,
    warmup: bool,
) -> dict[str, Any]:
    """Capture a single image using LeRobot camera system."""
    try:
        # Validate save path before any filesystem operations
        save_path = validate_save_path(save_path, label="save_path")

        # Create save directory
        os.makedirs(save_path, exist_ok=True)

        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cam_name = str(camera_id).replace("/dev/", "").replace("/", "_")
            filename = f"lerobot_{camera_type}_{cam_name}_{timestamp}"

        file_path = resolve_output_path(save_path, f"{filename}.{format}", label="filename and format")

        # Create camera configuration
        camera = _create_camera(camera_type, camera_id, width, height, fps, color_mode, rotation)

        # Connect and capture
        connect_started_mono = time.monotonic()
        camera.connect(warmup=warmup)
        connect_time = time.monotonic() - connect_started_mono

        capture_started_mono = time.monotonic()
        if async_mode:
            frame = camera.async_read(timeout_ms=timeout_ms)
        else:
            frame = camera.read()
        capture_time = time.monotonic() - capture_started_mono

        # Save image
        success = cv2.imwrite(file_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        camera.disconnect()

        if not success:
            return {
                "status": "error",
                "content": [{"text": f"Failed to save image: {file_path}"}],
            }

        # Get image info
        img_height, img_width = frame.shape[:2]
        file_size = os.path.getsize(file_path)

        result_info = [
            " **Image Capture Success!**",
            f"Camera: {camera_type.upper()} @ {camera_id}",
            f"Saved: `{file_path}`",
            f"Resolution: {img_width}x{img_height}",
            f"File size: {file_size:,} bytes",
            f"Connect time: {connect_time:.3f}s",
            f"Capture time: {capture_time:.3f}s",
            f"Async mode: {'on' if async_mode else 'off'}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        # Create image content for Converse API
        image_content = _frame_to_image_content(frame, format)

        return {
            "status": "success",
            "content": [{"text": "\n".join(result_info)}, image_content],
        }

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Image capture failed: {str(e)}"}],
        }


def _capture_batch_images(
    camera_type: str,
    camera_ids: list[int | str],
    save_path: str,
    filename: str,
    width: int,
    height: int,
    fps: int,
    color_mode: str,
    rotation: str,
    format: str,
    async_mode: bool,
    timeout_ms: float,
    warmup: bool,
) -> dict[str, Any]:
    """Capture images from multiple cameras simultaneously."""
    try:
        save_path = validate_save_path(save_path, label="save_path")
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = []
        successful_captures = 0
        batch_started_mono = time.monotonic()

        def capture_single_camera(cam_id):
            try:
                # Generate unique filename for this camera
                cam_name = str(cam_id).replace("/dev/", "").replace("/", "_")
                if filename:
                    cam_filename = f"{filename}_{cam_name}_{timestamp}"
                else:
                    cam_filename = f"batch_{camera_type}_{cam_name}_{timestamp}"

                file_path = resolve_output_path(save_path, f"{cam_filename}.{format}", label="filename and format")

                # Create and use camera
                camera = _create_camera(camera_type, cam_id, width, height, fps, color_mode, rotation)

                capture_started_mono = time.monotonic()
                camera.connect(warmup=warmup)

                if async_mode:
                    frame = camera.async_read(timeout_ms=timeout_ms)
                else:
                    frame = camera.read()

                capture_time = time.monotonic() - capture_started_mono

                # Save image
                success = cv2.imwrite(file_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                camera.disconnect()

                if success:
                    file_size = os.path.getsize(file_path)
                    return {
                        "camera_id": cam_id,
                        "status": "success",
                        "file_path": file_path,
                        "file_size": file_size,
                        "capture_time": capture_time,
                        "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                        "frame": frame,  # Include frame for image content
                    }
                else:
                    return {
                        "camera_id": cam_id,
                        "status": "error",
                        "message": "Failed to save image",
                    }

            except Exception as e:
                return {"camera_id": cam_id, "status": "error", "message": str(e)}

        # Use ThreadPoolExecutor for parallel capture
        with ThreadPoolExecutor(max_workers=len(camera_ids)) as executor:
            future_to_camera = {executor.submit(capture_single_camera, cam_id): cam_id for cam_id in camera_ids}

            for future in as_completed(future_to_camera):
                result = future.result()
                results.append(result)
                if result["status"] == "success":
                    successful_captures += 1

        total_time = time.monotonic() - batch_started_mono

        # Format results and prepare content list
        result_info = [" **Batch Camera Capture Results:**", ""]
        content_list = []

        for result in results:
            if result["status"] == "success":
                result_info.append(
                    f"**{result['camera_id']}**: {result['resolution']} "
                    f"({result['file_size']:,} bytes, {result['capture_time']:.3f}s)"
                )
                # Add image content if frame is available
                if "frame" in result:
                    image_content = _frame_to_image_content(result["frame"], format)
                    content_list.append(image_content)
            else:
                result_info.append(f"**{result['camera_id']}**: {result['message']}")

        result_info.extend(
            [
                "",
                " **Summary:**",
                f"   - Success: {successful_captures}/{len(camera_ids)} cameras",
                f"   - Total time: {total_time:.3f}s",
                f"   - Save path: `{save_path}`",
                f"   - Async mode: {'on' if async_mode else 'off'}",
            ]
        )

        # Add text summary first, then all images
        final_content = [{"text": "\n".join(result_info)}] + content_list

        return {
            "status": "success" if successful_captures > 0 else "error",
            "content": final_content,
        }

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Batch capture failed: {str(e)}"}],
        }


def _record_video_sequence(
    camera_type: str,
    camera_id: int | str,
    save_path: str,
    filename: str,
    width: int,
    height: int,
    fps: int,
    color_mode: str,
    rotation: str,
    capture_duration: float,
    async_mode: bool,
    timeout_ms: float,
    warmup: bool,
) -> dict[str, Any]:
    """Record a video sequence from camera."""
    try:
        save_path = validate_save_path(save_path, label="save_path")
        os.makedirs(save_path, exist_ok=True)

        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cam_name = str(camera_id).replace("/dev/", "").replace("/", "_")
            filename = f"lerobot_video_{camera_type}_{cam_name}_{timestamp}"

        video_path = resolve_output_path(save_path, f"{filename}.mp4", label="filename")

        # Create camera
        camera = _create_camera(camera_type, camera_id, width, height, fps, color_mode, rotation)
        camera.connect(warmup=warmup)

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]  # cv2 stubs incomplete
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        frames_captured = 0
        started_mono = time.monotonic()
        target_frames = int(fps * capture_duration)

        try:
            while frames_captured < target_frames:
                if async_mode:
                    frame = camera.async_read(timeout_ms=timeout_ms)
                else:
                    frame = camera.read()

                # Convert RGB to BGR for video writer
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)
                frames_captured += 1

                # Progress update every second
                if frames_captured % fps == 0:
                    elapsed = time.monotonic() - started_mono
                    remaining = capture_duration - elapsed
                    print(f"Recording... {elapsed:.1f}s / {capture_duration:.1f}s ({remaining:.1f}s remaining)")

        finally:
            video_writer.release()
            camera.disconnect()

        actual_duration = time.monotonic() - started_mono
        file_size = os.path.getsize(video_path)

        result_info = [
            " **Video Recording Complete!**",
            f"Camera: {camera_type.upper()} @ {camera_id}",
            f"Saved: `{video_path}`",
            f"Resolution: {width}x{height}",
            f"Frames: {frames_captured} @ {fps} FPS",
            f"Duration: {actual_duration:.2f}s (target: {capture_duration:.2f}s)",
            f"File size: {file_size:,} bytes",
            f"Async mode: {'on' if async_mode else 'off'}",
            f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        return {"status": "success", "content": [{"text": "\n".join(result_info)}]}

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Video recording failed: {str(e)}"}],
        }


def _preview_camera_live(
    camera_type: str,
    camera_id: int | str,
    width: int,
    height: int,
    fps: int,
    color_mode: str,
    rotation: str,
    preview_duration: float,
    async_mode: bool,
    timeout_ms: float,
    warmup: bool,
) -> dict[str, Any]:
    """Show live preview from camera."""
    try:
        camera = _create_camera(camera_type, camera_id, width, height, fps, color_mode, rotation)
        camera.connect(warmup=warmup)

        frames_displayed = 0
        # Durations, so they are measured on time.monotonic(): a wall-clock step
        # would cut the preview short, mis-report the FPS window, or skew the
        # reported duration.
        start_time = time.monotonic()
        fps_counter_start = time.monotonic()
        fps_frame_count = 0

        print(f"Starting live preview from {camera_type.upper()} camera {camera_id}")
        print(f"Duration: {preview_duration}s | Press 'q' to quit early")

        try:
            while time.monotonic() - start_time < preview_duration:
                frame_start = time.monotonic()

                if async_mode:
                    frame = camera.async_read(timeout_ms=timeout_ms)
                else:
                    frame = camera.read()

                # Convert RGB to BGR for display
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Add info overlay
                info_text = f"Camera: {camera_id} | Frame: {frames_displayed} | FPS: {fps}"
                cv2.putText(
                    bgr_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(f"LeRobot Camera Preview - {camera_id}", bgr_frame)

                frames_displayed += 1
                fps_frame_count += 1

                # Calculate and display FPS every second
                if time.monotonic() - fps_counter_start >= 1.0:
                    actual_fps = fps_frame_count / (time.monotonic() - fps_counter_start)
                    print(f"Live FPS: {actual_fps:.1f} | Frames: {frames_displayed}")
                    fps_counter_start = time.monotonic()
                    fps_frame_count = 0

                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print(" Preview stopped by user")
                    break

                # Maintain target FPS
                frame_time = time.monotonic() - frame_start
                target_frame_time = 1.0 / fps
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)

        finally:
            cv2.destroyAllWindows()
            camera.disconnect()

        actual_duration = time.monotonic() - start_time
        avg_fps = frames_displayed / actual_duration if actual_duration > 0 else 0

        result_info = [
            " **Live Preview Complete!**",
            f"Camera: {camera_type.upper()} @ {camera_id}",
            f"Resolution: {width}x{height}",
            f"Frames displayed: {frames_displayed}",
            f"Duration: {actual_duration:.2f}s",
            f"Average FPS: {avg_fps:.2f}",
            f"Target FPS: {fps}",
            f"Async mode: {'on' if async_mode else 'off'}",
        ]

        return {"status": "success", "content": [{"text": "\n".join(result_info)}]}

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Preview failed: {str(e)}"}],
        }


def _test_camera_performance(
    camera_type: str,
    camera_id: int | str,
    width: int,
    height: int,
    fps: int,
    color_mode: str,
    rotation: str,
    async_mode: bool,
    timeout_ms: float,
    warmup: bool,
) -> dict[str, Any]:
    """Test camera performance and capabilities."""
    try:
        test_results = []
        test_results.append(" **Camera Performance Test**\n")

        # Connection test
        connect_started_mono = time.monotonic()
        camera = _create_camera(camera_type, camera_id, width, height, fps, color_mode, rotation)
        camera.connect(warmup=warmup)
        connect_time = time.monotonic() - connect_started_mono

        test_results.append(f"**Connection Test**: {connect_time:.3f}s")

        # Frame capture test (sync)
        capture_times = []
        for i in range(10):
            read_started_mono = time.monotonic()
            frame = camera.read()
            capture_time = time.monotonic() - read_started_mono
            capture_times.append(capture_time)

        avg_sync_time = np.mean(capture_times)
        min_sync_time = np.min(capture_times)
        max_sync_time = np.max(capture_times)

        test_results.append(" **Sync Capture (10 frames)**:")
        test_results.append(f"   - Average: {avg_sync_time:.3f}s")
        test_results.append(f"   - Min: {min_sync_time:.3f}s")
        test_results.append(f"   - Max: {max_sync_time:.3f}s")
        test_results.append(f"   - Est. FPS: {1 / avg_sync_time:.1f}")

        # Frame capture test (async)
        if async_mode:
            async_times = []
            for i in range(10):
                read_started_mono = time.monotonic()
                frame = camera.async_read(timeout_ms=timeout_ms)
                async_time = time.monotonic() - read_started_mono
                async_times.append(async_time)

            avg_async_time = np.mean(async_times)
            min_async_time = np.min(async_times)
            max_async_time = np.max(async_times)

            test_results.append(" **Async Capture (10 frames)**:")
            test_results.append(f"   - Average: {avg_async_time:.3f}s")
            test_results.append(f"   - Min: {min_async_time:.3f}s")
            test_results.append(f"   - Max: {max_async_time:.3f}s")
            test_results.append(f"   - Est. FPS: {1 / avg_async_time:.1f}")
            test_results.append(f"   - Speedup: {avg_sync_time / avg_async_time:.2f}x")

        # Frame properties test
        test_results.append(" **Frame Properties**:")
        test_results.append(f"   - Resolution: {frame.shape[1]}x{frame.shape[0]}")
        test_results.append(f"   - Channels: {frame.shape[2]}")
        test_results.append(f"   - Data type: {frame.dtype}")
        test_results.append(f"   - Memory size: {frame.nbytes:,} bytes")

        # Camera properties
        if hasattr(camera, "fps"):
            test_results.append("**Camera Configuration**:")
            test_results.append(f"   - Configured FPS: {camera.fps}")
            test_results.append(f"   - Resolution: {camera.width}x{camera.height}")
            test_results.append(f"   - Color mode: {camera.color_mode.value}")

        camera.disconnect()

        test_results.append("\n **Performance Summary**:")
        test_results.append(f"   - Connection: {'Fast' if connect_time < 1.0 else 'Slow'} ({connect_time:.3f}s)")
        test_results.append(f"   - Sync capture: {'Good' if avg_sync_time < 0.1 else 'Slow'} ({avg_sync_time:.3f}s)")
        if async_mode:
            test_results.append(
                f"   - Async capture: {'Better' if avg_async_time < avg_sync_time else 'Worse'} ({avg_async_time:.3f}s)"
            )
        test_results.append(f"   - Frame rate: {'Stable' if max_sync_time - min_sync_time < 0.05 else 'Variable'}")

        return {"status": "success", "content": [{"text": "\n".join(test_results)}]}

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Performance test failed: {str(e)}"}],
        }


def _configure_camera_settings(
    camera_type: str,
    camera_id: int | str,
    width: int,
    height: int,
    fps: int,
    color_mode: str,
    rotation: str,
    save_path: str,
    save_config: bool,
    warmup: bool,
) -> dict[str, Any]:
    """Configure camera settings and optionally save configuration."""
    try:
        camera = _create_camera(camera_type, camera_id, width, height, fps, color_mode, rotation)
        camera.connect(warmup=warmup)

        # Get actual camera properties
        actual_config = {
            "camera_type": camera_type,
            "camera_id": camera_id,
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps,
            "color_mode": camera.color_mode.value,
            "warmup": warmup,
            "timestamp": datetime.now().isoformat(),
        }

        if hasattr(camera, "rotation") and camera.rotation is not None:
            actual_config["rotation"] = rotation

        config_info = [
            "**Camera Configuration**",
            f"Camera: {camera_type.upper()} @ {camera_id}",
            f"Resolution: {actual_config['width']}x{actual_config['height']}",
            f"FPS: {actual_config['fps']}",
            f"Color mode: {actual_config['color_mode']}",
            f"Rotation: {actual_config.get('rotation', 'NO_ROTATION')}",
            f"Warmup: {'on' if warmup else 'off'}",
        ]

        # Save configuration if requested
        if save_config:
            save_path = validate_save_path(save_path, label="save_path")
            os.makedirs(save_path, exist_ok=True)
            cam_id_safe = str(camera_id).replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_filename = f"camera_config_{camera_type}_{cam_id_safe}_{timestamp}.json"
            config_path = os.path.join(save_path, config_filename)

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(actual_config, f, indent=2)

            config_info.extend(
                [
                    "",
                    " **Configuration Saved**:",
                    f"   - File: `{config_path}`",
                    "   - Format: JSON",
                ]
            )

        camera.disconnect()

        return {"status": "success", "content": [{"text": "\n".join(config_info)}]}

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Configuration failed: {str(e)}"}],
        }


def _create_camera(
    camera_type: str,
    camera_id: int | str,
    width: int,
    height: int,
    fps: int,
    color_mode: str,
    rotation: str,
) -> Camera:
    """Create and configure a camera instance."""

    if camera_type.lower() == "opencv":
        # Resolve the string selectors through the module-level vocabularies, which
        # are the same maps the dispatcher validates against. The fallbacks are
        # unreachable for a value that came through the tool - it is refused by
        # then - and keep this helper total for a direct caller.
        color_mode_enum = _COLOR_MODES.get(color_mode.upper(), ColorMode.BGR)
        rotation_enum = _ROTATIONS.get(rotation.upper(), Cv2Rotation.NO_ROTATION)

        config = OpenCVCameraConfig(
            index_or_path=camera_id,
            fps=fps,
            width=width,
            height=height,
            color_mode=color_mode_enum,
            rotation=rotation_enum,
        )
        return OpenCVCamera(config)

    elif camera_type.lower() == "realsense":
        # "realsense" is a supported type on every platform this package runs
        # on, so an absent SDK is reported as the absent SDK. Falling through to
        # the unsupported-type refusal below would answer a question the caller
        # did not ask, and send them looking for a spelling that does not exist.
        if not REALSENSE_AVAILABLE:
            raise ImportError(REALSENSE_SDK_ABSENT, name="pyrealsense2")
        config = RealSenseCameraConfig(serial_number_or_name=str(camera_id), fps=fps, width=width, height=height)
        return RealSenseCamera(config)

    else:
        raise ValueError(f"Unsupported camera type: {camera_type}")


def _get_opencv_backend_name() -> str:
    """Get the name of the current OpenCV backend."""
    backend = cv2.CAP_ANY
    backend_names = {
        cv2.CAP_V4L2: "V4L2",
        cv2.CAP_MSMF: "MSMF",
        cv2.CAP_AVFOUNDATION: "AVFoundation",
        cv2.CAP_ANY: "Auto",
    }
    return backend_names.get(backend, "Unknown")
