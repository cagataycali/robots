"""Backend-agnostic LeRobotDataset recording lifecycle.

The :class:`DatasetRecordingMixin` holds the parts of the recording workflow
that have no engine-specific physics dependency: querying recording state,
flushing an episode boundary, stopping/finalizing a session, streaming a
dataset back, and reporting status. Every method operates purely through the
state mapping returned by :meth:`DatasetRecordingMixin._recording_state` (by
default the shared ``self._world._backend_state`` dict) and the
backend-agnostic :class:`~strands_robots.dataset_recorder.DatasetRecorder`, so
the MuJoCo, Newton, and Isaac backends all mix it in unchanged.

Each backend supplies the engine-specific half separately:

* ``start_recording`` - declares the dataset schema (joint names + cameras)
  from the live scene, which requires reading the engine's model.
* ``_make_run_policy_hook`` - captures per-step observations/cameras and feeds
  them to the active recorder.

**Coupling**: every method operates on the mutable state mapping returned by
:meth:`DatasetRecordingMixin._recording_state`. The default accessor reaches
into ``self._world._backend_state`` (the ``SimWorld`` contract the MuJoCo and
Newton backends share); backends whose world object is not a ``SimWorld``
(Isaac Sim's ``self._world`` is the Isaac ``World`` handle) override the
accessor to supply their own dict instead of forking the mixin. The
``TYPE_CHECKING`` stub documents the default contract for mypy; it is not an
enforceable protocol.
"""

import logging
import math
import numbers
import shutil
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strands_robots.dataset_transfer import sync_dataset_to_bucket
from strands_robots.utils import (
    boolean_flag_error,
    camera_schema_key,
    is_boolean,
    positive_whole_number_error,
)

logger = logging.getLogger(__name__)


def recorded_cameras_line(
    joint_names: Sequence[str],
    recorded_cameras: Mapping[str, str],
    scene_cameras: Sequence[str],
    cameras: Sequence[str] | None,
    fps: float,
) -> str:
    """The schema line of ``start_recording``'s reply, naming the cameras.

    ``"6 joints, 1 cameras @ 10fps"`` told an agent how many image columns the
    dataset has but not which: a replayed agent read it, then asked ``render``
    for ``top_camera`` - the name it assumed the recording used - and got "not
    found. Available: ['default']". The count is replaced by the cameras
    themselves, so the names an agent needs next are in the reply that created
    the columns.

    The name it lists is the SCENE camera name, because that is the one every
    camera surface accepts: ``render``, ``render_depth``, ``get_frame``,
    ``get_camera_params`` and ``start_cameras_recording`` resolve a name
    through the compiled model (or a registered alias), and ``cameras=`` on the
    next ``start_recording`` takes either spelling. The dataset column is named
    by :func:`~strands_robots.utils.camera_schema_key`, which collapses the
    ``/`` of a robot-scoped camera to ``__`` because a LeRobot feature name
    cannot carry it - so listing the column key instead would hand back
    ``so101__wrist`` for a camera the render surfaces only answer for as
    ``so101/wrist``, reproducing the very refusal above one namespace deeper.
    Both names matter (the column is what a policy's ``input_features`` and a
    dataset reader see), so when they differ the column is named too rather
    than chosen between.

    When no camera is recorded, a second line says what the dataset WILL carry
    (joint state and actions, no ``observation.images.*``) and why - which is
    read off the scene rather than assumed, because the three causes have three
    different remedies: a scene with no camera needs ``add_camera(...)``, a
    ``cameras=`` that scoped every camera out needs a different subset, and a
    scene whose cameras produce no frame (Isaac's ``render_mode="headless"``)
    needs neither and would be sent on a false errand by either.

    Args:
        joint_names: The scalar joint columns of ``observation.state``.
        recorded_cameras: Scene camera name -> dataset column key, in dataset
            column order. Empty when the dataset carries no image column.
        scene_cameras: Every camera the scene offers, recorded or not, which is
            what distinguishes "there are none" from "none was recorded".
        cameras: The caller's ``cameras=`` argument, ``None`` when it recorded
            every camera. Only its presence is read - an empty selection is the
            one cause the scene alone cannot explain.
        fps: The dataset frame rate, as reported. Not the caller's raw value:
            :func:`dataset_recording_option_error` has already refused anything
            but a positive whole number inside the range of a 64-bit float, so
            this renders a number rather than an arbitrary object.

    Returns:
        The schema line, newline-terminated, plus the no-camera line when there
        is one.
    """
    names = list(recorded_cameras)
    word = "camera" if len(names) == 1 else "cameras"
    line = f"{len(joint_names)} joints, {len(names)} {word} {names} @ {fps}fps\n"
    if names:
        renamed = [(name, key) for name, key in recorded_cameras.items() if name != key]
        if renamed:
            columns = ", ".join(f"{name!r} -> observation.images.{key}" for name, key in renamed)
            line += (
                f"Dataset columns: {columns} (a dataset feature name cannot carry '/', so it "
                "collapses to '__'; render and cameras= take the scene name above).\n"
            )
        return line
    if not scene_cameras:
        return line + (
            "No cameras in the scene: the dataset carries joint state and actions only, "
            "no observation.images.*. Call add_camera(...) before start_recording to record "
            "images.\n"
        )
    if cameras is not None:
        return line + (
            f"No cameras recorded (cameras= scoped out {sorted(scene_cameras)}): the dataset "
            "carries joint state and actions only, no observation.images.*. Omit cameras= or "
            "name the ones to keep.\n"
        )
    return line + (
        f"No cameras recorded: the scene's camera(s) {sorted(scene_cameras)} produce no frame to "
        "record, so the dataset carries joint state and actions only, no observation.images.* "
        "(the reason is logged as a warning above).\n"
    )


def resumed_dataset_line(*, episodes: int, frames: int) -> str:
    """The line ``start_recording``'s reply adds when it resumed a dataset.

    ``start_recording`` read exactly the same on a fresh dataset and on one it
    appended to, so nothing told an agent recording demonstrations in a loop
    that the episodes it was about to capture would join episodes already on
    disk - nor how to start over instead. Every backend that resumes says so
    through this one wording, because the counts named here are the counts
    :meth:`DatasetRecordingMixin.stop_recording` measures the session against.

    Args:
        episodes: Episodes already in the dataset being resumed.
        frames: Frames already in the dataset being resumed.

    Returns:
        One newline-terminated line, for interpolation into the reply.
    """
    return (
        f"Resuming the existing dataset ({episodes} episode(s), {frames} frames); "
        "this session's episodes are appended. Pass overwrite=True to record "
        "from scratch instead.\n"
    )


def dataset_recording_option_error(method: str, fps: Any) -> dict[str, Any] | None:
    """Reject a LeRobotDataset recording option no dataset can be written at.

    Pre-flight guard shared by every backend's ``start_recording`` (MuJoCo,
    Newton, Isaac), so the three surfaces cannot disagree on what a usable
    ``fps`` is. ``fps`` is a frame count per second, so the accepted domain is
    the shared one the plain-MP4 recorders and the ``run_policy(video=...)``
    dict already enforce
    (:func:`~strands_robots.utils.positive_whole_number_error`):
    a positive whole number.

    The ``fps`` reaching this guard is forwarded to
    :meth:`~strands_robots.dataset_recorder.DatasetRecorder.create` unchanged, and
    that method applies the same shared domain to its direct callers - so this is
    the envelope-returning half of one rule, not a rule of its own. Any narrowing
    belongs in the shared domain rather than here: a value refused only deeper
    would raise ``ValueError`` out of a ``start_recording`` that had already
    reported it usable.

    Without this guard an unusable ``fps`` was reported as ``status="success"``
    and then cost the caller the episode: LeRobot only rejects ``fps <= 0``, so
    a fractional ``2.7`` or a ``nan`` created the dataset, killed the video
    encoder thread on the first frame and aborted the rollout ("on_frame hook
    failed 5 times in a row"), after which ``stop_recording`` could not save the
    pending frames; ``fps=True`` silently recorded a 1 fps dataset (an ``int``
    subclass acting as a 1); and ``fps="30"`` dead-ended in a raw
    ``TypeError: '<=' not supported between instances of 'str' and 'int'``
    instead of naming the parameter.

    Args:
        method: Public method name, used to prefix the error message.
        fps: Caller-supplied dataset frame rate.

    Returns:
        A structured ``{"status": "error", ...}`` dict naming ``fps``, or
        ``None`` when the value is usable.
    """
    if text := positive_whole_number_error(fps, "fps", method):
        return {"status": "error", "content": [{"text": text}]}
    return None


def dataset_recording_posture_error(method: str, param: str, value: Any) -> dict[str, Any] | None:
    """Reject a recording posture flag that was not supplied as a boolean.

    Sibling of :func:`dataset_recording_option_error` for the two flags in the
    same ``start_recording`` signature that select a *posture* rather than
    scaling a quantity, and for the ``push_to_hub`` that
    :meth:`DatasetRecordingMixin.stop_recording` accepts as an override. Shared
    by every backend (MuJoCo, Newton, Isaac) for the same reason the ``fps``
    guard is: the three surfaces must not disagree on what a usable value is.

    The domain is :func:`~strands_robots.utils.boolean_flag_error` - the one the
    mesh provisioning entry points and ``lerobot_teleoperate``'s execution flags
    already apply to their own posture flags. Read by truthiness instead, both
    flags fail toward the branch the caller was opting *out* of, because every
    non-empty string is truthy:

    * ``overwrite="false"`` (also ``"no"``, ``"off"``, ``"0"``, ``1``, ``nan``)
      reached :meth:`DatasetRecordingMixin._prepare_dataset_target` as True and
      deleted the caller's dataset with ``shutil.rmtree``. That method already refuses to
      clobber a non-empty *non*-dataset directory, so the one path it deleted
      without asking was a real LeRobotDataset - the recorded episodes the caller
      meant to append to. ``start_recording`` returned ``status="success"``.
    * ``push_to_hub="false"`` (same spellings) was stashed on the recording state
      and *published* the finished dataset to the Hub at ``stop_recording``.

    Neither could be honoured as written, and neither failure is recoverable:
    the episodes are gone, and an upload cannot be taken back.

    Args:
        method: Public method name, used to prefix the error message.
        param: Flag name, for the message.
        value: The flag as supplied.

    Returns:
        A structured ``{"status": "error", ...}`` dict naming *param*, or
        ``None`` when the value is a boolean and can be honoured.
    """
    if text := boolean_flag_error(value, param, method):
        return {"status": "error", "content": [{"text": text}]}
    return None


def _schema_key_collisions(camera_names: Iterable[str]) -> dict[str, list[str]]:
    """Group the camera names that collapse to one :func:`~strands_robots.utils.camera_schema_key`.

    Blank names are ignored: a backend skips an unnamed camera, so it names
    neither a dataset column nor a file.

    Args:
        camera_names: Scene camera names, in the backend's own order.

    Returns:
        ``{the shared key: the names that collapse to it}``, sorted by key and
        holding only the keys more than one name claims.
    """
    groups: dict[str, list[str]] = {}
    for name in camera_names:
        if not name:
            continue
        groups.setdefault(camera_schema_key(name), []).append(name)
    return {key: members for key, members in sorted(groups.items()) if len(members) > 1}


def camera_schema_key_collision_error(method: str, camera_names: Iterable[str]) -> dict[str, Any] | None:
    """Error envelope when two scene cameras share one dataset feature name.

    A dataset column is named by :func:`~strands_robots.utils.camera_schema_key`,
    which collapses a camera's ``/`` namespace separator to ``__``. That mapping
    is not injective, so ``arm0/wrist`` and ``arm0__wrist`` are two cameras in
    the scene and one column in the dataset. Nothing downstream can tell which
    of them a column was declared for, and the three ways of asking each fail
    differently:

    * recording every camera declares the key twice, which
      :meth:`~strands_robots.dataset_recorder.DatasetRecorder.create` refuses as
      a repeated ``camera_keys`` entry - a true refusal, but it reads as the
      caller having named one camera twice rather than as two cameras sharing a
      name;
    * ``cameras=`` naming both drops one without saying so (the requested names
      are distinct, so they pass :func:`~strands_robots.utils.name_list_error`,
      and only their keys collide) and the surviving column carries the other
      camera's frames. When the two render at different sizes the mismatch
      surfaces as the FIRST ``add_frame`` being rejected, after
      ``overwrite=True`` has already wiped the dataset being replaced;
    * ``cameras=`` naming one of them succeeds, but which camera lands under the
      key depends on the spelling used - the same
      ``observation.images.arm0__wrist`` column is the robot's wrist view when
      asked for as ``arm0/wrist`` and the other camera when asked for as
      ``arm0__wrist``.

    So the ambiguity belongs to the scene, not to one way of recording it, and
    it is refused once here - before any dataset is created, resumed or wiped -
    rather than in each of the three. This is the same rule
    :func:`~strands_robots.utils.name_list_error` applies to a literally
    repeated name, read through the collapse: two entries naming one column
    declare fewer columns than were asked for.

    Args:
        method: The public method name, used to prefix the message.
        camera_names: The scene's camera names, in the backend's own order.
            Blank names are ignored - a backend skips an unnamed camera when it
            declares the schema, so they are not columns and cannot collide.

    Returns:
        A tool-style error envelope, or ``None`` when every name has a distinct
        key.
    """
    collisions = _schema_key_collisions(camera_names)
    if not collisions:
        return None
    described = "; ".join(f"{key!r} <- {sorted(members)}" for key, members in collisions.items())
    return {
        "status": "error",
        "content": [
            {
                "text": (
                    f"{method}: these scene cameras do not have distinct dataset feature names: "
                    f"{described}. A LeRobot feature name cannot contain '/' (it addresses nested "
                    "features), so a camera's namespace separator is recorded as '__' - which makes "
                    "these names one column. Whichever camera won it, the column would be named "
                    "after the other one, and if they render at different sizes the first frame is "
                    "rejected and the episode is lost. Rename one of them (add_camera(name=...)) so "
                    "the names still differ once '/' becomes '__', then record again."
                )
            }
        ],
    }


def camera_clip_name_collision_error(method: str, camera_names: Iterable[str]) -> dict[str, Any] | None:
    """Error envelope when two cameras would write to one MP4 clip.

    A raw camera recording writes one clip per camera into ``output_dir``, and a
    camera's ``/`` namespace separator cannot survive into a file name - there
    it names a directory, so ``arm0/wrist`` would put the clip one level below
    the directory the caller asked for, under a name the recording tag is
    missing from. The separator is therefore written as ``__``
    (:func:`~strands_robots.utils.camera_schema_key`, the same collapse that
    names the dataset column for that camera, so a clip and its
    ``observation.images.*`` feature are spelled alike).

    That mapping is not injective: ``arm0/wrist`` and ``arm0__wrist`` are two
    cameras in the scene and one file on disk, so one camera's clip would
    overwrite the other's while the result reported both as written. It is
    refused as the recording starts, before a frame is captured - where
    :func:`camera_schema_key_collision_error` refuses the same pair of names for
    the dataset sink.

    Args:
        method: The public method name, used to prefix the message.
        camera_names: The cameras this recording will capture.

    Returns:
        A tool-style error envelope, or ``None`` when every camera names its own
        clip.
    """
    collisions = _schema_key_collisions(camera_names)
    if not collisions:
        return None
    described = "; ".join(f"{key!r} <- {sorted(members)}" for key, members in collisions.items())
    return {
        "status": "error",
        "content": [
            {
                "text": (
                    f"{method}: these cameras do not name distinct MP4 clips: {described}. A "
                    "camera's '/' namespace separator cannot be part of a file name (it names a "
                    "directory), so it is written as '__' - which makes these names one file, and "
                    "the clip flushed first would be overwritten by the other. Rename one of them "
                    "(add_camera(name=...)) or pass cameras=[...] naming only one of them, then "
                    "record again."
                )
            }
        ],
    }


def encoder_absent_flush_refusal(reason: object, name: str, buffered: Mapping[str, int]) -> dict[str, Any]:
    """The refusal a raw-camera flush owes buffers whose encoder is not installed.

    A ``stop_cameras_recording`` flush is best-effort and never-raises, so every
    other way an encode can fail is folded into its success envelope and the
    recording is deregistered. The absent encoder is the one exception, and it is
    the reason this refusal exists rather than being one more artifact line:
    :func:`~strands_robots.rendering.require_clip_encoder` raises before any
    writer is opened, so NO camera was written and NO buffer was touched. The
    frames are all still in memory, and the remedy - install the encoder, call
    the verb again - is only followable while the recording is still registered.

    Deregistering there discarded exactly the frames the message promised were
    recoverable. So the envelope and the retention are one decision, and this is
    the one place that words it: both raw-camera recorders (MuJoCo's daemon-
    thread capture and Isaac's ``on_frame`` capture) return it, which is what
    keeps the two from drifting into different answers about the same absence.

    Args:
        reason: The encoder's own refusal - an ``ImportError`` raised through
            :func:`~strands_robots.utils.require_optional`. It is quoted rather
            than re-diagnosed: ``imageio`` and the MP4 plugin it leaves optional
            are two different absences, so a fixed line names the wrong module
            half the time and prescribes an install that changes nothing.
        name: The registered recording's tag, echoed so a caller holding several
            knows which one is still resident.
        buffered: Frames held per camera, which is what the caller loses by not
            following the remedy.

    Returns:
        A tool-style error envelope whose ``json`` carries ``stopped=False``,
        ``recording`` and ``buffered_frames`` - the state a caller needs to
        decide, without parsing the message.
    """
    held = dict(buffered)
    return {
        "status": "error",
        "content": [
            {
                "text": (
                    f"{reason}\n"
                    f"Nothing was encoded and nothing was dropped: camera recording "
                    f"{name!r} is left registered holding {held}. Install "
                    f"the encoder and call stop_cameras_recording() again to flush it."
                )
            },
            {
                "json": {
                    "stopped": False,
                    "recording": name,
                    "buffered_frames": held,
                }
            },
        ],
    }


def recorder_dataset_fps(recorder: Any) -> int | None:
    """Read the frame rate of a live recorder's dataset, or None if unavailable.

    ``LeRobotDataset`` exposes ``fps`` directly and via ``meta.fps``; both are
    probed so a layout that only carries the metadata object still compares.

    Only a positive WHOLE rate is reported. A dataset whose rate is fractional
    cannot be recorded at any rate ``start_recording`` accepts (it requires a
    positive whole number), so there is no value to advise the caller to pass
    and the comparison is skipped rather than dead-ending the call - matching
    the best-effort posture of the rest of the schema check.

    Shared by the two places that must compare a dataset's declared rate
    against another rate: the resume schema check (against the ``fps`` the
    caller asked to append at) and :func:`dataset_rate_mismatch_error`
    (against the ``control_frequency`` a rollout will actually capture at).

    Args:
        recorder: The ``DatasetRecorder`` whose dataset rate to read.

    Returns:
        The dataset frame rate as an int, or ``None`` when the dataset does not
        report a usable whole rate (an unexpected LeRobot layout must not block
        a valid resume). A rate beyond the float64 range is reported that way
        too: it is not a rate any recording can be written at, and raising the
        conversion error out of this reader told the caller the dataset had
        failed to open when it had opened fine.
    """
    dataset = getattr(recorder, "dataset", None)
    for value in (getattr(dataset, "fps", None), getattr(getattr(dataset, "meta", None), "fps", None)):
        # Classified on ``numbers.Real``, not ``int | float``: ``numpy.int64`` and
        # ``numpy.float32`` are neither ``int`` nor ``float`` subclasses, so the
        # narrower spelling read a whole rate this function calls "usable" as an
        # unreadable layout and returned ``None`` - which the one caller treats as
        # "do not judge", skipping the refusal. ``numpy.float64`` IS a ``float``
        # subclass, so the narrowing failed for some numpy spellings and not
        # others. The boolean question goes to the shared predicate, which also
        # catches ``numpy.bool_`` (not a ``bool`` subclass).
        if is_boolean(value) or not isinstance(value, numbers.Real):
            continue
        # Judged and returned through ``float``, the same hop both sibling
        # guards take (``declared = float(fps)`` ... ``fps_int = int(declared)``):
        # ``numbers.Real`` carries no ordering against ``int`` and no ``int()``
        # overload, so asking ``value > 0`` or truncating it directly is
        # untypeable. Deliberately not ``math.trunc(value)``, which would keep
        # a >2**53 rate exact but raises ``TypeError`` on ``numpy.int64`` - it
        # defines ``__int__``, not ``__trunc__`` - and numpy spellings are the
        # reason this classifies on ``Real`` at all. A rate that large is
        # float-rounded by both siblings too, so routing through ``float`` keeps
        # the three guards answering identically, which is the property the
        # cross-ordering test asserts. The conversion count is unchanged: the
        # previous spelling already called ``float`` to ask ``is_integer``.
        try:
            declared = float(value)
        except (OverflowError, TypeError, ValueError):
            # Reported as the unreadable layout it is, not raised. This rate
            # arrives off disk, where no domain has been asked: ``meta/info.json``
            # is JSON, whose integer literals are unbounded, and LeRobot's ``fps``
            # is an unenforced dataclass annotation, so ``LeRobotDataset`` opens
            # such a dataset without complaint and hands the value straight here.
            # Letting the conversion raise escaped a reader documented to answer
            # ``None`` for a rate it cannot read, and ``start_recording`` reported
            # it as ``Dataset init failed: int too large to convert to float`` -
            # a subject that had not failed, naming neither the field nor a
            # remedy. Resolving the rate through ``float`` converts before the
            # sign can be tested, so both signs reach the conversion and both are
            # answered here. Same handled exceptions, and the same reason, as
            # :func:`requested_rate_mismatch_reason`: it is the other guard in
            # this module asked before any domain has classified its rate.
            continue
        if declared > 0 and declared.is_integer():
            return int(declared)
    return None


def rate_mismatch_explanation(fps: int, rate: float) -> str:
    """State why a dataset rate other than the capture rate can only mislabel.

    Shared by both orderings of the same disagreement - a rollout started
    against an open recording (:func:`dataset_rate_mismatch_reason`) and a
    recording opened against a running rollout
    (:func:`rollout_rate_mismatch_reason`) - so the two cannot explain the same
    physics differently.

    Args:
        fps: Rate the dataset declares (or would declare) in its metadata.
        rate: Rate frames are actually captured at, in Hz.

    Returns:
        One sentence naming both intervals, the resulting distortion factor and
        what downstream consumes it. Callers supply their own prefix and remedy.
    """
    return (
        f"The dataset recorder writes one frame per control step with no decimation, so "
        f"frames captured {1.0 / rate:.4f}s apart would be timestamped {1.0 / fps:.4f}s "
        f"apart - a {rate / fps:.3f}x distortion of the episode duration, which a policy "
        f"trains on as its control period and which replay_episode reproduces at the wrong "
        f"speed."
    )


def dataset_rate_mismatch_reason(method: str, recorder: Any, control_frequency: float) -> str | None:
    """Explain why the active dataset cannot describe a rollout's capture rate.

    ``start_recording(fps=...)`` fixes the frame rate written into the dataset
    metadata, and LeRobot derives every frame's timestamp from it positionally
    (``timestamp = frame_index / fps``). The dataset recorder is driven once per
    control step with **no decimation**, so the rate frames are actually
    captured at is the rollout's ``control_frequency`` - which means a differing
    ``fps`` cannot be honored, only mislabelled.

    The two library defaults are exactly such a pair (``fps=30`` against
    ``control_frequency=50.0``), so the documented record-then-rollout sequence
    silently produced a distorted episode: frames captured 0.0200 s apart were
    timestamped 0.0333 s apart, declaring a 1.30 s episode for a 0.78 s capture.
    Nothing reported it - ``start_recording``, the rollout and
    ``stop_recording`` all returned ``status="success"``.

    That distortion is not cosmetic. It is the control period a policy trains
    on, and :meth:`~strands_robots.simulation.base.SimEngine.replay_episode`
    derives its per-frame physics budget from the dataset rate on the stated
    invariant that "the recorded control frequency IS the dataset fps" - so a
    mislabelled episode also replays at the wrong speed. Measured on a
    position-servo arm, record then replay round-tripped to 0.0000 rad at
    matching rates and to 0.0317 rad at the two defaults.

    Refusing (rather than warning) matches the sibling rate guard in this
    module: ``_verify_resume_schema`` already raises for an ``fps`` that
    disagrees with the dataset on disk, pointing at the rate to pass instead.
    This is the same disagreement one step earlier, and it is reported before
    any frame is written so the caller loses nothing.

    Args:
        method: Public method name, used to prefix the error message.
        recorder: The active ``DatasetRecorder``.
        control_frequency: Rate the rollout will capture frames at. Assumed
            already validated as a positive finite number by the caller's own
            ``control_frequency`` guard.

    Returns:
        The reason text naming both rates and the remedies, or ``None`` when the
        rates agree (or the dataset does not report a usable whole rate, which
        must not block a valid rollout). Callers reporting through a tool
        envelope want :func:`dataset_rate_mismatch_error`; a caller that must
        raise - a directly driven ``PolicyRunner`` - uses this text verbatim.
    """
    fps = recorder_dataset_fps(recorder)
    if fps is None:
        return None
    rate = float(control_frequency)
    if abs(rate - fps) < 1e-9:
        return None

    remedy = f"pass control_frequency={fps} to {method}()"
    if rate.is_integer():
        # ``fps`` must be a positive whole number, so only an integral capture
        # rate has a matching dataset rate to advise re-recording at.
        remedy += (
            f", or record at the rollout's rate (stop_recording(), then "
            f"start_recording(fps={int(rate)}, overwrite=True))"
        )
    text = (
        f"{method}: the active recording declares {fps} fps but this rollout captures at "
        f"control_frequency={rate:g} Hz. {rate_mismatch_explanation(fps, rate)} "
        f"Align the two rates: {remedy}."
    )
    return text


def dataset_rate_mismatch_error(method: str, recorder: Any, control_frequency: float) -> dict[str, Any] | None:
    """Envelope form of :func:`dataset_rate_mismatch_reason` for tool callers.

    Args:
        method: Public method name, used to prefix the error message.
        recorder: The active ``DatasetRecorder``.
        control_frequency: Rate the rollout will capture frames at.

    Returns:
        A structured ``{"status": "error", ...}`` dict carrying the reason text,
        or ``None`` when there is nothing to refuse. See
        :func:`dataset_rate_mismatch_reason` for the contract and why a mismatch
        is refused rather than warned.
    """
    reason = dataset_rate_mismatch_reason(method, recorder, control_frequency)
    if reason is None:
        return None
    return {"status": "error", "content": [{"text": reason}]}


def rollout_rate_mismatch_reason(method: str, fps: Any, rates: Mapping[str, float]) -> str | None:
    """Explain why an opening recording cannot describe an in-flight rollout.

    The inverse ordering of :func:`dataset_rate_mismatch_reason`. That guard
    covers a rollout started against an open recording; this one covers a
    recording opened against a rollout that is already running - which
    ``start_policy`` makes reachable by design, since it submits the rollout to
    an executor and returns while it continues.

    Both orderings produce the same distortion, because the frames and the
    timestamps come from the same two rates whichever call happened first: the
    recorder is driven once per control step with no decimation, and LeRobot
    derives each timestamp positionally from the declared ``fps``. Measured on
    the library defaults - ``start_policy(control_frequency=50.0)`` followed by
    ``start_recording(fps=30)`` - the episode saved 81 frames captured 0.0200 s
    apart and declared them 0.0333 s apart: a 2.6667 s episode for a 1.62 s
    capture, with ``start_policy``, ``start_recording`` and ``stop_recording``
    all returning ``status="success"``.

    Concurrent rollouts at different rates are refused outright, even when
    ``fps`` matches one of them: the frames interleave into one episode whose
    single declared rate can only describe one capture rate, so there is no
    value the caller could pass instead.

    Args:
        method: Public method name, used to prefix the error message.
        fps: Caller-supplied dataset frame rate. Validate it with
            :func:`dataset_recording_option_error` first; a value outside that
            domain returns ``None`` here so it is reported as the parameter
            error it is rather than as a rate disagreement. Classified on
            ``numbers.Real``, the predicate that domain uses, so every spelling
            it accepts - a ``numpy.int64`` rate read out of a config included -
            is judged here rather than silently passed through.
        rates: Capture rate in Hz per robot with a rollout in flight, as
            reported by
            :meth:`~strands_robots.simulation.base.SimEngine._active_rollout_rates`.

    Returns:
        The reason text naming the rollout(s), both rates and the remedies, or
        ``None`` when nothing is running or every running rollout already
        captures at ``fps``.
    """
    if not rates:
        return None
    # ``numbers.Real``, the predicate ``positive_whole_number_error`` classifies
    # this same ``fps`` with, so every spelling that domain accepts is judged
    # here. The narrower ``int | float`` declined to judge ``numpy.int64(30)`` -
    # a value that domain accepts, and is test-pinned as accepting - so the
    # disagreement this function exists to refuse was skipped and the episode was
    # written on a timebase that mislabels it. See the identical reasoning in
    # :func:`requested_rate_mismatch_reason`.
    if is_boolean(fps) or not isinstance(fps, numbers.Real):
        return None
    # ``float()`` cannot overflow here: this guard is asked only after
    # ``dataset_recording_option_error``, whose domain refuses an ``fps`` beyond
    # the float64 range with a reason of its own. That is why it needs no
    # ``try`` - unlike ``requested_rate_mismatch_reason``, which is asked first.
    declared = float(fps)
    if declared <= 0 or not declared.is_integer():
        return None
    fps_int = int(declared)

    listed = ", ".join(f"'{name}' at {rate:g} Hz" for name, rate in sorted(rates.items()))
    distinct = {round(rate, 9) for rate in rates.values()}
    if len(distinct) > 1:
        return (
            f"{method}: this recording would declare {fps_int} fps but {len(rates)} rollouts are "
            f"already running at {len(distinct)} different capture rates ({listed}). The dataset "
            f"recorder writes one frame per control step with no decimation and LeRobot timestamps "
            f"every frame from the single declared rate, so no fps can describe them all - their "
            f"frames interleave into one episode on a timebase that mislabels at least one of "
            f"them. Stop all but one rollout (stop_policy(robot_name=...)), or restart them at a "
            f"common control_frequency, then record at that rate."
        )

    rate = float(next(iter(rates.values())))
    if abs(rate - fps_int) < 1e-9:
        return None

    remedy = ""
    if rate.is_integer():
        # ``fps`` must be a positive whole number, so only an integral capture
        # rate is one this method could be re-invoked with.
        remedy = f"record at the rollout's rate ({method}(fps={int(rate)})), or "
    stop_targets = ", ".join(f"stop_policy(robot_name='{name}')" for name in sorted(rates))
    remedy += (
        f"restart the rollout at the recording's rate ({stop_targets}, then "
        f"start_policy(..., control_frequency={fps_int}))"
    )
    return (
        f"{method}: this recording would declare {fps_int} fps but a rollout is already running "
        f"({listed}). {rate_mismatch_explanation(fps_int, rate)} Align the two rates: {remedy}."
    )


def rollout_rate_mismatch_error(method: str, fps: Any, rates: Mapping[str, float]) -> dict[str, Any] | None:
    """Envelope form of :func:`rollout_rate_mismatch_reason` for tool callers.

    Args:
        method: Public method name, used to prefix the error message.
        fps: Caller-supplied dataset frame rate.
        rates: Capture rate in Hz per robot with a rollout in flight.

    Returns:
        A structured ``{"status": "error", ...}`` dict carrying the reason text,
        or ``None`` when there is nothing to refuse. See
        :func:`rollout_rate_mismatch_reason` for the contract and why a mismatch
        is refused rather than warned.
    """
    reason = rollout_rate_mismatch_reason(method, fps, rates)
    if reason is None:
        return None
    return {"status": "error", "content": [{"text": reason}]}


def requested_rate_mismatch_reason(method: str, fps: Any, control_frequency: Any, fps_param: str = "fps") -> str | None:
    """Explain why one call's own two rates cannot describe the same episode.

    The third ordering of the disagreement its two siblings cover, and the only
    one in which neither rate can be read off live state:
    :func:`dataset_rate_mismatch_reason` reads the frame rate from an open
    recorder, and :func:`rollout_rate_mismatch_reason` reads the capture rate
    from a rollout already in flight, so neither can be asked before either
    exists. A caller that supplies *both* rates in one call and opens the
    recording itself has them in hand while nothing is open yet - and must be
    answered then, because that call is also the one that destroys the target.

    Its caller is :func:`~strands_robots.tools.run_policy.run_policy`, which
    starts its recording with ``overwrite=True`` before the episode loop. Each
    rate there is already checked on its own domain - ``control_frequency`` by
    the tool's own guard, ``dataset_fps`` by ``start_recording`` ahead of the
    target - but their *equality* was left to the rollout entry point, which the
    tool reaches only inside the loop. Measured against an existing dataset of
    one episode / five frames, ``dataset_fps=30`` with
    ``control_frequency=50.0`` wiped it to ``total_episodes=0, total_frames=0``
    and then reported ``0/2 episodes ok`` - the caller lost the dataset and
    recorded nothing, for a pair of arguments neither of which was wrong on its
    own.

    No envelope twin is shipped alongside this reason (unlike its two siblings):
    the one caller reports through a structured-error helper of its own, so an
    envelope form here would be dead code.

    Args:
        method: Public surface name, used to prefix the error message.
        fps: Caller-supplied dataset frame rate. A value outside the writable
            domain returns ``None`` here so it is reported as the parameter
            error it is by :func:`dataset_recording_option_error`, exactly as in
            :func:`rollout_rate_mismatch_reason`.
        control_frequency: Rate the rollout will capture frames at. A value
            outside the positive-finite domain likewise returns ``None``, left
            to the caller's own ``control_frequency`` guard. Both rates are
            classified on ``numbers.Real``, the same predicate those two domains
            use, so every spelling they accept is judged here too.
        fps_param: How the caller spells the frame-rate argument, so the advised
            remedy is one the caller can type. Defaults to ``fps``, the name
            every backend's ``start_recording`` uses.

    Returns:
        The reason text naming both rates, the distortion and the remedies, or
        ``None`` when the rates agree or either is outside its own domain.
    """
    # Unlike its two siblings, this guard is asked BEFORE either rate has been
    # through its own domain - that is the point of it - so it classifies raw
    # caller input, and it must accept every spelling those domains accept or it
    # silently declines to judge a pair they will both honor. Hence
    # ``numbers.Real``, exactly as ``positive_whole_number_error`` and
    # ``positive_finite_number_error`` use: ``numpy.int64`` and ``numpy.float32``
    # are neither ``int`` nor ``float`` subclasses, so an ``isinstance(int |
    # float)`` narrowing here would have passed a colliding pair read out of a
    # config straight through. The boolean question goes to the shared predicate
    # (``bool`` IS a ``numbers.Real``, so it would otherwise be compared as
    # 1 Hz).
    if is_boolean(fps) or not isinstance(fps, numbers.Real):
        return None
    if is_boolean(control_frequency) or not isinstance(control_frequency, numbers.Real):
        return None
    try:
        declared = float(fps)
        rate = float(control_frequency)
    except (OverflowError, TypeError, ValueError):
        # An ``int`` beyond the float64 range cannot be compared with anything;
        # its own domain refuses it with a reason of its own.
        return None

    if declared <= 0 or not declared.is_integer():
        return None
    fps_int = int(declared)
    if rate <= 0 or not math.isfinite(rate):
        return None

    if abs(rate - fps_int) < 1e-9:
        return None

    remedy = f"pass control_frequency={fps_int}"
    if rate.is_integer():
        # ``fps`` must be a positive whole number, so only an integral capture
        # rate is one the recording could be opened at instead.
        remedy += f", or record at the rollout's rate ({fps_param}={int(rate)})"
    return (
        f"{method}: this call would open a recording declaring {fps_int} fps for a rollout "
        f"capturing at control_frequency={rate:g} Hz. {rate_mismatch_explanation(fps_int, rate)} "
        f"Align the two rates: {remedy}."
    )


def _camera_height_width(shape: Sequence[Any], names: Sequence[str] | None) -> tuple[Any, Any]:
    """Read ``(height, width)`` from a camera feature declared in either layout.

    The recorder declares cameras the way lerobot does, HWC ``(H, W, 3)`` with
    names ``[height, width, channels]``; datasets it recorded before that are
    CHW ``(3, H, W)`` with names ``[channels, height, width]``. Names decide
    when present (lerobot's own transposition rule); without names the
    3-channel axis is located by value.
    """
    if names and set(names) >= {"height", "width"}:
        return shape[list(names).index("height")], shape[list(names).index("width")]
    if shape[0] == 3 and shape[-1] != 3:
        return shape[1], shape[2]
    return shape[0], shape[1]


def _resume_schema_error(diffs: list[str]) -> str:
    """Format the resume-refusal message from the collected schema differences.

    Args:
        diffs: One human-readable line per divergence found.

    Returns:
        The full error message, listing every difference.
    """
    return (
        "Cannot resume recording: the current scene does not match the existing dataset schema. "
        "Use overwrite=True for a fresh dataset, or restore the original scene. Differences:\n  - "
        + "\n  - ".join(diffs)
    )


def undriven_robot_state(engine: Any, driven_robots: Collection[str], robot_names: Iterable[str]) -> dict[str, Any]:
    """Read the scalar state of every robot a recorded frame does not drive.

    ``start_recording`` declares ``observation.state`` over EVERY robot in the
    scene, prefixing each column with its robot's name
    (``alice__shoulder_pan``). A rollout's recording hook supplies only the
    observation of the robots it drives, so every column belonging to another
    robot is absent from the frame and
    :meth:`~strands_robots.dataset_recorder.DatasetRecorder.add_frame` writes it
    as ``0.0`` - under ``status="success"``, for every frame of the episode.

    That zero is not a missing value the consumer can detect. It is recorded in
    the same column, with the same dtype, as a measurement: a policy trained on
    the dataset reads ``observation.state`` and learns the other robot is
    permanently at its zero pose, and anything replaying or analysing the
    episode reads the same. The disagreement is unbounded - the undriven robot
    keeps whatever pose it was placed in, and contact with the driven robot can
    move it further during the episode.

    Unlike the *action* columns of the same frame, there is nothing to decide
    here. An action column asks what command was issued, and no command was
    issued to a robot this rollout does not drive (#1715 works through why no
    substitute is truthful). A state column asks where the robot is, the robot
    is in the scene, and its joint positions are readable at that instant from
    the same engine, at the same step, through the same
    :meth:`~strands_robots.simulation.base.SimEngine.get_observation` the driven
    robot's own columns come from. So the undriven columns are filled with the
    measurement rather than left to the ``0.0`` fill.

    Every recording entry point needs this, because each one declares the
    schema over the whole scene and then supplies a frame covering only the
    robots it drives. That is the three single-policy hooks, whose frame
    carries one robot, and ``run_multi_policy``'s synchronized loop, whose
    merged frame carries the keys of its ``policies`` mapping - which
    :meth:`~strands_robots.simulation.base.SimEngine.run_multi_policy`
    requires to name robots in the scene but never requires to name all of
    them. A synchronized call that drives a subset therefore leaves the rest
    to the fill exactly as a single-policy hook does, so both go through here
    and which rollout entry point recorded an episode does not change whether
    its state columns are measurements.

    Images are deliberately not collected. A camera array is keyed by the
    camera's own name rather than a robot's, and the recording paths scope
    cameras through ``start_recording(cameras=...)``, so an undriven robot
    contributes no image column. Only scalars are returned, prefixed exactly as
    the hooks prefix the driven robot's.

    Args:
        engine: The simulation engine, read through its public
            ``get_observation(robot_name=..., skip_images=True)``.
        driven_robots: The robots this frame drives; their columns are
            supplied by the caller and are not re-read here. Pass ``(name,)``
            from a single-policy hook and the ``policies`` mapping from a
            synchronized multi-robot loop.
        robot_names: Every robot the dataset schema declares columns for.

    Returns:
        ``{"<robot>__<joint>": value}`` for each scalar observation of each
        robot outside ``driven_robots``. Empty when every robot in the scene
        is driven, which includes the single-robot case where the schema is
        not prefixed at all.

    Raises:
        TypeError: ``driven_robots`` is a bare ``str``. A string is iterable
            per character, so it would make the skip below a substring test:
            a robot named ``ali`` satisfies ``"ali" in "alice"`` and would be
            skipped as though it were driven, dropping its columns to the
            very fill this helper exists to avoid.
    """
    if isinstance(driven_robots, str):
        raise TypeError(
            "undriven_robot_state: 'driven_robots' must be a collection of robot names, "
            f"not the bare string {driven_robots!r}. Pass ({driven_robots!r},) to name one."
        )
    driven = frozenset(driven_robots)
    merged: dict[str, Any] = {}
    for name in robot_names:
        if name in driven:
            continue
        try:
            observation = engine.get_observation(robot_name=name, skip_images=True)
        except (AttributeError, KeyError, RuntimeError, ValueError):
            # ``engine`` is duck-typed - every backend's own engine reaches here,
            # and a state read it cannot serve must degrade to the recorder's
            # fill rather than end the episode. The driven robot's columns are
            # the rollout's primary product: losing a whole episode of them to a
            # bystander's read failure is strictly worse than the fill this
            # helper exists to avoid, so the failure is reported and the frame
            # still lands. ``AttributeError`` is in the set deliberately - an
            # engine that has not built (or has torn down) the state buffers a
            # per-robot read needs surfaces exactly that way.
            logger.warning(
                "recording: could not read state for robot %r this step; its declared "
                "observation.state columns fall back to the recorder's fill for this frame.",
                name,
            )
            continue
        if not isinstance(observation, Mapping):
            continue
        for key, value in observation.items():
            # Mirror the hooks' own split: a non-scalar value is a camera array,
            # which is not a per-robot column.
            if hasattr(value, "shape"):
                continue
            merged[f"{name}__{key}"] = value
    return merged


class DatasetRecordingMixin:
    """Engine-independent recording lifecycle shared by sim backends.

    Provides ``stop_recording`` / ``save_episode`` / ``get_recording_status`` /
    ``stream_dataset`` plus the ``_is_recording`` / ``_active_recorder`` /
    ``_active_dataset_root`` overrides the base :class:`SimEngine` run-policy
    loop reads. Backends mix this in and add their own ``start_recording``
    (schema declaration) and ``_make_run_policy_hook`` (per-step capture).
    """

    if TYPE_CHECKING:
        from strands_robots.simulation.models import SimWorld
        from strands_robots.streaming_dataset import StreamingDatasetReader

        _world: "SimWorld | None"
        default_width: int
        default_height: int

        def _validate_recording_start_rate(self, fps: Any, method: str) -> dict[str, Any] | None:
            """Type-only stub for the engine-provided rate guard.

            Declared once here rather than in each backend mixin: all three
            ``start_recording`` implementations call it and all three inherit
            this class, so one declaration keeps the contract in a single
            place. Implemented by
            :meth:`~strands_robots.simulation.base.SimEngine._validate_recording_start_rate`.
            """

    def _recording_state(self) -> dict[str, Any] | None:
        """Mutable recording-state mapping, or ``None`` when no world exists.

        This is the single seam every engine-independent lifecycle method goes
        through to reach the recording flag, trajectory mirror,
        ``dataset_recorder`` handle, ``recording_cameras`` scope and
        ``last_dataset_root``. The default implementation reads
        ``self._world._backend_state`` - the ``SimWorld`` contract shared by
        the MuJoCo and Newton backends, which need no override. Backends whose
        ``self._world`` is not a ``SimWorld`` (the Isaac backend holds the
        Isaac Sim ``World`` handle there) override this accessor to return
        their own dict, keeping one shared mixin instead of a fork.

        Returns:
            The live state dict, or ``None`` when there is no world (recording
            is then reported as inactive and lifecycle calls degrade to their
            documented no-world responses).
        """
        world = self._world
        if world is None:
            return None
        return world._backend_state

    @staticmethod
    def _dataset_recorder_or_refusal(alternative: str) -> tuple[Any, dict[str, Any] | None]:
        """Resolve ``DatasetRecorder``, or the refusal naming why it is missing.

        Every backend's ``start_recording`` needs the same two facts before it
        touches the scene: the recorder class, and - when lerobot's dataset
        stack is not installed - a diagnosis that names the missing piece
        instead of a bare ``ImportError`` from inside the first write. The three
        backends each carried a verbatim copy of the probe, so a change to the
        diagnosis had to be made three times and could drift in two of them.
        Only the alternative to recording a dataset differs per backend, so
        that sentence is the parameter.

        Args:
            alternative: What to do instead, appended to the refusal - the plain
                MP4 path this backend offers under its own extra.

        Returns:
            ``(DatasetRecorder, None)`` when the stack is importable, otherwise
            ``(None, refusal)`` where ``refusal`` is the tool reply to return.
        """
        recorder_cls: Any = None
        unavailable: str | None = None
        try:
            # Deferred because the import is itself what this probe reports on:
            # at module scope, a partial install would make `import
            # strands_robots.simulation` fail rather than this verb refuse.
            from strands_robots.dataset_recorder import DatasetRecorder as recorder_cls
            from strands_robots.dataset_recorder import lerobot_dataset_import_error

            unavailable = lerobot_dataset_import_error()
        except ImportError as exc:
            # strands_robots.dataset_recorder itself did not import (a partial or
            # drifted install); report that rather than blaming the lerobot extra.
            unavailable = f"strands_robots.dataset_recorder is unavailable ({exc})."
        if unavailable is None and recorder_cls is None:
            unavailable = "strands_robots.dataset_recorder did not provide DatasetRecorder."
        if unavailable is None:
            return recorder_cls, None
        return None, {
            "status": "error",
            "content": [
                {
                    "text": (
                        "start_recording produces a LeRobotDataset (parquet + video), which "
                        "needs lerobot's dataset stack:\n"
                        "\n"
                        f"  {unavailable}\n"
                        "\n"
                        f"{alternative}"
                    )
                }
            ],
        }

    @staticmethod
    def _arm_dataset_recorder(state: dict[str, Any], recorder: Any, *, resumed: bool = False) -> str:
        """Open a recording session on ``recorder``, and say what opening it means.

        Every backend's ``start_recording`` arms the recorder through here, so
        the shared lifecycle can measure a SESSION rather than a dataset.
        ``DatasetRecorder.resume`` seeds ``frame_count``/``episode_count`` with
        the dataset's totals (so the saved report covers the whole dataset),
        which leaves those counters unable to say whether THIS session captured
        anything: read alone they let a resumed session that captured nothing
        clear :meth:`stop_recording`'s empty-capture guard on the PREVIOUS
        sessions' frames and be reported as a saved episode. The counts the
        session starts from are stashed here once, for every backend - a backend
        that assigned ``state["dataset_recorder"]`` itself would silently get
        that behaviour back.

        Arming and announcing are one call because they are one decision: the
        reply names the counts the session will be measured against, read back
        out of the stash so the two can never disagree. The counters are read
        defensively - a recorder that does not carry them opens a session at
        zero rather than turning ``start_recording`` into "Dataset init failed".

        Args:
            state: The recording state mapping (see :meth:`_recording_state`).
            recorder: The freshly created or resumed ``DatasetRecorder``.
            resumed: Whether ``recorder`` was resumed from an existing dataset.

        Returns:
            The line the reply adds for a resume, or ``""`` for a fresh dataset.
        """
        state["dataset_recorder"] = recorder
        state["frames_at_start"] = int(getattr(recorder, "frame_count", 0) or 0)
        state["episodes_at_start"] = int(getattr(recorder, "episode_count", 0) or 0)
        if not resumed:
            return ""
        return resumed_dataset_line(episodes=state["episodes_at_start"], frames=state["frames_at_start"])

    @staticmethod
    def _release_dataset_recorder(state: dict[str, Any]) -> None:
        """Drop the active recorder and everything else scoped to its session.

        The trajectory mirror and the start-of-session counts stashed by
        :meth:`_arm_dataset_recorder` describe the recorder being released, so
        they go with it rather than outliving it.

        Args:
            state: The recording state mapping (see :meth:`_recording_state`).
        """
        state["dataset_recorder"] = None
        state["trajectory"] = []
        state.pop("frames_at_start", None)
        state.pop("episodes_at_start", None)

    @staticmethod
    def _prepare_dataset_target(dataset_dir: Path, overwrite: bool) -> bool:
        """Resolve create-vs-resume and make ``dataset_dir`` safe for create().

        ``LeRobotDataset.create()`` raises ``FileExistsError`` when its target
        directory already exists - even when that directory is empty. Callers
        very commonly pass an existing empty directory as ``root`` (for example
        one returned by ``tempfile.mkdtemp()``), which would otherwise dead-end
        recording with a cryptic ``[Errno 17] File exists``. This resolves the
        situation up front:

        * ``overwrite``: remove any existing target, then create() fresh.
        * existing dataset (has a ``meta/`` dir): resume (append episodes).
        * existing EMPTY dir: remove it so create() can recreate it cleanly.
        * existing NON-empty, non-dataset dir: raise ``ValueError`` with an
          actionable message instead of clobbering unrelated files.

        ``overwrite`` is a *posture*, and every public caller checks it on
        :func:`dataset_recording_posture_error` first, so the value arriving here
        is a boolean. That bound matters because the branch below is the only one
        that deletes a real LeRobotDataset without asking: a truthy non-boolean -
        ``"false"``, the spelling an operator reaches for when opting out - used
        to land in it.

        Args:
            dataset_dir: Resolved on-disk dataset root.
            overwrite: When True, replace any existing target. Bounded to a
                boolean by every public caller.

        Returns:
            True if an existing dataset should be resumed (append), False if a
            fresh ``create()`` should run.

        Raises:
            ValueError: When the target exists, is not a LeRobotDataset, is not
                empty, and ``overwrite`` is False - so create() would fail with
                a cryptic ``FileExistsError``.
        """
        if not dataset_dir.exists():
            return False
        if overwrite:
            if dataset_dir.is_dir():
                shutil.rmtree(dataset_dir)
            else:
                dataset_dir.unlink()
            logger.info("Removed existing dataset target: %s", dataset_dir)
            return False
        if not dataset_dir.is_dir():
            raise ValueError(
                f"Recording target {dataset_dir} exists and is not a directory. "
                "Pass a directory path as root=, or overwrite=True to replace it."
            )
        if (dataset_dir / "meta").exists():
            return True  # real dataset on disk -> resume/append
        if not any(dataset_dir.iterdir()):
            # Empty dir (e.g. from tempfile.mkdtemp()): clear it so create()
            # does not trip over its own pre-existing-directory guard.
            shutil.rmtree(dataset_dir)
            logger.info("Cleared empty recording target for fresh dataset: %s", dataset_dir)
            return False
        raise ValueError(
            f"Recording target {dataset_dir} already exists, is not a LeRobotDataset "
            "(no meta/ directory), and is not empty. Refusing to overwrite unrelated "
            "files. Pass overwrite=True to replace it, or choose a new/empty root=."
        )

    def _is_recording(self) -> bool:
        """True when a dataset-recording session is active.

        Overrides :meth:`SimEngine._is_recording` so the multi-episode
        ``run_policy`` loop flushes an episode boundary after each rollout
        only while a recording is open.
        """
        state = self._recording_state()
        return state is not None and bool(state.get("recording", False))

    def _active_recorder(self) -> Any:
        """Live dataset recorder, or ``None`` when no session is open.

        Overrides :meth:`SimEngine._active_recorder` so the base ``run_policy``
        episode-contract fields can read the recorder's in-memory episode count.
        """
        state = self._recording_state()
        if state is None:
            return None
        return state.get("dataset_recorder")

    def _active_dataset_root(self) -> str | None:
        """On-disk root of the active or most-recently-recorded dataset.

        Overrides :meth:`SimEngine._active_dataset_root` so
        :meth:`verify_dataset_episodes` can locate the parquet AFTER
        ``stop_recording`` has finalized it and dropped the recorder. Prefers
        the live recorder's root; falls back to the ``last_dataset_root`` stashed
        at ``start_recording``.
        """
        recorder = self._active_recorder()
        if recorder is not None:
            try:
                return str(recorder.root)
            except (AttributeError, TypeError):
                pass
        state = self._recording_state()
        if state is None:
            return None
        last = state.get("last_dataset_root")
        return str(last) if last else None

    def _active_dataset_repo_id(self) -> str | None:
        """Id of the active or most-recently-recorded dataset.

        Overrides :meth:`SimEngine._active_dataset_repo_id`, resolved exactly
        like its :meth:`_active_dataset_root` counterpart - the live recorder
        first, then the ``last_dataset_repo_id`` stashed at ``start_recording``.
        """
        recorder = self._active_recorder()
        if recorder is not None:
            try:
                return str(recorder.repo_id)
            except (AttributeError, TypeError):
                pass
        state = self._recording_state()
        if state is None:
            return None
        last = state.get("last_dataset_repo_id")
        return str(last) if last else None

    def _stash_dataset_target(self, repo_id: str, root: str | None) -> Path:
        """Resolve the directory a recording writes to, stashed with its id.

        Every backend's ``start_recording`` resolves its target with
        :func:`~strands_robots.dataset_source.resolve_dataset_dir` - the same
        resolver ``DatasetRecorder.create()`` uses, so the facade and the
        recorder agree on where a dataset lives (honouring ``$HF_LEROBOT_HOME``)
        - and stashes it as ``last_dataset_root`` for the consumers that run
        after the recorder is dropped (:meth:`_active_dataset_root`). The id it
        was recorded under is stashed beside it, because a reader handed only an
        ``owner/name`` id cannot derive a directory the caller chose with
        ``root=``.

        Resolving and stashing in ONE place is the point: the pair was written
        out longhand in all three backends, so a value added to one of them left
        the other two recording datasets no reader could locate.

        Args:
            repo_id: HuggingFace dataset id (``owner/name``) or a local path.
            root: Explicit local dataset directory, if any.

        Returns:
            The resolved directory, for the caller's overwrite/resume logic.
        """
        from strands_robots.dataset_source import resolve_dataset_dir

        dataset_dir = resolve_dataset_dir(repo_id, root)
        state = self._recording_state()
        if state is not None:
            state["last_dataset_root"] = str(dataset_dir)
            state["last_dataset_repo_id"] = repo_id
        return dataset_dir

    def stop_recording(
        self,
        *,
        push_to_hub: bool = False,
        bucket: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Stop recording and save episode to LeRobotDataset.

        Idempotent - a bare call when not recording succeeds with a
        'Was not recording' message so callers can safely call it unconditionally.
        When ``bucket=`` / ``run_id=`` / ``push_to_hub=True`` are passed while
        NOT recording, they are never silently dropped (AGENTS.md: forward all
        advertised kwargs): ``bucket=`` syncs the last-finalized dataset of this
        sim (the ``last_dataset_root`` stashed at ``start_recording``) so the
        "re-run stop_recording(bucket=...) as the daily sync" workflow works;
        ``push_to_hub=True`` (or ``run_id=`` without ``bucket=``) returns a
        structured ``status="error"`` because there is nothing to publish.

        Returns a structured ``status="error"`` when the recording captured no
        frames (the dataset would contain only ``meta/info.json``), rather than
        silently writing an empty dataset. Only ``run_policy`` feeds the active
        recorder (via its ``on_frame`` hook); ``eval_policy`` / ``evaluate`` /
        ``replay_episode`` and bare ``step`` loops do not, so recording around
        those produces zero frames and is reported as an error.

        The frames a ``strict=False`` recorder dropped are reported here too,
        because this is where the recorder is released - a count left unread
        here is a loss no caller can measure afterwards. It reaches the caller
        two ways. When every write failed the dataset is empty for that reason
        and the refusal names it, rather than the loop classification above,
        which would prescribe the recipe the caller already followed. When only
        some failed the session stays a success (``strict=False`` documents
        dropping a failed write and completing) that reports how short it is,
        in the text and in ``dropped_frame_count``.

        Takes no destination. The dataset root is chosen once, at
        ``start_recording(root=...)``, and the recorder has been writing there
        for the whole episode, so nothing is left here to redirect - an
        ``output_path`` could only be discarded. Both entry points refuse one:
        the agent dispatcher by name (``output_path`` is a published schema
        field, so it used to bind here and be dropped), python with a
        ``TypeError``.

        Args:
            push_to_hub: Publish to a versioned HF *dataset* repo (the finished
                artifact). Overrides the ``push_to_hub`` set at start_recording.
                Requires an open recording session; on the idle path this
                returns ``status="error"`` instead of a silent no-op. Must be a
                boolean: a publication posture is not read by truthiness
                (:func:`dataset_recording_posture_error`).
            bucket: If set (e.g. ``"my-org/robot-fave"``), sync the dataset into
                a mutable HF Storage Bucket instead of/in addition to the dataset
                repo - the Phase 1/2 collection target (Xet-deduped, overwrite in
                place). When no recording is open, syncs the last dataset this
                sim finalized (errors if there is none).
            run_id: Optional subpath inside the bucket (defaults to dataset name).

        Returns:
            The standard agent-tool envelope. Its json block reports the episode
            bookkeeping as three fields: ``episode_count`` (the canonical count -
            the dataset's own when that could be read, else the recorder's),
            ``parquet_episode_count`` (the dataset's ``meta.total_episodes``, or
            ``None`` when the recorder exposes no dataset handle, that layout
            carries no such attribute, or the value cannot be read as an int -
            an unreadable count is reported as no reading, never as a zero),
            ``episode_count_mismatch`` (the two counts were both read and
            disagreed, so the on-disk one won) and ``dropped_frame_count``
            (frames the recorder was fed and could not write, swallowed by
            ``strict=False``; ``frame_count`` plus this is what the session
            attempted).
        """
        # ``push_to_hub`` selects whether the finished dataset is published, so
        # it is checked before it is read - by the idle path just below and by
        # the upload after the episode is finalized. Read by truthiness a
        # non-boolean opt-out ("false", "no", "off", "0") published the dataset.
        if error := dataset_recording_posture_error("stop_recording", "push_to_hub", push_to_hub):
            return error
        state = self._recording_state()
        if state is None or not state.get("recording", False):
            return self._stop_recording_idle(push_to_hub=push_to_hub, bucket=bucket, run_id=run_id)

        state["recording"] = False
        # The step-fed frame clock belongs to the session that just closed.
        state.pop("step_recording_due", None)
        recorder = state.get("dataset_recorder", None)

        if recorder is None:
            return {"status": "error", "content": [{"text": "No dataset recorder active."}]}

        # Save the trailing (unsaved) episode, then guard against an empty
        # dataset. ``episode_frame_count`` is the frames captured since the last
        # save_episode; ``frame_count`` is the monotonic total across the
        # dataset. Three cases:
        #
        #   1. Unsaved frames pending (episode_frame_count > 0): flush them with
        #      save_episode. If LeRobot rejects the flush, surface the error.
        #   2. No pending frames but the dataset already has some (callers that
        #      save per-episode and call stop_recording last): nothing to flush,
        #      just finalize - calling save_episode here would hit LeRobot's
        #      "add frames before add_episode" guard on the empty buffer and
        #      wrongly fail an otherwise-complete dataset.
        #   3. Nothing ever captured (frame_count == 0): fail loudly instead of
        #      writing a 0-frame dataset. This happens when the rollout was
        #      driven by eval_policy / evaluate / replay_episode without a hook
        #      (only a rollout's on_frame hook calls add_frame, and run_policy,
        #      start_policy and run_multi_policy each launch one), or by a step
        #      loop on a backend other than MuJoCo, whose ``step`` feeds the
        #      recorder at the dataset fps. Previously stop_recording reported
        #      success with "0 frames, 0 episode(s)", silently producing a
        #      dataset with only meta/info.json (no parquet/video).
        pending = getattr(recorder, "episode_frame_count", 0)
        # ``frame_count`` is seeded with the dataset's total when a dataset is
        # RESUMED, so on its own it cannot say whether THIS session captured
        # anything: a resumed session that captured nothing used to pass the
        # empty-capture guard below on the previous sessions' frames and report
        # "Episode saved" for an episode that was never written. Measure the
        # session against the count stashed at start_recording.
        total_frames = getattr(recorder, "frame_count", 0)
        frames_at_start = int(state.get("frames_at_start", 0) or 0)
        episodes_at_start = int(state.get("episodes_at_start", 0) or 0)
        captured = max(0, total_frames - frames_at_start)
        # Frames the recorder was fed and could not write. A ``strict=False``
        # recorder drops those and counts them here instead of raising, so they
        # are the only record that the session lost data - and this method is
        # where the recorder is released, so a count not reported here is a
        # count no caller can ever read.
        dropped = getattr(recorder, "dropped_frame_count", 0)
        if pending > 0:
            save_result = recorder.save_episode()
            if isinstance(save_result, dict) and save_result.get("status") == "error":
                self._release_dataset_recorder(state)
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                "stop_recording failed to save the final episode "
                                f"({pending} pending frames). save_episode: "
                                f"{save_result.get('message')}"
                            )
                        }
                    ],
                }
        elif captured == 0 and dropped > 0:
            # The recorder WAS fed and every write failed, so the loop
            # classification below is the one cause that cannot apply: it would
            # prescribe the recipe this caller already followed and never
            # mention the drops. ``strict=False`` chose to swallow them, so the
            # rollout reported success and this is the first refusal.
            self._release_dataset_recorder(state)
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"stop_recording: all {dropped} frame(s) the recorder was fed failed "
                            "to write, so the dataset holds 0 frames. The recorder was built with "
                            "strict=False, which drops a failed write and counts it in "
                            "dropped_frame_count instead of raising - which is why the rollout "
                            "reported success. Fix the write failure (the per-drop warnings from "
                            "strands_robots.dataset_recorder name it) or record with strict=True, "
                            "the default, so the first lost frame fails the rollout at the frame "
                            "that lost it."
                        )
                    }
                ],
            }
        elif captured == 0:
            self._release_dataset_recorder(state)
            # A resumed session is measured against the dataset it resumed:
            # nothing was appended, so the dataset is exactly as it was.
            resumed_note = (
                (
                    f"This session captured no frames: the resumed dataset {recorder.repo_id} "
                    f"({frames_at_start} frames, {episodes_at_start} episode(s)) is unchanged "
                    "and no episode was saved. "
                )
                if frames_at_start > 0
                else ""
            )
            recipe = (
                "stop_recording captured no frames - dataset would be empty "
                "(0 frames). run_policy(...) feeds the recorder on its own: it "
                "installs the per-step on_frame hook that calls add_frame. "
                "eval_policy / evaluate_benchmark feed it the same way while a "
                "recording is open (one dataset episode per evaluation episode) "
                "unless the caller passes an on_frame of their own, which then "
                "records only if it calls add_frame. "
                "replay_episode and teleoperate have no such hook and cannot feed "
                "the recorder. On the MuJoCo backend step() records too: one frame per "
                "1/fps seconds of sim time while a recording is open, so "
                "set_joint_positions(hold=True) + step is a scripted demonstration "
                "(a step call covering less sim time than one frame period captures "
                "nothing, and its reply says so). To record a dataset: "
                "start_recording -> run_policy(n_episodes=N), or run_policy then reset "
                "per episode (reset closes the open episode), or step through the "
                "motion -> stop_recording."
            )
            return {"status": "error", "content": [{"text": resumed_note + recipe}]}

        repo_id = recorder.repo_id
        frame_count = recorder.frame_count
        episode_count = recorder.episode_count
        root = recorder.root

        # Finalize FIRST so meta/ (stats/info) is written before any bucket sync
        # - streaming/training downstream needs it.
        recorder.finalize()

        # #708 - parquet-truth gate. The recorder's ``episode_count`` is the
        # author-side bookkeeping (incremented by every ``save_episode`` call).
        # The dataset's own ``meta.total_episodes`` / parquet rowcount is what
        # downstream consumers (HF hub, training loaders, audit tools) trust.
        # If they disagree, the on-disk dataset is the source of truth (Law-7
        # in AGENTS.md: parquet num_rows > meta/info.json > markdown). Surface
        # the mismatch in the returned payload so the caller - and any CI that
        # parses the status dict - can fail loudly instead of shipping a
        # silent-collapse dataset.
        parquet_episode_count: int | None = None
        episode_count_mismatch: bool = False
        episode_count_mismatch_orig: int = episode_count
        try:
            ds_meta = getattr(getattr(recorder, "dataset", None), "meta", None)
            # A layout that exposes no ``total_episodes`` is a FAILED PROBE, not
            # a dataset holding zero episodes. Reading it with a zero default
            # would hand that zero to the gate below as ground truth and
            # overwrite the episode count the recorder actually measured, so a
            # session that saved N episodes would report 0 with the mismatch
            # flag raised. Probe with ``None`` and skip the gate when the
            # attribute is absent - the same "unavailable means skip" the
            # missing-``dataset`` branch above and ``recorder_dataset_fps``
            # already use on this path.
            raw_total_episodes = getattr(ds_meta, "total_episodes", None) if ds_meta is not None else None
            if raw_total_episodes is not None:
                parquet_episode_count = int(raw_total_episodes)
                if parquet_episode_count != episode_count:
                    episode_count_mismatch = True
                    logger.warning(
                        "stop_recording: recorder.episode_count=%d but "
                        "dataset.meta.total_episodes=%d. Trust the parquet. "
                        "(#708 silent-collapse gate)",
                        episode_count,
                        parquet_episode_count,
                    )
                    # The parquet is the ground truth - report it as the
                    # canonical episode_count downstream. Stash the original
                    # recorder.episode_count so the text payload can name both.
                    episode_count_mismatch_orig = episode_count
                    episode_count = parquet_episode_count
        except Exception as e:  # noqa: BLE001 - never fail finalize on a probe
            logger.debug("episode_count gate probe failed: %s", e)

        extra = ""
        # Bucket sync (Phase 1/2): mutable, Xet-deduped collection dump.
        if bucket:
            sync_result = recorder.sync_to_bucket(bucket, run_id=run_id)
            if sync_result.get("status") == "success":
                extra += f"\nSynced to bucket: {sync_result['bucket_uri']}"
            else:
                extra += f"\nBucket sync FAILED: {sync_result.get('message')}"
        # Versioned dataset-repo publish (Phase 4 hand-off).
        if push_to_hub or state.get("push_to_hub", False):
            push_result = recorder.push_to_hub(tags=["strands-robots", "sim"])
            if push_result and push_result.get("status") == "success":
                extra += "\nPushed to HuggingFace Hub"
            elif push_result:
                extra += f"\npush_to_hub FAILED: {push_result.get('message')}"

        self._release_dataset_recorder(state)
        # The four facts of THIS save, kept past that teardown so
        # get_recording_status can answer from them. It reads the trajectory
        # mirror the release just cleared, so a 37-frame save was reported as
        # "last episode: 0 steps" - the very sentence a sim that never recorded
        # gives. They travel as one mapping because they describe one save: a
        # reader must never pair this id with another save's counts.
        state["last_save"] = {
            "repo_id": repo_id,
            "root": str(root),
            "frame_count": frame_count,
            "episode_count": episode_count,
        }

        # #708 - if recorder.episode_count and parquet disagree, surface
        # it in the human-readable text too so an operator scanning the
        # status log sees the gate firing.
        if episode_count_mismatch:
            text_episode_note = (
                f"\n[#708 gate] recorder reported {episode_count_mismatch_orig} "
                f"episodes but parquet has {episode_count}. Trusting parquet."
            )
        else:
            text_episode_note = ""

        # A partial best-effort loss is reported rather than refused:
        # ``strict=False`` documents dropping a failed write and completing, so
        # the session is a success that is short by a measured amount.
        if dropped:
            logger.warning(
                "stop_recording: %d frame(s) were dropped by this session (strict=False); "
                "the dataset holds %d of the %d frames the recorder was fed",
                dropped,
                frame_count,
                frame_count + dropped,
            )
            text_dropped_note = (
                f"\n{dropped} frame(s) failed to write and were dropped (strict=False): "
                f"the dataset holds {frame_count} of the {frame_count + dropped} frames recorded"
            )
        else:
            text_dropped_note = ""

        session_note = (
            f" (+{captured} frames, +{max(0, episode_count - episodes_at_start)} episode(s) this session)"
            if frames_at_start > 0
            else ""
        )
        text = (
            f"Episode saved to LeRobotDataset\n"
            f"{repo_id} -- {frame_count} frames, {episode_count} episode(s){session_note}"
            f"{text_episode_note}{text_dropped_note}\n"
            f"Local: {root}{extra}"
        )

        return {
            "status": "success",
            "content": [
                {"text": text},
                {
                    "json": {
                        "repo_id": repo_id,
                        "frame_count": frame_count,
                        "episode_count": episode_count,
                        "parquet_episode_count": parquet_episode_count,
                        "episode_count_mismatch": episode_count_mismatch,
                        "dropped_frame_count": dropped,
                        "root": root,
                    }
                },
            ],
        }

    def _stop_recording_idle(
        self,
        *,
        push_to_hub: bool,
        bucket: str | None,
        run_id: str | None,
    ) -> dict[str, Any]:
        """Handle ``stop_recording`` when no recording session is open.

        A bare ``stop_recording()`` stays the idempotent success no-op so
        callers (including ``run_policy``'s finally block) can invoke it
        unconditionally. But when the caller passed upload kwargs, dropping
        them behind a ``status="success"`` is a silent-drop bug (AGENTS.md:
        "Forward all advertised kwargs" / "No silent defaults on error") - the
        agent believes data was uploaded when nothing happened:

        * ``bucket=``: sync the last dataset this sim finalized (the
          ``last_dataset_root`` stashed at ``start_recording``). This is the
          documented "call stop_recording(bucket=...) again as the daily
          sync" workflow - the sync only needs the on-disk dataset, not a
          live recorder. Errors when this sim never recorded anything.
        * ``push_to_hub=True``: structured error - publishing a versioned
          dataset repo requires the recorder of an open session.
        * ``run_id=`` without ``bucket=``: structured error - it only applies
          together with ``bucket=``.
        """
        if not push_to_hub and not bucket and not run_id:
            return {"status": "success", "content": [{"text": "Was not recording."}]}

        if push_to_hub or (run_id and not bucket):
            detail = "push_to_hub=True" if push_to_hub else f"run_id={run_id!r} without bucket="
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"stop_recording: {detail} given but no recording session is "
                            "open - nothing was published or synced. push_to_hub requires "
                            "an open session (start_recording -> run_policy -> "
                            "stop_recording(push_to_hub=True)); run_id= only applies "
                            "together with bucket=."
                        )
                    }
                ],
            }

        state = self._recording_state()
        last_root = state.get("last_dataset_root") if state is not None else None
        if not last_root:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"stop_recording: bucket={bucket!r} given but no recording "
                            "session is open and this sim has no previously recorded "
                            "dataset to sync - nothing was uploaded. Record first "
                            "(start_recording -> run_policy -> stop_recording), or pass "
                            "bucket= on the stop_recording call that closes the session."
                        )
                    }
                ],
            }

        # Every no-bucket combination returned above, so bucket is set here.
        assert bucket is not None
        sync_result = sync_dataset_to_bucket(str(last_root), bucket, run_id=run_id)
        if sync_result.get("status") == "success":
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            "Was not recording; synced the last recorded dataset instead.\n"
                            f"Local: {last_root}\n"
                            f"Synced to bucket: {sync_result['bucket_uri']}"
                        )
                    }
                ],
            }
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        "stop_recording: was not recording, and syncing the last recorded "
                        f"dataset ({last_root}) to bucket {bucket!r} FAILED: "
                        f"{sync_result.get('message')}"
                    )
                }
            ],
        }

    def save_episode(self) -> dict[str, Any]:
        """Close the current episode and start a fresh one in the same session.

        This is the explicit episode-boundary primitive for multi-episode
        recording. The documented collection workflow is one ``run_policy``
        call per episode:

            sim.start_recording(repo_id=..., task=...)
            for _ in range(n_episodes):
                sim.run_policy(robot_name=..., n_steps=...)
                sim.save_episode()   # flush this rollout as its own episode
                sim.reset()          # next rollout starts from the scene pose
            sim.stop_recording()

        ``run_policy(n_episodes=N)`` does all three and is the first-class API
        for the common case; drive the loop yourself only when episodes need
        different instructions, randomization or conditional logic.

        Without this call, every ``run_policy`` rollout in a session appends to
        the SAME buffer, so ``stop_recording`` flushes them as a single
        ``episode_index=0`` (1200 steps land in one episode instead of N). Each
        ``save_episode`` writes the buffered frames as a distinct episode with
        its own ``episode_index`` / ``length`` / ``from_index`` / ``to_index``
        and resets the per-episode frame buffer; ``stop_recording`` flushes any
        trailing rollout automatically, so a final ``save_episode`` is optional.

        The ``reset()`` is not optional bookkeeping. This method cuts a dataset
        episode boundary; it does not re-initialize the world. Without it the
        next rollout begins wherever the last one left the robot, so the dataset
        has the requested episode COUNT but a bimodal set of recorded start
        states - episode 0 from the scene's reset pose and every later episode
        from a pose the robot is never reset into. ``verify_dataset_episodes``
        counts episodes and passes either way, so nothing downstream reports it.
        (``reset()`` is itself an episode boundary while recording: it flushes
        buffered frames before teleporting, so the explicit ``save_episode``
        above is what makes the boundary unconditional rather than what creates
        it - see ``docs/recording.md``.)

        Per-episode stats (LeRobot computes ``stats.json`` per episode, then
        aggregates) stay correct because each rollout's frames are isolated to
        their own episode across the ``reset()`` teleport between rollouts.

        Idempotent on an empty buffer: when no frames have been captured since
        the last boundary (or since ``start_recording``), it succeeds with a
        "no frames to flush" message rather than tripping LeRobot's
        "add frames before add_episode" guard, so callers can invoke it
        unconditionally inside a loop.

        Returns:
            Standard status dict. On success the ``content`` text reports the
            episode index and frame count; a structured ``status="error"`` is
            returned when no recording is active or the underlying flush fails.
        """
        state = self._recording_state()
        if state is None or not state.get("recording", False):
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "save_episode: not recording. Call start_recording first, "
                            "then run_policy -> save_episode per episode -> stop_recording."
                        )
                    }
                ],
            }

        recorder = state.get("dataset_recorder", None)
        if recorder is None:
            return {"status": "error", "content": [{"text": "No dataset recorder active."}]}

        pending = getattr(recorder, "episode_frame_count", 0)
        if pending <= 0:
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            "save_episode: no frames to flush (buffer empty). Run a policy "
                            "while recording before closing an episode."
                        )
                    }
                ],
            }

        save_result = recorder.save_episode()
        if isinstance(save_result, dict) and save_result.get("status") == "error":
            # The recorder marks itself closed on a failed flush (the LeRobot
            # episode buffer is in an undefined state); drop it so callers do
            # not keep appending into a poisoned recorder.
            state["recording"] = False
            self._release_dataset_recorder(state)
            return {
                "status": "error",
                "content": [{"text": f"save_episode failed: {save_result.get('message')}"}],
            }

        # Reset the in-memory trajectory mirror so get_recording_status reports
        # the NEXT episode from zero (matching the recorder's per-episode reset).
        state["trajectory"] = []

        episode = save_result.get("episode")
        ep_frames = save_result.get("episode_frames")
        total = save_result.get("total_frames")
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Episode {episode} saved -- {ep_frames} frames "
                        f"({total} total across dataset). Buffer reset for the next episode."
                    )
                }
            ],
        }

    def _flush_open_episode_before_reset(self) -> dict[str, Any] | None:
        """Close the open dataset episode before a reset re-initializes the world.

        A reset is the start of a new rollout, so the frames buffered since the
        last episode boundary belong to the rollout that just ended. Flushing
        them here is what makes :meth:`save_episode`'s "``reset()`` is itself an
        episode boundary while recording" true, and ``docs/recording.md`` states
        the same rule without naming a backend. Without it a
        ``run_policy`` + ``reset`` collection loop appends every rollout to the
        same buffer, and ``stop_recording`` flushes the lot as a single
        ``episode_index=0`` whose frames span every teleport in between -
        ``total_episodes`` stuck at 1 for a dataset downstream training and eval
        slice by episode. Nothing reports it: the episode count the recorder and
        the parquet agree on is 1, so ``stop_recording``'s own
        author-versus-parquet gate passes, and ``verify_dataset_episodes``
        counts episodes rather than comparing them to the rollouts that were
        run.

        Owned here rather than per backend because all three concrete engines
        reset the same recording session through the same
        :meth:`_recording_state` accessor: a copy per backend is what left two
        of them without the boundary while the shared docstring promised it.

        Returns:
            ``None`` when there is nothing to flush - no recording is open, or
            no frame has been captured since the last boundary - so a reset that
            is not preceded by recorded frames is unaffected. Otherwise the
            :meth:`save_episode` result: a success envelope whose text names the
            episode just written, for the caller to quote, or an error envelope
            the caller must return instead of resetting, because a failed flush
            leaves the recorder closed and its buffer in an undefined state.
        """
        state = self._recording_state()
        if state is None or not state.get("recording", False):
            return None
        recorder = state.get("dataset_recorder", None)
        # ``save_episode`` reports an empty buffer as a success ("no frames to
        # flush"), which would add a note to every reset in a session. Ask the
        # recorder first so a reset with nothing buffered reads exactly as it
        # did before there was a boundary to cut.
        if getattr(recorder, "episode_frame_count", 0) <= 0:
            return None
        return self.save_episode()

    def stream_dataset(self, repo_id: str, **kwargs: Any) -> "StreamingDatasetReader":
        """Open a streaming reader for a LeRobotDataset - read frames straight
        from the Hub (or a local root) with no full materialization.

        This is the in-process counterpart to ``start_recording`` /
        ``stop_recording``: where those WRITE a dataset, ``stream_dataset``
        READS one back lazily for eval / replay / inspection. Training scripts
        can instead use ``lerobot-train --dataset.streaming=true`` which uses
        the same underlying StreamingLeRobotDataset.

        Sugar for the module-level :func:`strands_robots.stream_dataset` -
        reading a dataset does not require a simulator, so scripts without a
        GL stack should call that function directly.

        Args:
            repo_id: HF dataset id (e.g. ``"lerobot/svla_so100_pickplace"``) or
                a ``repo_id`` that is itself a path, which streams the directory
                it recorded to with no ``root`` restated
                (:func:`~strands_robots.dataset_source.local_dataset_dir`).
            **kwargs: Forwarded to
                :meth:`StreamingDatasetReader.open` - e.g. ``root``,
                ``delta_timestamps``, ``episodes``, ``shuffle`` (which decides
                cross-epoch reproducibility, NOT read order - see that method's
                "Ordering" note), ``buffer_size`` and ``max_num_shards`` (both
                ``1`` to read in capture order),
                ``drop_videos`` (proprio-only,
                torchcodec-free; requires ``delta_timestamps`` with at least one
                non-video key, else ValueError), ``repo_type`` (``"dataset"``
                or ``"bucket"``, forwarded unconditionally: every
                lerobot-bearing extra floors lerobot at
                ``BUCKET_STREAMING_MIN_LEROBOT``, whose constructor accepts the
                keyword, so a below-floor install surfaces lerobot's own
                ``TypeError`` naming it rather than a refusal from here).

        Returns:
            A :class:`~strands_robots.streaming_dataset.StreamingDatasetReader`.

        Example:
            reader = sim.stream_dataset(
                "local/agent_demo", root="/tmp/strands_agent_dataset",
                delta_timestamps={"observation.state": [-0.0667, 0.0],
                                  "action": [0.0, 0.0667]},
                buffer_size=1, max_num_shards=1,  # capture order for replay
            )
            for frame in reader:
                ...
        """
        from strands_robots.streaming_dataset import stream_dataset

        return stream_dataset(repo_id, **kwargs)

    def _already_recording_error(self, verb: str, requested_repo_id: str) -> dict[str, Any] | None:
        """The refusal a second ``start_recording`` gets while one is live, or None.

        Falling through replaced the live recorder object, and the frames it
        had buffered since its last ``save_episode`` went with it - never saved,
        never mentioned; when the new dataset then refused (schema mismatch on
        resume) the caller was left with ``recording`` False and both sessions'
        frames gone. Refusing here leaves the live recording exactly as it was
        and names what ``stop_recording`` will save.

        Every backend's ``start_recording`` calls this first, before it writes
        any session state, so the refusal is one behaviour rather than three - a
        backend that reached its own bookkeeping first silently got the frame
        loss back.

        Both figures are scoped to the LIVE SESSION, because the question the
        caller is about to answer is what THIS session would lose. The frame
        count is the open episode's. The episode count is measured against the
        ``episodes_at_start`` stash :meth:`_arm_dataset_recorder` takes for
        exactly this reason: a resumed recorder's ``episode_count`` is seeded
        with the dataset's totals, so reading it raw reported the episodes a
        PREVIOUS session had saved ("1 episode(s) saved" for a session that had
        saved none) - the reassurance a caller weighing a stop must not be
        given. Same arithmetic and same "this session" wording as
        :meth:`stop_recording`'s own report, so the two cannot disagree.
        """
        state = self._recording_state()
        if state is None:
            return None
        live = state.get("dataset_recorder")
        if not state.get("recording") or live is None:
            return None
        # The id is resolved by the shared accessor (live recorder first, then
        # the target stashed at start_recording), so every backend names the
        # live dataset the same way and none needs a stash of its own.
        live_repo = self._active_dataset_repo_id() or "?"
        pending = int(getattr(live, "episode_frame_count", 0) or 0)
        saved = max(0, int(getattr(live, "episode_count", 0) or 0) - int(state.get("episodes_at_start", 0) or 0))
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"{verb}: already recording '{live_repo}' ({pending} frame(s) buffered since the last "
                        f"saved episode, {saved} episode(s) saved this session). Call stop_recording first - it "
                        f"saves the buffered frames - then {verb} '{requested_repo_id}'. The live recording is "
                        f"untouched."
                    )
                },
                {
                    "json": {
                        "recording": True,
                        "repo_id": live_repo,
                        "frames_buffered": pending,
                        "episodes_saved_this_session": saved,
                        "requested_repo_id": requested_repo_id,
                    }
                },
            ],
        }

    def get_recording_status(self) -> dict[str, Any]:
        """Returns success in every lifecycle state (no world / not
        recording / recording) with a distinguishing message so callers can
        poll it unconditionally without try/except.

        Every branch also carries a json block of the same shape - ``world``,
        ``recording``, ``steps``, ``repo_id``, ``root``, ``last_save`` - so a
        poller reads one mapping instead of parsing three sentences.

        Idle reports the dataset this sim last SAVED, not the trajectory
        mirror: ``stop_recording`` clears that mirror when it releases the
        recorder, so reading it here answered "last episode: 0 steps" for a
        37-frame save - byte-identical to the answer a sim that never recorded
        gives, and the false confirmation of the "stopped before any frames"
        diagnosis ``docs/troubleshooting.md`` sends an operator here to check.
        The recipe it names carries ``root=`` so it is runnable whatever this
        sim recorded afterwards.
        """
        state = self._recording_state()
        if state is None:
            return {
                "status": "success",
                "content": [
                    {"text": "No world. Call create_world to start recording."},
                    {"json": {"world": False, "recording": False, "steps": 0, "last_save": None}},
                ],
            }

        recording = state.get("recording", False)
        steps = len(state.get("trajectory", []))
        last = state.get("last_save")
        last = last if isinstance(last, dict) else None
        payload: dict[str, Any] = {
            "world": True,
            "recording": recording,
            "steps": steps,
            "last_save": last,
        }

        if recording:
            # The live recorder's own id and root, through the seams that
            # prefer it - the dataset being written is the one fact a caller
            # polling an open session cannot get anywhere else.
            payload["repo_id"] = self._active_dataset_repo_id()
            payload["root"] = self._active_dataset_root()
            into = f" into {payload['repo_id']} at {payload['root']}" if payload["repo_id"] and payload["root"] else ""
            # `steps` mirrors the OPEN episode only - it empties at every flush,
            # so read alone it said "0 steps captured" right after
            # run_policy(n_episodes=3) had saved 45 frames. The dataset's own
            # counts stand beside it.
            recorder = self._active_recorder()
            saved_episodes = int(getattr(recorder, "episode_count", 0) or 0)
            saved_frames = max(
                int(getattr(recorder, "frame_count", 0) or 0) - int(getattr(recorder, "episode_frame_count", 0) or 0),
                0,
            )
            payload["open_episode_index"] = saved_episodes
            payload["episodes_saved"] = saved_episodes
            payload["frames_saved"] = saved_frames
            text = (
                f"[recording] {steps} steps buffered in the open episode "
                f"(episode_index {saved_episodes}){into}; {saved_episodes} episode(s) / "
                f"{saved_frames} frames saved so far. reset closes the open episode as its own; "
                "run_policy(n_episodes=N) records N distinct; stop_recording saves the open "
                "episode and closes the dataset."
            )
        elif last is not None:
            text = (
                f"[idle] Not recording. Last saved: {last['repo_id']} - {last['frame_count']} frames, "
                f"{last['episode_count']} episode(s) at {last['root']} "
                f"(replay_episode(repo_id='{last['repo_id']}', root='{last['root']}') reads it back)"
            )
        else:
            text = "[idle] Not recording (nothing saved in this session)"

        return {
            "status": "success",
            "content": [{"text": text}, {"json": payload}],
        }

    def _verify_resume_schema(
        self,
        recorder: Any,
        joint_names: list[str],
        camera_keys: list[str],
        camera_dims: dict[str, tuple[int, int]],
        action_names: list[str] | None = None,
        *,
        fps: int,
    ) -> None:
        """Verify the live scene matches the resumed dataset's on-disk schema.

        ``DatasetRecorder.resume`` inherits the feature schema from disk; it does
        not validate it against the current scene. If the caller added a robot,
        renamed a joint, or changed a camera resolution between episodes, the
        mismatch would only surface as a cryptic per-feature shape error on the
        next ``add_frame``. Compare here and raise a clear schema diff instead.

        Compares the expected ``observation.state`` joint names, each
        ``observation.images.*`` camera (presence + height/width), and the
        dataset frame rate. Best-effort: if the dataset does not expose
        ``features`` / ``fps`` we skip that comparison rather than block a valid
        resume on an unexpected LeRobot layout.

        ``fps`` is checked here because a resumed dataset keeps the rate it was
        created at - ``LeRobotDataset.resume`` takes no ``fps`` - so a differing
        request cannot be honored. Appending anyway timestamps the new frames at
        the on-disk rate while they were captured at the requested one, which
        writes a wrong timebase into the dataset: episodes recorded at different
        cadences become indistinguishable, and a policy trained on them reads
        the wrong dt (and so the wrong velocities) for every appended episode.

        Args:
            recorder: The resumed DatasetRecorder.
            joint_names: The expected ``observation.state`` column names the
                current scene will emit (namespaced for multi-robot scenes;
                includes expanded per-component floating-base columns such as
                ``base_quat.w`` when the scene records a floating base).
            camera_keys: Sanitized camera feature names the current scene emits.
            camera_dims: Map of camera feature name -> (height, width).
            action_names: Action-column names the current scene will emit
                (actuator keys; namespaced for multi-robot scenes). When None
                the action feature is not compared.
            fps: Frame rate the caller asked to record at. Must equal the
                resumed dataset's on-disk rate; keyword-only and required so no
                backend can resume without comparing it.

        Raises:
            ValueError: If the live scene schema diverges from the on-disk one.
        """
        diffs: list[str] = []

        # Frame rate first: it is carried by the dataset metadata rather than
        # the feature dict, so it is comparable even on a LeRobot layout whose
        # ``features`` mapping is missing (the early return below).
        disk_fps = recorder_dataset_fps(recorder)
        if disk_fps is not None and disk_fps != int(fps):
            diffs.append(
                f"dataset fps differs: on-disk={disk_fps} vs requested={int(fps)} "
                f"(a resumed dataset keeps its on-disk rate; pass fps={disk_fps} to append at it)"
            )

        features = getattr(getattr(recorder, "dataset", None), "features", None)
        if not isinstance(features, dict):
            if diffs:
                raise ValueError(_resume_schema_error(diffs))
            return

        state = features.get("observation.state")
        if isinstance(state, dict):
            disk_joints = list(state.get("names") or [])
            if disk_joints and disk_joints != list(joint_names):
                diffs.append(f"observation.state joints differ: on-disk={disk_joints} vs scene={list(joint_names)}")

        if action_names is not None:
            action = features.get("action")
            if isinstance(action, dict):
                disk_action = list(action.get("names") or [])
                if disk_action and disk_action != list(action_names):
                    diffs.append(f"action columns differ: on-disk={disk_action} vs scene={list(action_names)}")

        for cam in camera_keys:
            key = f"observation.images.{cam}"
            disk_cam = features.get(key)
            if not isinstance(disk_cam, dict):
                diffs.append(f"camera '{cam}' is in the scene but not in the on-disk schema")
                continue
            shape = disk_cam.get("shape")
            scene_dim = camera_dims.get(cam)
            if shape and len(shape) == 3 and scene_dim is not None:
                disk_h, disk_w = _camera_height_width(shape, disk_cam.get("names"))
                scene_h, scene_w = scene_dim
                if (int(disk_h), int(disk_w)) != (int(scene_h), int(scene_w)):
                    diffs.append(
                        f"camera '{cam}' resolution differs: on-disk={(disk_h, disk_w)} vs scene={(scene_h, scene_w)}"
                    )

        disk_cams = {k[len("observation.images.") :] for k in features if k.startswith("observation.images.")}
        for cam in disk_cams - set(camera_keys):
            diffs.append(f"camera '{cam}' is in the on-disk schema but not in the current scene")

        if diffs:
            raise ValueError(_resume_schema_error(diffs))
