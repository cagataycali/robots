"""Backend-agnostic policy execution against any ``SimEngine``.

Runs the canonical obs -> act -> step loop using only the public ``SimEngine``
interface. Zero knowledge of the underlying physics engine - MuJoCo, Isaac,
Newton and any future backend get ``run_policy`` / ``replay`` / ``evaluate``
for free by implementing the ``SimEngine`` primitives.

Three entry points:

* :meth:`PolicyRunner.run` - blocking policy execution with optional video.
* :meth:`PolicyRunner.replay` - replay a recorded LeRobotDataset episode.
* :meth:`PolicyRunner.evaluate` - multi-episode evaluation with success metrics.

All three call only these public ``SimEngine`` methods:

* ``get_observation(robot_name)``
* ``send_action(action, robot_name, n_substeps)``
* ``step(n_steps)``
* ``reset()``
* ``render(camera_name, width, height)``

And two public helpers for robot discovery:

* ``list_robots()`` - ordered robot names in the world
* ``robot_joint_names(robot_name)`` - ordered joint names for a robot

Thread safety: ``PolicyRunner`` itself is stateless per invocation. The
underlying ``SimEngine`` is responsible for thread-safety inside its own
methods (e.g. MuJoCo acquires a lock inside ``send_action`` / ``step``).
"""

from __future__ import annotations

import difflib
import logging
import math
import numbers
import os
import random
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots._async_utils import _resolve_coroutine
from strands_robots.policies.base import resolve_chunk_length
from strands_robots.utils import positive_whole_number_error, process_rss_mb, require_optional

if TYPE_CHECKING:
    from strands_robots.policies.base import Policy
    from strands_robots.simulation.base import SimEngine
    from strands_robots.simulation.benchmark import BenchmarkProtocol

from strands_robots.simulation.models import TrajectoryStep
from strands_robots.simulation.safe_output import validate_output_path, video_sandbox_args

logger = logging.getLogger(__name__)


def set_eval_seed(seed: int) -> None:
    """Seed Python / NumPy / torch RNGs for reproducible eval rollouts.

    Seeds Python ``random``, NumPy's legacy global RNG, torch CPU + all CUDA
    devices, and pins cuDNN ``deterministic=True`` / ``benchmark=False``.
    NumPy / torch are imported lazily, so minimal installs without torch work.

    Deliberately narrower than NVIDIA's upstream GR00T ``set_seed``: it does
    NOT set the process-wide ``CUBLAS_WORKSPACE_CONFIG`` env var or
    ``torch.use_deterministic_algorithms``, both of which would leak past the
    eval into unrelated callers. Set those yourself for strict determinism.

    Public API, exported via ``__all__``: ``evaluate_benchmark`` calls it once
    per eval, and standalone rollout drivers can call it directly.
    """
    random.seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch as _torch

        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# Hook signature: called every control step after send_action.
# on_frame(step_idx, observation, action) -> None
OnFrame = Callable[[int, dict[str, Any], dict[str, Any]], None]

# Success function: called after each step during evaluate().
# success_fn(observation) -> bool
SuccessFn = Callable[[dict[str, Any]], bool]


def _extract_frame_ndarray(render_result: dict) -> np.ndarray | None:
    """Decode the PNG bytes emitted by ``SimEngine.render`` into an ndarray.

    Walks the ``{"image": {"source": {"bytes": <str|bytes>}}}`` content block
    (raw bytes legacy, base64 string current) and returns a contiguous
    (H, W, 3) RGB array (alpha dropped). Returns ``None`` when no decodable
    image is found, so the recorder skips the frame rather than aborting.
    """
    if not isinstance(render_result, dict):
        return None
    for block in render_result.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        image = block.get("image")
        if not isinstance(image, dict):
            continue
        source = image.get("source") or {}
        png_bytes = source.get("bytes")
        if png_bytes is None and source.get("data") is not None:
            import base64

            png_bytes = base64.b64decode(source["data"])
        if not png_bytes:
            continue
        # Handle base64-encoded strings (current render() output)
        if isinstance(png_bytes, str):
            import base64

            png_bytes = base64.b64decode(png_bytes)
        try:
            import io

            from PIL import Image

            return np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        except Exception:
            return None
    return None


# Canonical :class:`VideoConfig` field -> the dict keys accepted for it, canonical
# key first followed by the legacy/tool_spec aliases. Single source of truth for
# both the schema check (``VideoConfig.validation_error``) and the value lookup
# (``VideoConfig.from_dict``), so the accepted set cannot drift between the two.
_VIDEO_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "path": ("path", "record_video", "output_path"),
    "fps": ("fps", "video_fps"),
    "camera": ("camera", "video_camera", "camera_name"),
    "width": ("width", "video_width"),
    "height": ("height", "video_height"),
}

_VIDEO_ACCEPTED_KEYS: tuple[str, ...] = tuple(sorted(key for aliases in _VIDEO_KEY_ALIASES.values() for key in aliases))


@dataclass(frozen=True)
class VideoConfig:
    """Configuration for optional MP4 recording during :meth:`PolicyRunner.run`.

    Recording is opt-in: if ``path`` is falsy, no recording occurs and the
    other fields are ignored.

    Attributes:
        path: Output MP4 path (``None``/empty disables recording). LLM-supplied,
            so it is validated before a writer is opened; set
            ``STRANDS_ROBOTS_VIDEO_ROOT`` to confine it to a sandbox.
        fps: Frames per second to write. Capped at ``control_frequency`` so the
            rollout always plays back at real time (at most one frame renders
            per control step; the video cannot be up-sampled).
        camera: Camera name to render from. ``None`` -> backend default.
        width: Render width in pixels.
        height: Render height in pixels.
    """

    path: str | None = None
    fps: int = 30
    camera: str | None = None
    width: int = 640
    height: int = 480

    @property
    def enabled(self) -> bool:
        """``True`` iff an output ``path`` was set; other fields are ignored when off."""
        return bool(self.path)

    @staticmethod
    def _pick(d: dict[str, Any], field: str, default: Any = None) -> Any:
        """First present, non-``None`` value among ``field``'s accepted keys.

        Canonical key first, then legacy aliases. Membership - not truthiness -
        decides, so a caller-supplied ``0`` is returned (and later rejected by
        :meth:`validation_error`) instead of collapsing into ``default``.
        """
        for key in _VIDEO_KEY_ALIASES[field]:
            value = d.get(key)
            if value is not None:
                return value
        return default

    @staticmethod
    def _positive_int_error(value: Any, key: str) -> str | None:
        """Error text when a ``video`` dict value is not a positive whole number.

        Thin binding of the shared frame/pixel-count domain
        (:func:`positive_whole_number_error`) to the ``video:`` message prefix,
        so this dict schema and the plain-MP4 recorder's keyword parameters
        cannot drift apart on what counts as a usable ``fps`` / ``width`` /
        ``height``.

        Args:
            value: The caller-supplied value.
            key: The dict key it came from, used in the message.

        Returns:
            An error message, or ``None`` when the value is usable.
        """
        return positive_whole_number_error(value, key, "video")

    @classmethod
    def validation_error(cls, d: Any) -> str | None:
        """Error text when ``d`` is not a video config this class can honor.

        The config arrives as a free-form dict (LLM tool call or direct API),
        so a silently ignored mistyped key would record nothing - or at the
        wrong settings - while the rollout still reports success. Rejects any
        key outside the accepted set (with a closest-match hint) and any known
        key whose value cannot be honored. ``None`` (recording off) and an
        empty dict are valid. Returns the first problem found, or ``None``
        when the config is usable.
        """
        if d is None:
            return None
        if not isinstance(d, dict):
            return f"video must be a dict of recording options, got {type(d).__name__}."
        accepted = ", ".join(_VIDEO_ACCEPTED_KEYS)
        for key in d:
            if key in _VIDEO_ACCEPTED_KEYS:
                continue
            # Match case-insensitively so "FPS"/"Path" suggest their canonical
            # spelling; the cutoff is deliberately tight so an unrelated key
            # ("filename", "resolution") gets the accepted list rather than a
            # misleading nearest-neighbour.
            close = difflib.get_close_matches(str(key).lower(), _VIDEO_ACCEPTED_KEYS, n=1, cutoff=0.7)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            return f"video: unknown key {key!r}.{hint} Accepted keys: {accepted}."
        for field in ("path", "camera"):
            value = cls._pick(d, field)
            if value is not None and not isinstance(value, str):
                return f"video: {field} must be a string, got {value!r}."
        for field in ("fps", "width", "height"):
            value = cls._pick(d, field)
            if value is None:
                continue
            if error := cls._positive_int_error(value, field):
                return error
        return None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> VideoConfig | None:
        """Build from a plain dict (tool_spec dispatcher path). ``None``/empty -> ``None``.

        Accepts canonical keys and the legacy/tool_spec aliases.

        Raises:
            ValueError: When ``d`` carries a key or value that cannot be
                honored (see :meth:`validation_error`). Public entry points
                pre-check and return a structured tool error, so this raise
                guards direct construction.
        """
        if not d:
            return None
        if error := cls.validation_error(d):
            raise ValueError(error)
        return cls(
            path=cls._pick(d, "path"),
            fps=int(cls._pick(d, "fps", 30)),
            camera=cls._pick(d, "camera"),
            width=int(cls._pick(d, "width", 640)),
            height=int(cls._pick(d, "height", 480)),
        )


class _RolloutVideoWriter:
    """Per-rollout MP4 writer: validate the (LLM-supplied) output path, probe
    the camera, then append frames at the requested fps cadence.

    Shared by :meth:`PolicyRunner.run` and the evaluation loops so the
    security-sensitive path validation and the frame-capture cadence have one
    source of truth.
    """

    def __init__(
        self,
        sim: Any,
        video: VideoConfig,
        writer: Any,
        resolved_path: str,
        control_frequency: float,
    ) -> None:
        self.sim = sim
        self.video = video
        self.path = resolved_path
        self._writer = writer
        self.frame_count = 0
        self._frame_interval = control_frequency / max(video.fps, 1)
        self._next_frame_step = 0.0

    @classmethod
    def open(
        cls, sim: Any, video: VideoConfig | None, control_frequency: float
    ) -> tuple[_RolloutVideoWriter | None, dict[str, Any] | None]:
        """Return ``(writer, error)``.

        ``(None, None)``   -> recording disabled (``video`` is falsy); proceed.
        ``(None, error)``  -> setup failed; the caller returns ``error`` verbatim.
        ``(writer, None)`` -> writer ready.
        """
        if video is None or not video.enabled:
            return None, None
        # video.enabled guarantees video.path is a non-empty str; narrow for mypy.
        assert video.path is not None
        # video.path is LLM-supplied: reject shell metacharacters, backslash
        # separators, ".." traversal, and a symlinked target before we makedirs +
        # open a writer on it. Absolute paths stay allowed (the historic
        # contract); set STRANDS_ROBOTS_VIDEO_ROOT to sandbox them.
        _sb_root, _allow_abs = video_sandbox_args()
        try:
            resolved = str(
                validate_output_path(video.path, sandbox_root=_sb_root, allow_abs=_allow_abs, label="video path")
            )
        except ValueError as _e:
            return None, {"status": "error", "content": [{"text": f"video recording: {_e}"}]}

        # Pre-validate the camera name ONCE before the step loop. This surfaces
        # "camera not found" as a clean up-front error rather than silently
        # writing a 0-byte MP4 (sim.render() returns status=error, the rollout
        # runs to completion, and the user gets an empty file with no hint).
        probe_cam = video.camera or "default"
        try:
            _probe = sim.render(camera_name=probe_cam, width=video.width, height=video.height)
        except Exception as e:
            return None, {
                "status": "error",
                "content": [{"text": f"Video recording requested but render probe crashed: {e}"}],
            }
        if _probe.get("status") != "success":
            probe_text = (_probe.get("content") or [{}])[0].get("text", "")
            return None, {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Video recording requested but camera "
                            f"'{probe_cam}' is not renderable.\n"
                            f"{probe_text}\n"
                            "Hint: robot cameras are namespaced, e.g. a "
                            "camera named 'side' inside robot 'arm1' compiles "
                            "as 'arm1/side'. Pass video={'camera': 'arm1/side', ...}."
                        )
                    }
                ],
            }

        imageio = require_optional(
            "imageio",
            pip_install="imageio imageio-ffmpeg",
            extra="sim-mujoco",
            purpose="video recording",
        )
        os.makedirs(os.path.dirname(os.path.abspath(resolved)), exist_ok=True)
        # A rollout renders at most one frame per control step, so an fps above
        # control_frequency cannot add frames - it would only make the MP4 play
        # back faster than real time. Cap the writer fps at control_frequency;
        # a lower fps down-samples and is already real time.
        write_fps = video.fps
        if control_frequency > 0 and video.fps > control_frequency:
            write_fps = max(1, round(control_frequency))
            logger.warning(
                "Video fps=%d exceeds control_frequency=%.1f Hz; a rollout can "
                "render at most one frame per control step, so the MP4 is "
                "written at %d fps to play back at real time (requesting a "
                "higher fps would only speed the video up).",
                video.fps,
                control_frequency,
                write_fps,
            )
        writer = imageio.get_writer(  # type: ignore[attr-defined]
            resolved, fps=write_fps, quality=8, macro_block_size=1
        )
        return cls(sim, video, writer, resolved, control_frequency), None

    def capture(self, step_count: int) -> None:
        """Append one frame if the fps cadence is due at ``step_count``.

        Call once per applied control step; the cadence (``control_frequency /
        fps``) decides which steps actually render. A render/decode failure is
        skipped silently rather than aborting the rollout (a renderer hiccup
        must not kill training).
        """
        if step_count < self._next_frame_step:
            return
        frame = self.sim.render(
            camera_name=self.video.camera or "default",
            width=self.video.width,
            height=self.video.height,
        )
        img_arr = _extract_frame_ndarray(frame)
        if img_arr is not None:
            self._writer.append_data(img_arr)
            self.frame_count += 1
        self._next_frame_step += self._frame_interval

    def close(self) -> None:
        self._writer.close()


# on_frame hooks that raise are logged at WARN - user telemetry must not kill
# the rollout. But a hook that fails on EVERY step would silently record an
# empty dataset, so after this many *consecutive* failures the runner fails the
# episode loudly. Overridable via ``max_onframe_failures`` on ``PolicyRunner.run``.
_MAX_CONSECUTIVE_ONFRAME_FAILURES = 5

# Fail-fast probe window for 100%-unresolved action keys. If EVERY action step
# in this opening window drives zero actuators, the rollout can never move the
# robot, so :meth:`PolicyRunner.run` raises at the probe boundary instead of
# burning the whole episode. Once any step resolves a single key the probe is
# permanently disarmed.
_FAIL_FAST_PROBE_STEPS = 3


def _recorder_module_file() -> str | None:
    """Path of the imported dataset-recorder module, or ``None`` when unimported.

    Resolved from ``sys.modules`` so the module is named by its dotted path and
    its location is whatever the running install actually loaded. An unimported
    module cannot appear in a traceback, so ``None`` is a complete answer.
    """
    module = sys.modules.get("strands_robots.dataset_recorder")
    filename = getattr(module, "__file__", None)
    return filename.replace("\\", "/") if isinstance(filename, str) else None


def _is_recorder_frame_failure(exc: BaseException) -> bool:
    """Did this ``on_frame`` exception come from the dataset recorder?

    The ``on_frame`` tolerance (``_MAX_CONSECUTIVE_ONFRAME_FAILURES``) is right
    for user telemetry and wrong for a recorder failure: lerobot derives frame
    indices positionally, so a dropped frame is renumbered away instead of
    leaving a gap, and the episode silently encodes a trajectory that was never
    executed. So the two have to be told apart.

    Detected by walking the traceback for a frame defined in the
    :mod:`strands_robots.dataset_recorder` module (or any ``add_frame`` call),
    rather than by exception TYPE: lerobot raises plain ``ValueError`` for a
    feature-shape mismatch, which a user hook could raise too. The module's file
    is read from the imported module itself, so no source path is hardcoded.

    Args:
        exc: The exception raised out of the ``on_frame`` hook.

    Returns:
        ``True`` when the recorder was in the failing call stack.
    """
    recorder_file = _recorder_module_file()
    tb = exc.__traceback__
    while tb is not None:
        code = tb.tb_frame.f_code
        filename = code.co_filename.replace("\\", "/")
        if recorder_file is not None and filename == recorder_file:
            return True
        if code.co_name == "add_frame":
            return True
        tb = tb.tb_next
    # Also follow an explicit cause/context chain (a hook that wraps the
    # recorder error rather than letting it propagate).
    for chained in (exc.__cause__, exc.__context__):
        if chained is not None and chained is not exc and _is_recorder_frame_failure(chained):
            return True
    return False


def _extract_result_json(result: object) -> dict[str, Any] | None:
    """Return the ``{"json": {...}}`` payload from a backend status dict.

    ``send_action`` reports unresolved action keys via a ``json`` content block
    (``{"unresolved_keys": [...], "applied": [...]}``). Returns that mapping, or
    ``None`` when the result carries no structured block.
    """
    if not isinstance(result, dict):
        return None
    for block in result.get("content", []) or []:
        if isinstance(block, dict):
            payload = block.get("json")
            if isinstance(payload, dict):
                return payload
    return None


def _validate_action_key_map(action_key_map: Any) -> dict[str, Any] | None:
    """Reject a ``replay`` ``action_key_map`` no backend could honor.

    The map binds recorded action-vector indices to the action keys
    ``send_action`` resolves, so it must be a non-empty ordered collection of
    unique strings. Rejected: a bare ``str`` (consumed one key per character),
    a non-string entry, a duplicate key (a later index silently overwrites the
    earlier one), and an empty collection. ``None`` selects the default
    ``robot_action_keys`` ordering and is always accepted.

    Returns:
        An agent-tool error dict describing the problem, or ``None`` when the
        map is usable.
    """
    if action_key_map is None:
        return None

    def _error(text: str) -> dict[str, Any]:
        return {"status": "error", "content": [{"text": f"replay: {text}"}]}

    if isinstance(action_key_map, str | bytes):
        return _error(
            f"action_key_map must be a list of action keys, not a bare string (got {action_key_map!r}); "
            "a string is consumed one character per action index."
        )
    if not isinstance(action_key_map, list | tuple):
        return _error(f"action_key_map must be a list or tuple of action keys (got {type(action_key_map).__name__}).")
    if not action_key_map:
        return _error("action_key_map is empty; pass one action key per recorded action-vector index.")
    bad = [key for key in action_key_map if not isinstance(key, str)]
    if bad:
        return _error(f"action_key_map entries must be action-key strings; got non-string entries {bad!r}.")
    duplicates = sorted({key for key in action_key_map if action_key_map.count(key) > 1})
    if duplicates:
        return _error(
            f"action_key_map has duplicate keys {duplicates}; each recorded action index needs its own key "
            "(a repeated key silently overwrites the earlier index's value)."
        )
    return None


class CooperativeStop(BaseException):
    """Raised by an ``on_frame`` hook to cooperatively stop a run.

    Inherits ``BaseException`` (not ``Exception``) so hook authors don't
    accidentally swallow it with a broad ``except Exception``. Honored by
    ``PolicyRunner.run`` and by the ``evaluate``/``evaluate_benchmark``
    paths: it is caught at the episode loop to return a normal
    stopped-early success result (``stopped_early=True``) rather than
    propagating as an uncaught exception.
    """


class _ChunkPipeline:
    """Yield ``(observation, action)`` pairs for a policy rollout.

    Two acquisition strategies behind one iterator:

    * **synchronous** (``async_rtc=False``): query the policy, fully drain the
      returned chunk, then re-query - inference never overlaps execution.
    * **async-RTC** (``async_rtc=True``): while the current chunk drains, fire
      the next ``get_actions`` on a single background worker once the chunk is
      ~50% consumed, then atomically swap it in at the seam - a policy whose
      inference latency is <= the chunk's execution time pays (almost) zero
      visible stall, the way an async real-time controller hides latency on
      real hardware.

    Thread-safety: the worker only ever calls the supplied ``query_chunk``
    (pure policy inference); the observation for a prefetch is captured on the
    CONSUMING thread before the worker is submitted; and the sim is only ever
    stepped by the consumer - no MuJoCo/Warp array is touched from two threads
    at once.

    The pipeline is an unbounded iterator - the consumer terminates by breaking
    out of the loop. Use it as a context manager so the inference worker is
    always joined on exit, even when the consumer breaks mid-chunk.

    ``chunks_acquired`` / ``prefetch_hits`` / ``prefetch_blocks`` make latency
    masking provable from the result payload without grepping logs.
    """

    def __init__(
        self,
        query_chunk: Callable[[dict[str, Any], int], list[dict[str, Any]]],
        observation_fn: Callable[[], dict[str, Any]],
        *,
        async_rtc: bool,
        rtc_inference_timeout_s: float | None,
    ) -> None:
        self._query_chunk = query_chunk
        self._observation_fn = observation_fn
        self._async_rtc = async_rtc
        self._timeout = rtc_inference_timeout_s
        self.chunks_acquired = 0
        self.prefetch_hits = 0
        self.prefetch_blocks = 0
        self._executor: Any = None

    def __enter__(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        return self._iter_async() if self._async_rtc else self._iter_sync()

    def __exit__(self, *exc: object) -> None:
        # Join any in-flight inference so no background thread touches the
        # policy/sim after the rollout returns (the caller may immediately
        # reset() or destroy() the world). Returns None so an exception raised
        # inside the ``with`` block (e.g. a prefetch timeout) propagates.
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def _iter_sync(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        while True:
            observation = self._observation_fn()
            # The world is paused during inference on the synchronous path, so
            # the policy observed exactly 0 control steps of delay.
            chunk = self._query_chunk(observation, 0)
            self.chunks_acquired += 1
            if not chunk:
                raise RuntimeError("policy returned an empty action chunk; cannot run rollout")
            for action in chunk:
                yield observation, action

    def _iter_async(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        from concurrent.futures import Future, ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeout

        def _swap_in(fut: Future[list[dict[str, Any]]]) -> list[dict[str, Any]]:
            # A prefetch HIT means inference already finished (the seam is
            # invisible); a BLOCK means inference ran slower than the chunk's
            # execution - the actionable "shorten the chunk / earlier trigger"
            # signal, so log it. A hard timeout turns a stuck model into a
            # structured error instead of an unbounded sim hang.
            if fut.done():
                self.prefetch_hits += 1
            else:
                self.prefetch_blocks += 1
                logger.warning(
                    "async-RTC seam starvation: prefetched chunk was not ready at the swap "
                    "point (inference slower than chunk execution). Blocking on it; consider a "
                    "shorter chunk or an earlier prefetch trigger."
                )
            try:
                return fut.result(timeout=self._timeout)
            except FuturesTimeout as e:
                raise RuntimeError(
                    f"async-RTC prefetch exceeded rtc_inference_timeout_s={self._timeout}s; "
                    f"policy inference is stuck. Raise the timeout or check the policy/server."
                ) from e

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-prefetch-eval")
        cur_obs = self._observation_fn()
        cur_chunk = self._query_chunk(cur_obs, 0)
        self.chunks_acquired += 1
        if not cur_chunk:
            raise RuntimeError("policy returned an empty action chunk; cannot run rollout")
        idx = 0
        prefetch_trigger = max(1, len(cur_chunk) // 2)
        prefetch: Future[list[dict[str, Any]]] | None = None
        prefetch_obs: dict[str, Any] | None = None

        while True:
            if idx >= len(cur_chunk):
                if prefetch is not None:
                    cur_chunk = _swap_in(prefetch)
                    if prefetch_obs is not None:
                        cur_obs = prefetch_obs
                    prefetch = None
                    prefetch_obs = None
                    self.chunks_acquired += 1
                else:
                    # Chunk too short to have triggered a prefetch -> one
                    # synchronous re-query.
                    cur_obs = self._observation_fn()
                    cur_chunk = self._query_chunk(cur_obs, 0)
                    self.chunks_acquired += 1
                if not cur_chunk:
                    # Drop-and-requery: a prefetched chunk arriving empty (a
                    # transient policy hiccup) degrades to ONE synchronous
                    # re-query before erroring, rather than killing an
                    # otherwise-healthy rollout on a single empty result.
                    logger.warning("async-RTC chunk arrived empty; falling back to one synchronous re-query.")
                    cur_obs = self._observation_fn()
                    cur_chunk = self._query_chunk(cur_obs, 0)
                    self.chunks_acquired += 1
                    if not cur_chunk:
                        raise RuntimeError(
                            "policy returned an empty action chunk twice (prefetch + synchronous "
                            "re-query); cannot continue rollout"
                        )
                idx = 0
                prefetch_trigger = max(1, len(cur_chunk) // 2)
                continue

            if prefetch is None and idx >= prefetch_trigger:
                prefetch_obs = self._observation_fn()
                # The prefetched chunk first applies after the remaining steps of
                # the current chunk drain - a known integer independent of how
                # long inference actually takes in wall-clock time.
                observed_delay = max(0, len(cur_chunk) - prefetch_trigger)
                prefetch = self._executor.submit(self._query_chunk, prefetch_obs, observed_delay)

            yield cur_obs, cur_chunk[idx]
            idx += 1


class PolicyRunner:
    """Backend-agnostic policy execution against a ``SimEngine``.

    Construct with any ``SimEngine`` and call :meth:`run`, :meth:`replay`, or
    :meth:`evaluate`. The runner is stateless across calls - safe to reuse.

    Args:
        sim: Any ``SimEngine`` implementation.
    """

    def __init__(self, sim: SimEngine):
        self.sim = sim

    def _control_substeps(self, control_frequency: float, override: int | None = None) -> int:
        """Physics steps per applied action so a position-servo arm tracks the
        full control period (1/control_frequency), not a single physics dt.

        Shared by :meth:`run`, :meth:`replay` and the eval paths so every route
        integrates the same wall-clock period per action.

        Args:
            control_frequency: Control-loop rate in Hz, combined with the
                backend's physics timestep to derive the substep count.
            override: Explicit substeps per action, or ``None`` to derive.

        Returns:
            Physics steps to advance per applied action (always ``>= 1``).

        Raises:
            ValueError: If ``override`` is not a positive integer - clamping or
                truncating it would silently reinstate the under-integration
                this helper exists to prevent. Public entry points reject such
                values first; this raise covers callers driving
                ``PolicyRunner`` directly.
        """
        if override is not None:
            if isinstance(override, bool) or not isinstance(override, int) or override < 1:
                raise ValueError(f"control_substeps must be a positive integer, got {override!r}.")
            return override
        dt = None
        try:
            dt = self.sim.physics_timestep()
        except Exception:  # noqa: BLE001 - never fail a run on a probe
            dt = None
        if dt and dt > 0 and control_frequency > 0:
            return max(1, round((1.0 / control_frequency) / dt))
        return 1

    def _achieved_control_frequency(self, control_frequency: float, n_substeps: int) -> float:
        """The control rate the loop can actually run, given integer substeps.

        A control period is realisable only as a whole number of physics steps,
        so the loop runs at ``1/(n_substeps*dt)``, not the requested rate. RTC
        providers convert wall-clock latency into consumed action steps using
        the rate reported via ``set_control_frequency``, so callers must pass
        this achieved rate; a mismatch is logged once per rollout so the
        shortened horizon is diagnosable rather than silent.
        """
        try:
            dt = self.sim.physics_timestep()
        except Exception:  # noqa: BLE001 - never fail a run on a probe
            return control_frequency
        if not dt or dt <= 0 or n_substeps < 1:
            return control_frequency
        achieved = 1.0 / (n_substeps * dt)
        if abs(achieved - control_frequency) > 1e-9:
            logger.warning(
                "control_frequency=%.6g Hz is not realisable at physics timestep %.6g s: a control "
                "period must be a whole number of physics steps, so the loop runs at %.6g Hz "
                "(%d substeps). Sim time advances %.4gx the requested duration. Pass a frequency "
                "that divides 1/%.6g (or set control_substeps explicitly) for an exact horizon.",
                control_frequency,
                dt,
                achieved,
                n_substeps,
                control_frequency / achieved,
                dt,
            )
        return achieved

    def _finalize_recorder_episode(self) -> None:
        """Roll the attached dataset_recorder over to a new episode.

        The recorder on ``_world._backend_state`` keeps a single open LeRobot
        episode buffer; only ``save_episode`` rolls it over. Calling this at
        the end of each eval episode forces a per-episode boundary - without
        it, a multi-episode loop silently records ONE giant episode of
        ``n_episodes * max_steps`` frames. No-op when no recorder is attached
        or the episode buffer is empty (LeRobot raises on saving zero frames).
        """
        try:
            world = getattr(self.sim, "_world", None)
            if world is None:
                return
            recorder = world._backend_state.get("dataset_recorder")
        except AttributeError:
            return
        if recorder is None:
            return
        pending = getattr(recorder, "episode_frame_count", 0)
        if pending <= 0:
            return
        try:
            result = recorder.save_episode()
            if isinstance(result, dict) and result.get("status") != "success":
                logger.warning(
                    "Per-episode save_episode returned non-success: %s",
                    result.get("message", result),
                )
        except Exception as e:  # noqa: BLE001 - recorder errors must not abort eval
            logger.warning("Per-episode save_episode raised: %s", e)

    # run(): blocking policy execution
    def run(
        self,
        robot_name: str,
        policy: Policy,
        *,
        instruction: str = "",
        duration: float = 10.0,
        n_steps: int | None = None,
        control_frequency: float = 50.0,
        action_horizon: int = 8,
        fast_mode: bool = False,
        video: VideoConfig | None = None,
        on_frame: OnFrame | None = None,
        max_onframe_failures: int | None = None,
        control_substeps: int | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
        async_rtc: bool | None = None,
        rtc_inference_timeout_s: float | None = None,
        stop_when: Callable[[SimEngine], bool] | None = None,
    ) -> dict[str, Any]:
        """Run ``policy`` on ``robot_name`` for ``duration`` seconds.

        Args:
            robot_name: Name of robot in the sim.
            policy: Already-constructed ``Policy`` instance (callers own
                construction so tests can inject mocks).
            instruction: Natural-language instruction forwarded to the policy.
            duration: Seconds to run, interpreted as control steps via
                ``control_frequency``. Used only when ``n_steps`` is None.
            n_steps: Explicit control-step horizon. When set (and > 0) it is
                the exact number of steps executed, bypassing the lossy
                ``int(duration * control_frequency)`` recomputation.
            control_frequency: Target Hz for ``policy.get_actions`` calls.
            action_horizon: Max actions consumed per policy call before
                re-querying the observation. The effective horizon is
                ``max(action_horizon, policy.actions_per_step)`` so a model
                trained for N-step open-loop chunk replay never has its chunk
                truncated below N.
            fast_mode: If True, skip real-time ``time.sleep`` between steps.
            video: Optional :class:`VideoConfig` - set ``video.path`` to enable
                MP4 recording via :meth:`SimEngine.render`.
            on_frame: Optional hook ``(step_idx, obs, action) -> None`` called
                after every ``send_action``. Public extension point for
                recording / telemetry / graceful stop (raise
                :class:`CooperativeStop`).
            max_onframe_failures: Maximum *consecutive* non-``CooperativeStop``
                exceptions from ``on_frame`` before the runner aborts the
                episode (a hook broken on every step would otherwise silently
                produce an empty dataset). ``None`` (default) uses
                ``_MAX_CONSECUTIVE_ONFRAME_FAILURES`` (currently ``5``).
                Non-consecutive failures reset the counter.
            control_substeps: Explicit physics substeps per action, or ``None``
                to derive from the backend timestep (see ``_control_substeps``).
            policy_kwargs: Optional per-call goal payload forwarded verbatim to
                every ``policy.get_actions(obs, instruction, **policy_kwargs)``
                call - carries the well-known goal keys (``target_pose`` /
                ``target_joints`` / ``target_velocity`` / ``world_update``) to
                non-VLA providers that read their goal from kwargs. VLA
                providers ignore unknown kwargs, so forwarding is safe.
                ``None`` forwards no extra kwargs.
            seed: Optional RNG seed for a reproducible single rollout: reseeds
                Python / NumPy / torch / cuDNN via ``set_eval_seed`` and
                forwards ``policy.reset(seed=...)`` so the policy's stochastic
                ops draw from a deterministic state. ``None`` (default) leaves
                RNG state untouched.
            async_rtc: When ``True``, overlap policy inference with action
                execution: once the current chunk is ~50% consumed, the next
                ``get_actions()`` fires on a single background worker with a
                fresh mid-execution observation and is atomically swapped in
                when the chunk runs out, so inference latency up to the chunk's
                execution time costs (almost) zero visible stall at the seam.
                This flag only schedules the overlap; whether the seam is
                additionally BLENDED is a checkpoint-level property of the
                policy (an enabled ``rtc_config``) that the runner never
                touches - chunk-emitting policies without it get the latency
                masking but a plain chunk swap at the seam. ``False`` keeps the
                synchronous chunk-then-drain loop, correct for single-step
                policies and any policy whose ``get_actions`` reads live sim
                state. ``None`` (default) auto-resolves from
                ``policy.is_chunk_emitting()``; an explicit value always wins.
                The policy is only ever invoked from one thread at a time, and
                the runner joins any in-flight inference before returning so no
                thread touches the policy or sim after :meth:`run` exits.
            rtc_inference_timeout_s: Hard per-chunk timeout (seconds) for the
                async-RTC prefetch. When set and a prefetched inference has not
                returned by the time its chunk must be swapped in, the swap
                raises and :meth:`run` returns a structured ``status=error``
                result (with the RTC telemetry block) rather than waiting for
                every remaining chunk of a slow model. The runner still joins the
                single in-flight worker on shutdown (Python cannot forcibly kill a
                running thread, and a leaked worker would touch the policy after
                :meth:`run` returns), so the abort is bounded by ONE inference,
                not the whole rollout. ``None`` (default) waits without a deadline
                (historical behaviour). Ignored on the synchronous path.
            stop_when: Optional semantic early-return condition - a callable
                ``(sim) -> bool`` evaluated against the LIVE sim after every
                applied action, on BOTH the synchronous and async-RTC paths.
                The first ``True`` ends the rollout cleanly with
                ``stopped_reason="predicate"``; the remaining actions of an
                in-flight chunk are dropped, so the early-return latency bound
                is ONE control step regardless of the policy's chunk length
                (on the async-RTC path any in-flight prefetch is still joined
                before :meth:`run` returns). Callers driving the runner
                through :meth:`SimEngine.run_policy` pass a predicate-DSL dict
                compiled via
                :func:`~strands_robots.simulation.benchmark_spec.compile_stop_when`;
                programmatic callers may pass any callable (mirroring
                :meth:`evaluate`'s ``success_fn``). A raising ``stop_when`` is
                fatal (``status="error"``): the caller asked for an
                early-return semantics the runner can no longer honor, and
                silently running to the step budget would misreport the
                rollout. ``None`` (default) preserves the pure step-budget
                horizon.

        Returns:
            ``{"status": "success"|"error", "content": [{"text": ...},
            {"json": {...}}]}``. The ``json`` block is agent-consumable and
            carries the rollout facts as typed fields - ``robot_name``,
            ``policy``, ``instruction``, ``n_steps``, ``steps_used`` (the
            control steps actually executed, equal to ``n_steps``),
            ``elapsed_s``, ``stopped_early``, ``stopped_reason``
            (``"predicate"`` - the ``stop_when`` condition fired; ``"budget"``
            - the step/duration horizon was exhausted; ``"cancelled"`` - a
            cooperative stop, e.g. ``stop_policy``; on ``status="error"``
            results the field is ``"error"``), ``action_errors``,
            ``video_path`` (``None`` when
            no MP4 was written), ``video_frames`` and ``sim_time_s`` (when the
            backend reports sim time) - so callers can self-correct without
            regex-parsing the human-readable ``text``. The block also carries the
            async-RTC telemetry (``rtc_async_enabled``, ``rtc_chunks_acquired``,
            ``rtc_prefetch_hits``, ``rtc_prefetch_blocks``, ``rtc_avg_inference_ms``,
            ``rtc_max_inference_ms``) so latency masking is provable from the
            payload instead of from logs. It also carries the per-actuator
            resolution stats - ``action_resolution_rate`` (a
            ``{actuator_name: fraction_of_steps_driven}`` map) and
            ``partial_action_failure_rate`` (the mean fraction of the robot's
            DOF never driven; ``0.0`` == every actuator moved every step).

            Fail-fast: if EVERY action step in the opening probe window
            (``_FAIL_FAST_PROBE_STEPS``, currently 3) drives zero actuators -
            none of the policy's emitted keys resolve to any of the robot's
            actuators - the rollout can never move the robot, so this returns
            ``status=error`` at the probe boundary instead of running the full
            episode, enumerating the unresolved keys and the robot's valid
            actuator names. A PARTIAL failure (some keys resolve) is
            operational and runs to completion, surfaced via
            ``partial_action_failure_rate``.
        """
        # When a seed is given, reseed the client RNGs once and forward it to
        # the policy (mirrors the per-episode reseed in evaluate()). Default
        # None leaves RNG state untouched.
        if seed is not None:
            set_eval_seed(seed)
            try:
                policy.reset(seed=seed)
            except Exception as e:  # noqa: BLE001 - reset is best-effort
                logger.warning(
                    "policy.reset(seed=%d) raised %s; continuing without policy-side reseed",
                    seed,
                    e,
                )

        # Auto-resolve the async-RTC overlap from the policy's own shape when
        # the caller did not pin it: chunk-emitting policies overlap,
        # single-step policies stay synchronous. getattr so a duck-typed
        # policy_object without is_chunk_emitting() stays synchronous.
        if async_rtc is None:
            _emit = getattr(policy, "is_chunk_emitting", None)
            async_rtc = bool(_emit()) if callable(_emit) else False
            logger.info(
                "async_rtc auto-resolved to %s from %s.is_chunk_emitting()",
                async_rtc,
                type(policy).__name__,
            )

        # RTC telemetry, reported in the result json so latency masking is
        # provable without grepping logs. inference_ms collects every
        # get_actions wall-time (both paths); the prefetch hit/block counters and
        # chunks_acquired are async-only (0 on the synchronous path). list.append
        # is atomic under the GIL, so the worker thread appending an inference
        # time never races the main thread reading the list after shutdown(wait).
        inference_ms: list[float] = []
        rtc_chunks_acquired = 0
        rtc_prefetch_hits = 0
        rtc_prefetch_blocks = 0

        def _rtc_telemetry() -> dict[str, Any]:
            # The async-RTC telemetry block, merged into every result json
            # (success and error) so latency masking is provable from the
            # structured payload without grepping logs. On the synchronous path
            # the prefetch counters stay 0 and only the inference timings carry
            # information.
            _n = len(inference_ms)
            return {
                "rtc_async_enabled": bool(async_rtc),
                "rtc_chunks_acquired": rtc_chunks_acquired,
                "rtc_prefetch_hits": rtc_prefetch_hits,
                "rtc_prefetch_blocks": rtc_prefetch_blocks,
                "rtc_avg_inference_ms": round(sum(inference_ms) / _n, 3) if _n else 0.0,
                "rtc_max_inference_ms": round(max(inference_ms), 3) if _n else 0.0,
            }

        # Video recording lifecycle (path validation + camera probe + writer)
        # lives in _RolloutVideoWriter so run() and evaluate() record identically.
        vwriter, _video_err = _RolloutVideoWriter.open(self.sim, video, control_frequency)
        if _video_err is not None:
            # Every error result carries the stopped_reason="error" json block
            # (the "recorded on ALL exit paths" contract); the writer's error
            # dict is text-only because evaluate() shares it, so tag it here.
            _video_err.setdefault("content", []).append(
                {"json": {"stopped_reason": "error", "steps_used": 0, "n_steps": 0}}
            )
            return _video_err

        stopped_early = False
        # Why the rollout ended, reported in the result json so an agent
        # deciding whether to retry can distinguish "the world reached the
        # goal state" from "the step budget ran out" from "the user cancelled"
        # (stopped_early alone conflates the last two). "budget" is the
        # default (the loop ran its full horizon); the CooperativeStop handler
        # re-tags it "cancelled", a fired stop_when re-tags it "predicate",
        # and every error return reports "error".
        stopped_reason = "budget"
        stop_predicate_fired = False
        # T26: skip camera rendering when the policy does not need images.
        _skip_images = not getattr(policy, "requires_images", True)
        # Open-loop chunk replay consumes H actions from ONE observation, which
        # is the correct pre-action state for the FIRST action only. Re-using
        # it for every recorded frame would pair H frozen observations with H
        # DIFFERENT actions - a temporally-misaligned behavioural-cloning
        # dataset. So when the engine reports an active recording, refresh the
        # observation handed to on_frame per step; inference still consumes the
        # chunk-start observation (correct open-loop replay). Without a
        # recording, keep the historical single-fetch-per-chunk behaviour.
        _is_rec = getattr(self.sim, "_is_recording", None)
        _record_per_step_obs = bool(_is_rec()) if callable(_is_rec) else False
        # Normalise the per-call goal payload once. Forwarded verbatim to every
        # get_actions() call; an empty dict is the historical (no-kwargs) path.
        _policy_kwargs = policy_kwargs or {}

        # Initialize BEFORE try so CooperativeStop never sees unbound names.
        start_time = time.time()
        step_count = 0
        try:
            # Prefer an explicit integer step count: recomputing
            # int(duration * control_frequency) truncates on any frequency
            # that does not divide evenly.
            if n_steps is not None and n_steps > 0:
                total_steps = int(n_steps)
            else:
                # Round rather than truncate: binary float leaves an exact
                # product a hair below the integer (0.58 * 50 ==
                # 28.999999999999996), so int() silently dropped one step.
                total_steps = round(duration * control_frequency)
            action_sleep = 1.0 / control_frequency

            # Advance physics for the FULL control period per action (see
            # _control_substeps) so a position-servo robot actually tracks the
            # commanded target before the next action overwrites ctrl.
            n_substeps = self._control_substeps(control_frequency, control_substeps)
            # Tell the policy the loop's ACHIEVED control rate so RTC providers
            # convert inference latency into the correct number of consumed
            # action steps (see _achieved_control_frequency).
            policy.set_control_frequency(self._achieved_control_frequency(control_frequency, n_substeps))
            logger.info(
                "PolicyRunner: control_frequency=%.1f Hz, physics substeps/action=%d",
                control_frequency,
                n_substeps,
            )
            _action_errors = 0  # count send_action failures (unresolved keys)
            # Per-actuator resolution tracking. Init a counter to 0 for EVERY
            # robot actuator so a never-driven joint surfaces as rate 0.0
            # rather than being absent. ``_total_failure_steps`` counts steps
            # where keys were emitted but NONE resolved - the fail-fast trigger.
            try:
                _robot_actuators = list(self.sim.robot_action_keys(robot_name))
            except Exception:  # noqa: BLE001 - stats are best-effort, never fatal
                _robot_actuators = []
            _actuator_resolved: dict[str, int] = dict.fromkeys(_robot_actuators, 0)
            _total_failure_steps = 0
            _last_unresolved: list[str] = []

            onframe_failure_limit = (
                max_onframe_failures if max_onframe_failures is not None else _MAX_CONSECUTIVE_ONFRAME_FAILURES
            )
            consecutive_onframe_failures = 0

            # Per-action execution body shared by BOTH the synchronous loop and
            # the async-RTC pipeline so they send, record, count and pace
            # identically - only the chunk-ACQUISITION strategy differs between
            # the two paths.
            def _apply(observation: dict[str, Any], action_dict: dict[str, Any]) -> None:
                nonlocal step_count, _action_errors, consecutive_onframe_failures
                nonlocal _total_failure_steps, _last_unresolved

                _send_result = self.sim.send_action(action_dict, robot_name=robot_name, n_substeps=n_substeps)
                _is_error = isinstance(_send_result, dict) and _send_result.get("status") == "error"
                # Resolve which of the robot's actuators this step actually drove
                # and which emitted keys no actuator could absorb. On the success
                # path send_action returns no json block, so every emitted key
                # resolved; on the error path the block enumerates applied /
                # unresolved keys.
                _unresolved: list[str] = []
                if _is_error:
                    _action_errors += 1
                    _json = _extract_result_json(_send_result)
                    if _json is not None:
                        _unresolved = list(_json.get("unresolved_keys", []))
                        _applied = list(_json.get("applied", []))
                    else:
                        # Error without a per-key breakdown (e.g. missing world,
                        # vector length mismatch): treat the whole step as a
                        # 100% failure so it counts toward the fail-fast probe.
                        _applied = []
                        if isinstance(action_dict, dict):
                            _unresolved = list(action_dict.keys())
                elif isinstance(action_dict, dict):
                    _applied = list(action_dict)
                else:
                    # A numeric vector binds positionally to every joint.
                    _applied = list(_robot_actuators)
                for _name in _applied:
                    if _name in _actuator_resolved:
                        _actuator_resolved[_name] += 1
                # A step is a 100%-failure when the policy emitted keys but NONE
                # resolved to an actuator (the robot did not move at all). A
                # PARTIAL failure (some keys resolve) is operational and runs to
                # completion -- reported via partial_action_failure_rate.
                if _is_error and not _applied:
                    _total_failure_steps += 1
                    if _unresolved:
                        _last_unresolved = _unresolved

                if on_frame is not None:
                    try:
                        on_frame(step_count, observation, action_dict)
                        consecutive_onframe_failures = 0
                    except CooperativeStop:
                        # Backend (e.g. MuJoCo) signalled a graceful stop.
                        raise
                    except Exception as e:
                        # A RECORDER failure is not tolerated at all, however
                        # many times it happens. The tolerance below exists for
                        # user telemetry; a dropped dataset frame is different in
                        # kind, because lerobot derives timestamps POSITIONALLY
                        # (frame_index = len(buffer)). So the survivors are
                        # renumbered contiguously, the discontinuity is ERASED,
                        # and the episode reads as a clean trajectory that was
                        # never executed - and passes verify_dataset. Measured: 5
                        # frames with the 3rd rejected recorded
                        # j1=[0.0, 1.0, 3.0, 4.0] under timestamps
                        # [0, .0333, .0667, .1]. Swallowing one of those (the
                        # limit is 5 CONSECUTIVE) silently corrupts the data the
                        # rollout exists to produce, so abort the episode now and
                        # let the caller discard it.
                        if _is_recorder_frame_failure(e):
                            raise RuntimeError(
                                f"dataset recording failed at step {step_count} and the episode "
                                f"cannot be trusted: lerobot renumbers surviving frames "
                                f"contiguously, so a dropped frame leaves no gap in the timestamps "
                                f"and the episode would encode a trajectory that was never "
                                f"executed. Discard this episode. Cause: {e!r}"
                            ) from e
                        # on_frame is user-provided telemetry - never fatal per
                        # call, but a hook failing on every step would silently
                        # record an empty dataset, so fail the episode after
                        # ``onframe_failure_limit`` consecutive failures.
                        consecutive_onframe_failures += 1
                        logger.warning(
                            "on_frame hook failed (%d/%d consecutive): %s",
                            consecutive_onframe_failures,
                            onframe_failure_limit,
                            e,
                        )
                        if consecutive_onframe_failures >= onframe_failure_limit:
                            raise RuntimeError(
                                f"on_frame hook failed {onframe_failure_limit} times in a row; "
                                f"aborting episode to avoid silent dataset corruption. "
                                f"Last error: {e!r}"
                            ) from e

                step_count += 1

                # Fail fast: if EVERY step of the opening probe window drove zero
                # actuators, the policy's output keys cannot match this robot, so
                # the rollout is structurally dead -- raise now instead of running
                # the remaining steps (and inference / recording I/O). Once any
                # step resolves a key, _total_failure_steps < step_count forever
                # and this never fires.
                if step_count >= _FAIL_FAST_PROBE_STEPS and _total_failure_steps == step_count:
                    try:
                        _valid = self.sim.robot_action_keys(robot_name)
                    except Exception:  # noqa: BLE001
                        _valid = _robot_actuators
                    raise RuntimeError(
                        f"All of the first {step_count} action steps had 100% "
                        f"unresolved keys on '{robot_name}' -- the robot has not "
                        f"moved. Unresolved keys: {_last_unresolved}. Valid "
                        f"actuator/joint names: {_valid}. The policy is almost "
                        f"certainly running the wrong embodiment; inspect the "
                        f"expected keys via sim.get_features(robot_name="
                        f"'{robot_name}')."
                    )

                if vwriter is not None:
                    vwriter.capture(step_count)

                if not fast_mode:
                    time.sleep(action_sleep)

            def _stop_when_fired() -> bool:
                """Evaluate the caller's ``stop_when`` clause against the live sim.

                Called after every applied action on BOTH the synchronous and
                async-RTC paths, so the early-return latency bound is ONE
                control step regardless of chunk length: the check fires
                within the current chunk-slice and the remaining actions of
                the chunk are dropped. Call sites guard on ``stop_when is not
                None`` so the no-clause hot path pays no per-step call. A
                raising clause is fatal - the caller asked for early-return
                semantics the runner can no longer honor, and silently
                running to the step budget would misreport the rollout - so
                it surfaces as ``status="error"`` via the outer handler
                rather than being warn-and-continued.
                """
                nonlocal stop_predicate_fired
                assert stop_when is not None  # call sites hoist the None guard
                try:
                    fired = bool(stop_when(self.sim))
                except Exception as e:
                    raise RuntimeError(
                        f"stop_when predicate raised at step {step_count}: {e!r}. The early-return "
                        "condition cannot be evaluated, so the rollout is aborted rather than "
                        "silently running to its step budget."
                    ) from e
                if fired:
                    stop_predicate_fired = True
                    logger.info("stop_when fired at step %d; ending rollout early", step_count)
                return fired

            def _query_chunk(observation: dict[str, Any], observed_delay: int = 0) -> list[dict[str, Any]]:
                # Resolve ONE action chunk. Never truncate below the policy's
                # own intended chunk size (resolve_chunk_length): clamping a
                # chunk trained for N-step open-loop replay forces an
                # out-of-distribution re-query. Tell the policy how many
                # control steps elapse between this observation and the first
                # application of the chunk (an EXACT integer, not a wall-clock
                # estimate) so RTC providers slice the seam correctly; the set
                # and the get_actions call happen on the SAME thread and at
                # most one inference is in flight, so this never races.
                policy.set_rtc_observed_delay(observed_delay)
                _t_infer = time.perf_counter()
                coro_or_result = policy.get_actions(observation, instruction, **_policy_kwargs)
                actions = _resolve_coroutine(coro_or_result)
                # Record inference wall-time (ms) for both the sync and async
                # paths. Under async this runs on the prefetch worker; list
                # append is atomic under the GIL so the read after
                # shutdown(wait=True) sees every entry.
                inference_ms.append((time.perf_counter() - _t_infer) * 1000.0)
                _chunk = resolve_chunk_length(policy, action_horizon)
                return list(actions[:_chunk])

            if async_rtc:
                # Async chunk pipeline: overlap inference for chunk N+1 with
                # the EXECUTION of chunk N (see the async_rtc doc above). The
                # policy is invoked from AT MOST one thread at a time (a new
                # prefetch is only submitted after the previous one has been
                # consumed), and the sim is only ever touched from THIS thread,
                # so there is no MuJoCo data race.
                from concurrent.futures import Future, ThreadPoolExecutor
                from concurrent.futures import TimeoutError as FuturesTimeout

                def _swap_in(fut: Future[list[dict[str, Any]]]) -> list[dict[str, Any]]:
                    # Block on the prefetched chunk at the seam. A HIT means
                    # inference already finished; a BLOCK means inference ran
                    # slower than the chunk's execution (the actionable "tune
                    # prefetch_trigger / shorten the chunk" signal, so log it).
                    # A hard timeout turns a stuck model into a structured
                    # error instead of an unbounded sim hang.
                    nonlocal rtc_prefetch_hits, rtc_prefetch_blocks
                    if fut.done():
                        rtc_prefetch_hits += 1
                    else:
                        rtc_prefetch_blocks += 1
                        logger.warning(
                            "async-RTC seam starvation: prefetched chunk was not ready at the "
                            "swap point (inference slower than chunk execution). Blocking on it; "
                            "consider a shorter chunk or an earlier prefetch_trigger."
                        )
                    try:
                        return fut.result(timeout=rtc_inference_timeout_s)
                    except FuturesTimeout as e:
                        raise RuntimeError(
                            f"async-RTC prefetch exceeded rtc_inference_timeout_s="
                            f"{rtc_inference_timeout_s}s; policy inference is stuck. Raise the "
                            f"timeout or check the policy/server."
                        ) from e

                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-prefetch")
                try:
                    cur_obs = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
                    cur_chunk = _query_chunk(cur_obs)
                    rtc_chunks_acquired += 1
                    if not cur_chunk:
                        raise RuntimeError("policy returned an empty action chunk; cannot run rollout")
                    idx = 0
                    prefetch_trigger = max(1, len(cur_chunk) // 2)
                    prefetch: Future[list[dict[str, Any]]] | None = None
                    prefetch_obs: dict[str, Any] | None = None

                    while step_count < total_steps:
                        if idx >= len(cur_chunk):
                            # Current chunk drained -> swap in the next chunk.
                            if prefetch is not None:
                                cur_chunk = _swap_in(prefetch)
                                if prefetch_obs is not None:
                                    cur_obs = prefetch_obs
                                prefetch = None
                                prefetch_obs = None
                            else:
                                # Chunk was too short to trigger a prefetch;
                                # fall back to a synchronous re-query.
                                cur_obs = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
                                cur_chunk = _query_chunk(cur_obs)
                            rtc_chunks_acquired += 1
                            if not cur_chunk:
                                # Drop-and-requery: a prefetched chunk arriving
                                # empty (a transient policy hiccup) degrades to
                                # ONE synchronous re-query before we give up,
                                # rather than killing an otherwise-healthy
                                # rollout on a single empty result.
                                logger.warning(
                                    "async-RTC chunk arrived empty; falling back to one "
                                    "synchronous re-query before erroring."
                                )
                                cur_obs = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
                                cur_chunk = _query_chunk(cur_obs)
                                rtc_chunks_acquired += 1
                                if not cur_chunk:
                                    raise RuntimeError(
                                        "policy returned an empty action chunk twice (prefetch + "
                                        "synchronous re-query); cannot continue rollout"
                                    )
                            idx = 0
                            prefetch_trigger = max(1, len(cur_chunk) // 2)
                            continue

                        # Fire the next inference once we are ~50% through the
                        # current chunk, on a fresh mid-chunk observation.
                        if prefetch is None and idx >= prefetch_trigger:
                            prefetch_obs = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
                            # The prefetched chunk first applies after the
                            # remaining steps of the current chunk drain - a
                            # known integer, independent of how long inference
                            # actually takes in wall-clock time (a slow inference
                            # just stalls the loop; the robot does not advance
                            # past the chunk end while waiting).
                            observed_delay = max(0, len(cur_chunk) - prefetch_trigger)
                            prefetch = executor.submit(_query_chunk, prefetch_obs, observed_delay)

                        # When recording, the chunk observation (the initial
                        # query obs, or a horizon-shifted prefetch obs after a
                        # swap) is stale for the step being applied; refresh it
                        # so the recorded frame is time-aligned (see the
                        # _record_per_step_obs note above). Inference is
                        # unaffected - it already consumed cur_obs to produce
                        # this chunk.
                        if _record_per_step_obs:
                            step_obs = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
                        else:
                            step_obs = cur_obs
                        _apply(step_obs, cur_chunk[idx])
                        idx += 1
                        # Semantic early return: checked after EVERY applied
                        # action, so the stop lands within one control step of
                        # the world reaching the condition - the rest of the
                        # in-flight chunk (and any prefetched chunk) is
                        # dropped; the executor shutdown below joins the
                        # in-flight prefetch worker. The None guard is hoisted
                        # so the no-clause hot path pays no per-step call.
                        if stop_when is not None and _stop_when_fired():
                            break
                finally:
                    # Wait for any in-flight inference so no background thread
                    # touches the policy/sim after run() returns (the caller may
                    # immediately reset() or destroy() the world).
                    executor.shutdown(wait=True)
            else:
                while step_count < total_steps:
                    observation = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
                    chunk = _query_chunk(observation)
                    # An empty chunk advances nothing, so without this guard the
                    # loop would re-query the policy forever with no progress.
                    if not chunk:
                        raise RuntimeError("policy returned an empty action chunk; cannot run rollout")
                    for chunk_idx, action_dict in enumerate(chunk):
                        if step_count >= total_steps:
                            break
                        # The chunk-start observation is the correct pre-action
                        # state for the first action only. When recording,
                        # refresh it before each SUBSEQUENT action so the
                        # recorded frame is time-aligned (see the
                        # _record_per_step_obs note above). chunk_idx == 0 reuses
                        # the freshly-queried observation (no re-render, sim has
                        # not stepped yet). Inference is unaffected.
                        if _record_per_step_obs and chunk_idx > 0:
                            step_obs = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
                        else:
                            step_obs = observation
                        _apply(step_obs, action_dict)
                        # Semantic early return: checked after EVERY applied
                        # action (same cadence as the benchmark eval loop), so
                        # the remaining actions of the chunk are dropped as
                        # soon as the condition holds. The None guard is
                        # hoisted so the no-clause hot path pays no per-step
                        # call.
                        if stop_when is not None and _stop_when_fired():
                            break
                    if stop_predicate_fired:
                        break

        except CooperativeStop:
            stopped_early = True
            stopped_reason = "cancelled"
        except Exception as e:
            if vwriter is not None:
                vwriter.close()
            logger.exception("PolicyRunner.run failed")
            return {
                "status": "error",
                "content": [
                    {"text": f"Policy failed: {e}"},
                    {"json": {**_rtc_telemetry(), "stopped_reason": "error", "steps_used": step_count}},
                ],
            }

        # Either finished all steps, hit the stop_when condition, or was
        # cooperatively stopped.
        if stop_predicate_fired:
            stopped_early = True
            stopped_reason = "predicate"
        elapsed = time.time() - start_time
        sim_time = self._maybe_sim_time()
        if not stopped_early:
            prefix = "Policy complete"
        elif stopped_reason == "predicate":
            prefix = "Policy stopped early (stop_when condition met)"
        else:
            prefix = "Policy stopped"
        text = (
            f"{prefix} on '{robot_name}'\n{type(policy).__name__} | {instruction}\n{elapsed:.1f}s | {step_count} steps"
        )
        if sim_time is not None:
            text += f" | sim_t={sim_time:.3f}s"
        if vwriter is not None:
            assert video is not None
            video_path = vwriter.path
            frame_count = vwriter.frame_count
            vwriter.close()
            if frame_count > 0 and os.path.exists(video_path):
                file_kb = os.path.getsize(video_path) / 1024
                text += (
                    f"\nVideo: {video_path}\n"
                    f"{frame_count} frames, {video.fps}fps, "
                    f"{video.width}x{video.height} | {file_kb:.0f} KB"
                )
            else:
                # Log a loud warning so the user isn't blindsided by a silent
                # 0-byte MP4. We already pre-validate the camera name up-front,
                # so hitting this branch means frames failed DURING the rollout
                # (e.g. the camera was removed mid-episode).
                logger.warning(
                    "video recording requested but wrote 0 frames to %s - "
                    "MP4 file will be empty or absent. Check that the camera "
                    "remained valid throughout the rollout.",
                    video_path,
                )
                text += f"\nVideo requested but 0 frames captured ({video_path})"
        # Agent-consumable structured payload mirroring eval_policy()'s
        # ``{"json": {...}}`` block; the text block stays for humans. Keys are
        # stable: callers can rely on them.
        payload: dict[str, Any] = {
            "robot_name": robot_name,
            "policy": type(policy).__name__,
            "instruction": instruction,
            "n_steps": step_count,
            # Alias of n_steps under the retry-loop name: the control steps
            # actually executed before the rollout ended. Paired with
            # stopped_reason it makes "the predicate fired after 37 of 200
            # steps" queryable without arithmetic on the caller side.
            "steps_used": step_count,
            "elapsed_s": round(elapsed, 3),
            "stopped_early": stopped_early,
            "stopped_reason": stopped_reason,
            "action_errors": _action_errors,
            "video_path": None,
            "video_frames": 0,
            # Load telemetry: policy_load_cache_hit=False on episode 2+ of a
            # loop is a smell that the caller rebuilt the policy instead of
            # reusing policy_object=. Defaults cover policies without it.
            "policy_load_time_s": round(float(getattr(policy, "load_time_s", 0.0)), 3),
            "policy_load_cache_hit": bool(getattr(policy, "load_cache_hit", False)),
            # Routing-degradation telemetry: True means a heuristic remap
            # (positional camera routing, or observation.state composed from
            # generic/missing state keys) silently degraded the run while
            # status stays "success" - so the signal must be machine-readable.
            "positional_fallback_used": bool(getattr(policy, "positional_fallback_used", False)),
            "generic_state_keys_used": bool(getattr(policy, "generic_state_keys_used", False)),
            "missing_state_keys_used": bool(getattr(policy, "missing_state_keys_used", False)),
            # Process RSS (MB) at result time: confirms a heavy model is resident
            # and, across a loop, that it stays resident instead of oscillating
            # as it would on a per-episode reload. None when unmeasurable.
            "policy_resident_rss_mb": process_rss_mb(),
        }
        if sim_time is not None:
            payload["sim_time_s"] = round(sim_time, 3)
        if vwriter is not None and video is not None:
            _vp = vwriter.path
            wrote_video = vwriter.frame_count > 0 and os.path.exists(_vp)
            payload["video_path"] = _vp if wrote_video else None
            payload["video_frames"] = vwriter.frame_count
        payload.update(_rtc_telemetry())

        # Per-actuator resolution stats: fraction of steps each actuator was
        # actually driven. A joint stuck at 0.0 means no policy key ever
        # resolved to it (wrong name / missing DOF).
        if step_count > 0 and _robot_actuators:
            action_resolution_rate = {
                name: round(_actuator_resolved.get(name, 0) / step_count, 4) for name in _robot_actuators
            }
            # Aggregate: mean fraction of the robot's DOF NOT driven. Distinct
            # from action_errors (a step-level status count): driving 1 of 6
            # joints every step is status=success, action_errors=0, yet a
            # partial_action_failure_rate of ~0.83.
            _driven = sum(_actuator_resolved.get(n, 0) for n in _robot_actuators)
            partial_action_failure_rate = round(1.0 - _driven / (len(_robot_actuators) * step_count), 4)
            payload["action_resolution_rate"] = action_resolution_rate
            payload["partial_action_failure_rate"] = partial_action_failure_rate
            # Promote a high-but-not-total under-actuation to the human text so a
            # silently-crippled rollout (success_rate 0 because only 1 joint
            # moved) is not invisible. A total failure already errors above.
            if 0.5 < partial_action_failure_rate < 1.0:
                text += (
                    f"\n\nPartial action coverage: {partial_action_failure_rate:.0%} of this robot's "
                    f"actuators were never driven. Per-actuator resolution: {action_resolution_rate}."
                )
        else:
            payload["action_resolution_rate"] = {}
            payload["partial_action_failure_rate"] = 0.0

        # If EVERY step was a TOTAL failure (the policy emitted keys but none
        # resolved to an actuator), the robot never moved -- report this as an
        # error rather than a false success. This mirrors the fail-fast probe
        # and must key off ``_total_failure_steps``, NOT ``_action_errors``:
        # ``_action_errors`` also counts PARTIAL steps (some keys resolve, the
        # robot moves), so a policy that drives valid keys plus one extra
        # unresolved key every step (e.g. a 7-DOF-trained policy on a 6-DOF arm)
        # would otherwise be misreported as "the robot did not move". A partial
        # rollout is operational -- surfaced via partial_action_failure_rate.
        if _total_failure_steps >= step_count and step_count > 0:
            text += (
                f"\n\nALL {step_count} action steps had 100% unresolved keys "
                f"-- the robot did not move. Check that the policy's output keys "
                f"match the robot's actuator names."
            )
            # An error result always reports stopped_reason="error": the
            # rollout may have run its full budget, but the outcome is not a
            # retryable "budget" completion.
            payload["stopped_reason"] = "error"
            return {"status": "error", "content": [{"text": text}, {"json": payload}]}
        if _action_errors > 0:
            text += f"\n\n{_action_errors}/{step_count} action steps had unresolved keys."
        return {"status": "success", "content": [{"text": text}, {"json": payload}]}

    # replay(): replay a LeRobotDataset episode

    def replay(
        self,
        repo_id: str,
        robot_name: str | None = None,
        *,
        episode: int = 0,
        root: str | None = None,
        speed: float = 1.0,
        action_key_map: list[str] | None = None,
    ) -> dict[str, Any]:
        """Replay a recorded LeRobotDataset episode through ``send_action``.

        Each recorded frame is one control step at the dataset's fps, so replay
        advances physics for a full control period per frame (the same
        integration the recording used) - a position-servo robot can then track
        the recorded targets. ``speed`` scales only the wall-clock playback
        rate, never the physics per frame.

        Args:
            repo_id: HuggingFace dataset id (e.g. ``lerobot/pusht``).
            robot_name: Target robot. Defaults to the first robot in the sim;
                an explicit name not present in the sim is rejected with a
                structured error.
            episode: Episode index in the dataset (non-negative).
            root: Optional local dataset root override.
            speed: Playback speed multiplier (1.0 = real time). Must be a
                positive, finite real scalar (NumPy scalars accepted);
                anything else is rejected with a structured error.
            action_key_map: Optional list of action keys, one per recorded
                action-vector index; required when the dataset ordering
                differs from ``robot_action_keys(robot_name)``. ``None`` maps
                positionally onto ``robot_action_keys`` - the robot's
                *actuator* keys, the ordering the LeRobotDataset recorder
                writes (a robot's actuators are not always its joints). Must
                be a non-empty list/tuple of unique strings whose length
                equals the recorded action vector's width; a mismatch is
                rejected rather than positionally truncated.

        Returns:
            Standard status dict with per-frame stats. Replay aborts with an
            ``"error"`` status when a recorded frame cannot actually be applied
            (unresolvable action keys, or a width mismatch), reporting how many
            frames were applied before the abort - a success means every frame
            reached the actuators.
        """
        # speed divides into frame_interval and flows into time.sleep on the
        # real-time path: 0 raised a bare ZeroDivisionError and a negative
        # value played forward at full speed while "succeeding". Reject
        # non-positive / non-finite / non-numeric speed up front, before the
        # (potentially multi-minute) dataset download. Any real scalar is
        # accepted (NumPy scalars included); bool is rejected explicitly
        # (True would act as a silent 1.0x) and nan via isfinite (nan is
        # never <= 0).
        if (
            isinstance(speed, bool)
            or not isinstance(speed, numbers.Real)
            or not math.isfinite(float(speed))
            or float(speed) <= 0
        ):
            return {
                "status": "error",
                "content": [{"text": f"replay: speed must be a positive number (got {speed!r})."}],
            }
        # Coerce to a plain Python float: a NumPy scalar raises in time.sleep
        # and is not natively JSON-serialisable in the returned "speed" field.
        speed = float(speed)

        # Reject a malformed action_key_map before the (potentially
        # multi-minute) dataset download; see _validate_action_key_map.
        key_map_error = _validate_action_key_map(action_key_map)
        if key_map_error is not None:
            return key_map_error

        try:
            from strands_robots.dataset_recorder import load_lerobot_episode
        except ImportError:
            return {"status": "error", "content": [{"text": "lerobot not installed"}]}

        try:
            resolved_robot = robot_name or self._require_default_robot()
        except ValueError as e:
            return {"status": "error", "content": [{"text": f"{e}"}]}

        # Reject an unknown robot up front; otherwise replay silently no-ops
        # onto a phantom robot.
        robots = self.sim.list_robots()
        if resolved_robot not in robots:
            return {
                "status": "error",
                "content": [{"text": f"Robot '{resolved_robot}' not found in sim. Available robots: {robots}"}],
            }

        try:
            ds, episode_start, episode_length = load_lerobot_episode(repo_id, episode, root)
        except Exception as e:  # noqa: BLE001 - library errors are opaque
            return {"status": "error", "content": [{"text": f"{e}"}]}

        # The recorded ``action`` column is written in the robot's *actuator*
        # order (robot_action_keys), which diverges from robot_joint_names for
        # passive/mimic joints and tendon-driven grippers. Bind to the same
        # actuator keys the recorder used so record -> replay round-trips.
        action_keys = list(action_key_map) if action_key_map else self.sim.robot_action_keys(resolved_robot)

        dataset_fps = getattr(ds, "fps", 30)
        frame_interval = 1.0 / (dataset_fps * speed)
        # Step a FULL control period per recorded frame (the recorded control
        # frequency IS the dataset fps), matching run()/evaluate(); a single
        # physics dt per frame under-integrates and silently attenuates the
        # trajectory. ``speed`` scales only frame_interval, never the physics,
        # so it is deliberately excluded here.
        n_substeps = self._control_substeps(dataset_fps)
        frames_applied = 0
        start_time = time.time()

        # Read from ``ds.hf_dataset`` (columns only, no video decode) when
        # present: a full LeRobotDataset __getitem__ decodes every camera video
        # per frame - wasted work here, and it raises when a video decoder is
        # missing even though the actions are perfectly readable.
        frame_source: Any = ds
        hf_dataset = getattr(ds, "hf_dataset", None)
        if hf_dataset is not None:
            frame_source = hf_dataset

        for frame_idx in range(episode_length):
            step_start = time.time()
            try:
                frame = frame_source[episode_start + frame_idx]
            except Exception as e:  # noqa: BLE001 - decoder/library errors are opaque
                return {
                    "status": "error",
                    "content": [{"text": (f"Failed to read frame {episode_start + frame_idx} from '{repo_id}': {e}")}],
                }

            action_vals = frame.get("action") if isinstance(frame, dict) else None
            if action_vals is None:
                # No action at this index - advance physics one full control
                # period so the frame still occupies its recorded time slice.
                self.sim.step(n_steps=n_substeps)
                frames_applied += 1
            else:
                if hasattr(action_vals, "numpy"):
                    action_vals = action_vals.numpy()
                if hasattr(action_vals, "tolist"):
                    action_vals = action_vals.tolist()

                # A recorded vector whose width differs from the action-key map
                # cannot be replayed faithfully (surplus values have no key, or
                # surplus keys never receive a value); reject with the
                # recorded-vs-expected widths rather than truncating.
                if len(action_vals) != len(action_keys):
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"Replay aborted at frame {frame_idx}: recorded action vector has "
                                    f"{len(action_vals)} values but {len(action_keys)} action keys are "
                                    f"mapped ({action_keys}). Applied {frames_applied}/{episode_length} "
                                    "frames. Pass an action_key_map with one key per recorded action "
                                    f"value, or replay onto a robot whose actuators match the recording."
                                )
                            },
                            {
                                "json": {
                                    "episode": episode,
                                    "robot_name": resolved_robot,
                                    "frame": frame_idx,
                                    "recorded_action_width": len(action_vals),
                                    "action_keys": action_keys,
                                    "frames_applied": frames_applied,
                                    "total_frames": episode_length,
                                }
                            },
                        ],
                    }

                action_dict: dict[str, Any] = {action_keys[i]: float(val) for i, val in enumerate(action_vals)}

                # Abort on the first unapplied frame: ignoring send_action's
                # error status would let a typo'd action_key_map drop every
                # value at the actuator boundary while replay still reported
                # success - a replay that is not happening.
                send_result = self.sim.send_action(action_dict, robot_name=resolved_robot, n_substeps=n_substeps)
                if isinstance(send_result, dict) and send_result.get("status") == "error":
                    detail = next(
                        (
                            str(block["text"])
                            for block in send_result.get("content", []) or []
                            if isinstance(block, dict) and "text" in block
                        ),
                        "",
                    )
                    payload: dict[str, Any] = {
                        "episode": episode,
                        "robot_name": resolved_robot,
                        "frame": frame_idx,
                        "action_keys": action_keys,
                        "frames_applied": frames_applied,
                        "total_frames": episode_length,
                    }
                    send_json = _extract_result_json(send_result)
                    if send_json is not None:
                        payload.update(send_json)
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"Replay aborted at frame {frame_idx}: the recorded action could not be "
                                    f"applied to '{resolved_robot}'. Applied {frames_applied}/{episode_length} "
                                    f"frames. {detail}"
                                )
                            },
                            {"json": payload},
                        ],
                    }
                frames_applied += 1

            sleep_time = frame_interval - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        duration = time.time() - start_time
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Replayed episode {episode} from {repo_id} on '{resolved_robot}'\n"
                        f"Frames: {frames_applied}/{episode_length} | "
                        f"Duration: {duration:.1f}s | Speed: {speed}x"
                    )
                },
                {
                    "json": {
                        "episode": episode,
                        "robot_name": resolved_robot,
                        "frames_applied": frames_applied,
                        "total_frames": episode_length,
                        "duration_s": round(duration, 2),
                        "speed": speed,
                    }
                },
            ],
        }

    # evaluate(): multi-episode success metrics

    def evaluate(
        self,
        robot_name: str,
        policy: Policy,
        *,
        instruction: str = "",
        n_episodes: int = 10,
        max_steps: int = 300,
        success_fn: SuccessFn | str | None = None,
        spec: BenchmarkProtocol | None = None,
        seed: int | None = None,
        action_horizon: int = 8,
        on_frame: OnFrame | None = None,
        control_frequency: float = 50.0,
        control_substeps: int | None = None,
        async_rtc: bool = False,
        rtc_inference_timeout_s: float | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        video: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate ``policy`` for ``n_episodes`` episodes.

        Two evaluation paths:

        * **``spec=``** (preferred): drive a full :class:`BenchmarkProtocol`.
          Per-episode seeded RNG, ``on_episode_start`` / ``on_step`` /
          ``is_success`` / ``is_failure`` hooks, cumulative dense reward,
          robot-compatibility validation. ``max_steps`` from the spec wins.
        * **``success_fn=``**: legacy sparse-success path. Equivalent to a
          ``BenchmarkProtocol`` whose ``on_step`` always returns
          ``StepInfo(reward=0.0, done=False)``.

        Passing both ``spec`` and ``success_fn`` is an error - benchmarks
        define their own success predicate.

        Args:
            robot_name: Robot to evaluate.
            policy: Already-constructed ``Policy`` instance.
            instruction: Instruction forwarded to the policy.
            n_episodes: Number of reset -> rollout episodes.
            max_steps: Cap per episode. Ignored when ``spec`` is provided
                (``spec.max_steps`` wins).
            success_fn: Legacy success predicate (see above).
            spec: :class:`BenchmarkProtocol` to drive the eval. When
                provided, overrides the ``success_fn`` path.
            seed: Master RNG seed. Each episode derives a child RNG from it,
                so evaluations are reproducible within a process. Only used
                when ``spec`` is provided.
            on_frame: Optional ``(step, observation, action) -> None`` hook
                fired per applied control step on the eval thread, after
                ``sim.send_action``, on BOTH eval paths; ``step`` is a
                monotonic index continuing across episode boundaries. A
                non-``CooperativeStop`` hook exception is logged at WARN and
                never aborts the eval; raising :class:`CooperativeStop` stops
                gracefully after the episodes completed so far
                (``stopped_early=True``), matching :meth:`run`. Use this for
                synchronous recording when the eval runs on a thread distinct
                from the script main (e.g. agent tool dispatch under asyncio).
            action_horizon: Max actions consumed per policy call (see
                :meth:`run`).
            control_frequency: Target control rate in Hz.
            control_substeps: Explicit physics substeps per action, or ``None``
                to derive.
            async_rtc: Opt-in inference/execution overlap on the legacy
                ``success_fn`` path, mirroring :meth:`run`. ``False`` (default)
                pauses the world during inference, so the success-rate is
                bit-stable. ``True`` evaluates a chunk-emitting policy under
                the realistic control latency it faces in deployment - the
                mid-chunk (staler) seam observation can shift the measured
                success-rate, which is the point. ``True`` is rejected on the
                ``spec=`` path, which stays synchronous for bit-stable
                reproducibility; use :meth:`run` for latency masking there.
            rtc_inference_timeout_s: Hard per-chunk timeout (seconds) for the
                async prefetch; on expiry the eval fails with a structured
                error instead of hanging. ``None`` waits indefinitely.
            policy_kwargs: Per-call goal payload forwarded verbatim to every
                ``policy.get_actions`` call on both eval paths -
                goal-conditioned providers (``target_pose`` /
                ``target_joints`` / ``target_velocity`` / ``world_update``)
                need this to be evaluated against a goal at all.
            video: Optional per-episode MP4 recording config (same dict schema
                as :meth:`run`). One file per episode with ``_ep{i}`` inserted
                into the filename; written paths are returned in the result
                json ``video_paths``. Recorded on BOTH eval routes, captured
                synchronously on the eval thread at the ``on_frame`` point so
                recording never perturbs the bit-stable spec-path rollout.

        Returns:
            Standard status dict. The JSON payload carries the RTC telemetry
            block (see :meth:`run`) plus, on the ``spec`` path,
            ``cumulative_reward`` / ``avg_reward`` per episode and aggregate.

            Every payload carries ``success_measured`` (bool): ``False`` means
            no success criterion was given, so the reported ``success_rate``
            is a hard ``0.0`` that measures nothing - check this flag before
            trusting ``success_rate``.

            The payload also carries ``episodes_completed`` and
            ``stopped_early``; after a :class:`CooperativeStop` the aggregate
            metrics are computed over ``episodes_completed`` (which may be
            less than the requested ``n_episodes``).
        """
        if spec is not None and success_fn is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "evaluate() accepts either 'spec' or 'success_fn', not both. "
                            "'spec' defines its own success predicate."
                        )
                    }
                ],
            }

        # Per-call goal payload forwarded verbatim to every get_actions() call
        # on both eval paths; an empty dict is the historical no-kwargs path.
        _policy_kwargs = policy_kwargs or {}

        if async_rtc and spec is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "async_rtc is only supported on the success_fn eval path. "
                            "The spec/benchmark path stays synchronous for bit-stable "
                            "reproducibility; use run_policy(async_rtc=...) for "
                            "benchmark-style latency masking."
                        )
                    }
                ],
            }

        if spec is not None:
            return self._evaluate_with_spec(
                robot_name,
                policy,
                spec,
                instruction=instruction,
                n_episodes=n_episodes,
                seed=seed,
                action_horizon=action_horizon,
                on_frame=on_frame,
                control_frequency=control_frequency,
                control_substeps=control_substeps,
                policy_kwargs=_policy_kwargs,
                video=video,
            )

        try:
            resolved_check = self._resolve_success_fn(success_fn)
        except ValueError as e:
            return {"status": "error", "content": [{"text": f"{e}"}]}

        # With no success criterion the loop never sets success=True, so
        # success_rate is a hard 0.0 indistinguishable from genuine failure.
        # Warn loudly and flag the payload so 0.0 is not read as a measurement.
        success_measured = resolved_check is not None
        if not success_measured:
            logger.warning(
                "evaluate()/eval_policy called without a success criterion "
                "(success_fn=None and no spec): success_rate will be 0.0 for "
                "every episode regardless of what the policy does and does "
                "NOT measure task success. Pass success_fn (e.g. 'contact' "
                "or a callable) or a benchmark spec to measure success; the "
                "returned json flags this as success_measured=false."
            )

        # T26: skip camera rendering when the policy does not need images.
        _skip_images = not getattr(policy, "requires_images", True)
        # Step physics for the full control period per action, same derivation
        # as run(). The default n_substeps=1 made eval rollouts under-step.
        n_substeps = self._control_substeps(control_frequency, control_substeps)
        # The achieved rate, not the requested one (see run()).
        policy.set_control_frequency(self._achieved_control_frequency(control_frequency, n_substeps))

        # RTC telemetry, reported in the result json so inference cost (and,
        # under async_rtc, latency masking) is provable without grepping logs.
        # inference_ms collects every get_actions wall-time on both paths; the
        # prefetch hit/block counters are async-only (0 on the synchronous path).
        inference_ms: list[float] = []
        rtc_chunks_acquired = 0
        rtc_prefetch_hits = 0
        rtc_prefetch_blocks = 0

        def _observation_fn() -> dict[str, Any]:
            return self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)

        # The success predicate gets its OWN image-free fetch. Using
        # _observation_fn re-rendered every camera a second time per control
        # step for any image-consuming policy (i.e. every VLA), purely to hand
        # an observation to a predicate that does not read pixels: the built-in
        # check is ``def _contact_check(_obs)`` and ignores its argument, and
        # everything in predicates.py takes ``sim`` and reads sim state directly.
        # Measured over 20 eval steps with one 224x224 camera: 40
        # image-rendering observation calls where 20 suffice, at ~7.9 ms each.
        # ``requires_images`` exists precisely to avoid this cost.
        #
        # A caller-supplied predicate that genuinely needs pixels opts in by
        # setting ``requires_images = True`` on itself, so the default stays
        # cheap without silently withholding data from a predicate that wants it.
        _check_needs_images = bool(getattr(resolved_check, "requires_images", False))

        def _success_obs() -> dict[str, Any]:
            return self.sim.get_observation(robot_name=robot_name, skip_images=not _check_needs_images)

        def _query_chunk(observation: dict[str, Any], observed_delay: int = 0) -> list[dict[str, Any]]:
            # Tell RTC policies how many control steps elapse between this
            # observation and the chunk's first application (exact integer,
            # not a wall-clock estimate): 0 on the synchronous path, the
            # still-pending step count under the async pipeline.
            policy.set_rtc_observed_delay(observed_delay)
            _t_infer = time.perf_counter()
            actions = _resolve_coroutine(policy.get_actions(observation, instruction, **_policy_kwargs))
            inference_ms.append((time.perf_counter() - _t_infer) * 1000.0)
            # resolve_chunk_length is the single source of truth for the
            # re-query interval; truncating below it would force an
            # out-of-distribution re-query of chunk-predicting VLAs.
            return list(actions[: resolve_chunk_length(policy, action_horizon)])

        results: list[dict[str, Any]] = []
        # Monotonic global step index handed to ``on_frame``, continuous
        # across episode boundaries (matches the spec path and run()).
        global_step = 0

        # Optional per-episode rollout video, one MP4 per episode via the
        # _ep{i} filename templating; _fire_on_frame appends frames at the fps
        # cadence on the same synchronous eval-thread point as the hook.
        video_paths: list[str] = []
        current_vwriter: _RolloutVideoWriter | None = None

        def _fire_on_frame(obs: dict[str, Any], action: dict[str, Any], ep_step: int) -> None:
            # Fire AFTER ``send_action`` (post-action obs unavailable yet, so
            # pass the pre-action obs the chunk was queried with - matches
            # ``_evaluate_with_spec``). The hook is best-effort telemetry: a
            # failure is logged at WARN and never aborts the eval.
            nonlocal global_step
            if current_vwriter is not None:
                current_vwriter.capture(ep_step)
            if on_frame is not None:
                try:
                    on_frame(global_step, obs, action)
                except CooperativeStop:
                    # Documented graceful early-stop (the same signal run()
                    # honors). Propagate to the episode loop; never swallow
                    # it as a best-effort telemetry failure.
                    raise
                except Exception as e:  # noqa: BLE001 - hook is best-effort telemetry
                    logger.warning("on_frame hook failed at global_step=%d: %s", global_step, e)
            global_step += 1

        stopped_early = False
        try:
            for ep in range(n_episodes):
                self.sim.reset()
                success = False
                steps = 0

                # Per-episode MP4 (foo_ep{i}.mp4). Validation + camera probe happen
                # here; a bad path/camera fails the eval up-front (on ep 0) instead
                # of running N episodes and writing nothing.
                ep_vcfg = self.sim._episode_video_config(video, ep)
                current_vwriter, _video_err = _RolloutVideoWriter.open(self.sim, ep_vcfg, control_frequency)
                if _video_err is not None:
                    return _video_err

                if async_rtc:
                    # Opt-in async overlap (see _ChunkPipeline). The pipeline
                    # only ever calls the policy off-thread; the sim is stepped
                    # solely here. The context manager joins the worker on exit
                    # even when we break mid-chunk on success.
                    pipeline = _ChunkPipeline(
                        _query_chunk,
                        _observation_fn,
                        async_rtc=True,
                        rtc_inference_timeout_s=rtc_inference_timeout_s,
                    )
                    with pipeline as chunks:
                        for _observation, action_dict in chunks:
                            if steps >= max_steps:
                                break
                            self.sim.send_action(action_dict, robot_name=robot_name, n_substeps=n_substeps)
                            _fire_on_frame(_observation, action_dict, steps)
                            steps += 1
                            # Check success against the LIVE post-action observation
                            # (mirrors the synchronous path / _evaluate_with_spec).
                            if resolved_check is not None and resolved_check(_success_obs()):
                                success = True
                                break
                    rtc_chunks_acquired += pipeline.chunks_acquired
                    rtc_prefetch_hits += pipeline.prefetch_hits
                    rtc_prefetch_blocks += pipeline.prefetch_blocks
                else:
                    while steps < max_steps:
                        observation = _observation_fn()
                        chunk = _query_chunk(observation, 0)
                        rtc_chunks_acquired += 1

                        if not chunk:
                            # Policy returned nothing - still advance one physics
                            # step so episodes don't hang on degenerate policies,
                            # then check the post-step observation (same post-action
                            # semantics as the chunk branch below).
                            self.sim.step(n_steps=1)
                            steps += 1
                            if resolved_check is not None and resolved_check(_success_obs()):
                                success = True
                                break
                            continue

                        for action_dict in chunk:
                            if steps >= max_steps:
                                break
                            self.sim.send_action(action_dict, robot_name=robot_name, n_substeps=n_substeps)
                            _fire_on_frame(observation, action_dict, steps)
                            steps += 1
                            # Check success against the LIVE post-action
                            # observation: the stale pre-action obs detects
                            # success one step late and misses a task that
                            # completes on the final step.
                            if resolved_check is not None and resolved_check(_success_obs()):
                                success = True
                                break
                        if success:
                            break

                results.append({"episode": ep, "steps": steps, "success": success})
                # Per-episode recorder boundary (see _finalize_recorder_episode).
                self._finalize_recorder_episode()

                if current_vwriter is not None:
                    current_vwriter.close()
                    if current_vwriter.frame_count > 0 and os.path.exists(current_vwriter.path):
                        video_paths.append(current_vwriter.path)
                    else:
                        logger.warning(
                            "eval_policy episode %d: video requested but wrote 0 frames to %s",
                            ep,
                            current_vwriter.path,
                        )
                    current_vwriter = None

        except CooperativeStop:
            # A user/backend on_frame hook requested a graceful stop (the
            # same signal run() honors). End the evaluation over the episodes
            # completed so far instead of crashing with an uncaught
            # BaseException. Close any in-progress episode video cleanly.
            stopped_early = True
            logger.info(
                "on_frame requested a cooperative stop; ending evaluation after %d completed episode(s)",
                len(results),
            )
            if current_vwriter is not None:
                current_vwriter.close()
                current_vwriter = None
        n_completed = len(results)
        n_success = sum(1 for r in results if r["success"])
        success_rate = n_success / max(n_completed, 1)
        avg_steps = sum(r["steps"] for r in results) / max(n_completed, 1)
        _n_infer = len(inference_ms)
        rtc_telemetry = {
            "rtc_async_enabled": bool(async_rtc),
            "rtc_chunks_acquired": rtc_chunks_acquired,
            "rtc_prefetch_hits": rtc_prefetch_hits,
            "rtc_prefetch_blocks": rtc_prefetch_blocks,
            "rtc_avg_inference_ms": round(sum(inference_ms) / _n_infer, 3) if _n_infer else 0.0,
            "rtc_max_inference_ms": round(max(inference_ms), 3) if _n_infer else 0.0,
        }

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Evaluation: {type(policy).__name__} on '{robot_name}'\n"
                        f"Episodes: {n_completed}"
                        + (f" of {n_episodes} (stopped early)" if stopped_early else "")
                        + f" | Success: {n_success}/{n_completed} ({success_rate:.1%})"
                        + ("" if success_measured else " [no success criterion - not measured]")
                        + "\n"
                        f"Avg steps: {avg_steps:.0f}/{max_steps}"
                    )
                },
                {
                    "json": {
                        "success_rate": round(success_rate, 4),
                        "success_measured": success_measured,
                        "n_episodes": n_episodes,
                        "episodes_completed": n_completed,
                        "stopped_early": stopped_early,
                        "n_success": n_success,
                        "avg_steps": round(avg_steps, 1),
                        "max_steps": max_steps,
                        "policy_load_time_s": round(float(getattr(policy, "load_time_s", 0.0)), 3),
                        "policy_load_cache_hit": bool(getattr(policy, "load_cache_hit", False)),
                        "positional_fallback_used": bool(getattr(policy, "positional_fallback_used", False)),
                        "generic_state_keys_used": bool(getattr(policy, "generic_state_keys_used", False)),
                        "missing_state_keys_used": bool(getattr(policy, "missing_state_keys_used", False)),
                        **rtc_telemetry,
                        "policy_resident_rss_mb": process_rss_mb(),
                        "episodes": results,
                        "video_paths": video_paths,
                    }
                },
            ],
        }

    def _evaluate_with_spec(
        self,
        robot_name: str,
        policy: Policy,
        spec: BenchmarkProtocol,
        *,
        instruction: str,
        n_episodes: int,
        seed: int | None,
        action_horizon: int = 8,
        on_frame: OnFrame | None = None,
        control_frequency: float = 50.0,
        control_substeps: int | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        video: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Drive a :class:`BenchmarkProtocol` for ``n_episodes`` episodes.

        Split out from :meth:`evaluate`; both routes share the same return-dict
        schema, plus the spec route layers on cumulative-reward accounting.

        Robot compatibility is validated before episode 1: a loaded robot whose
        ``data_config`` is not in a non-empty ``spec.supported_robots`` returns
        a structured error with the allowed list.

        ``on_frame`` fires per applied control step on the eval thread, after
        ``sim.send_action``. Use it for synchronous recording/telemetry that
        must read sim state on the eval thread (a daemon-thread recorder races
        ``mjData`` mutations under multi-threaded eval). Failures are logged
        WARNING; the rollout continues. The hook receives a global step counter
        (across episodes).

        ``video`` (optional) records one rollout MP4 per episode (``_ep{i}``
        filename templating), captured synchronously at the ``on_frame`` point -
        render is read-only over ``mjData`` so it does not perturb the
        bit-stable rollout. Written paths are returned in ``video_paths``.
        """
        # Lazy import to avoid circular reference (benchmark module imports
        # `SimEngine` from base which imports this module under TYPE_CHECKING).
        from strands_robots.simulation.benchmark import BenchmarkCompatibilityError

        # T26: skip camera rendering when the policy does not need images.
        _skip_images = not getattr(policy, "requires_images", True)
        # Full control-period substeps per action (see run() / evaluate()).
        n_substeps = self._control_substeps(control_frequency, control_substeps)
        # The achieved rate, not the requested one (see run()).
        policy.set_control_frequency(self._achieved_control_frequency(control_frequency, n_substeps))
        # Seed once before the episode loop so policy stochastic ops are
        # reproducible across re-runs at the same seed; per-episode
        # reproducibility still flows through episode_rng below.
        if seed is not None:
            set_eval_seed(seed)
        master_rng = random.Random(seed)
        spec_name = type(spec).__name__
        max_steps = spec.max_steps

        def _spec_query_chunk(observation: dict[str, Any], observed_delay: int = 0) -> list[dict[str, Any]]:
            """Query the policy, declaring the RTC observed delay first.

            This method never called ``set_rtc_observed_delay``, so
            ``Policy.rtc_observed_delay_steps`` stayed ``None`` for the whole
            eval and ``LerobotLocalPolicy._run_rtc_inference`` took its
            ``_estimate_inference_delay(fps=...)`` wall-clock branch instead -
            which that code's own comment calls "non-reproducible - it warms up
            within an episode and varies run-to-run, so two otherwise-identical
            seeded episodes drift apart at the seam". Measured: 4 inferences on
            this path reported ``[None, None, None, None]`` where legacy
            ``evaluate`` reported ``[0, 0, 0, 0]``.

            That matters most HERE of all places: this is the path the codebase
            advertises as reproducible (``evaluate`` refuses ``async_rtc=True``
            with a spec for "bit-stable reproducibility", and each episode does
            its own ``set_eval_seed``). The world is PAUSED during inference on
            this synchronous path, so the exact answer is the trivially known 0 -
            an estimated 4 steps at 50Hz would feed lerobot's
            ``get_prefix_weights`` a wrong freeze length.

            Chunk-length resolution lives here too so the set-delay / query /
            truncate trio cannot drift apart again, mirroring the sibling helpers
            in ``run()`` and legacy ``evaluate()``.
            """
            policy.set_rtc_observed_delay(observed_delay)
            actions = _resolve_coroutine(
                policy.get_actions(observation, effective_instruction, **(policy_kwargs or {}))
            )
            return list(actions[: resolve_chunk_length(policy, action_horizon)])

        results: list[dict[str, Any]] = []

        # Global step counter passed to ``on_frame``; monotonic across
        # episode boundaries.
        global_step = 0

        # Fall back to ``spec.instruction`` when the user didn't pass one:
        # language-conditioned policies need the task description or they
        # produce off-task actions, and benchmarks ship the per-task language.
        # A non-empty user ``instruction`` still wins.
        spec_instruction = ""
        try:
            spec_instruction = spec.instruction or ""
        except Exception as e:  # noqa: BLE001 - back-compat for specs without the property
            logger.debug("spec.instruction lookup raised %s; defaulting to empty", e)
        effective_instruction = instruction or spec_instruction
        if not effective_instruction:
            logger.warning(
                "evaluate_benchmark: instruction is empty (user passed %r, spec.instruction=%r). "
                "Language-conditioned policies (GR00T, OpenVLA, etc.) will receive an empty "
                "string and may produce off-task actions. Pass instruction=... explicitly or "
                "override BenchmarkProtocol.instruction on your spec.",
                instruction,
                spec_instruction,
            )

        # Optional per-episode rollout video, one MP4 per episode via the
        # _ep{i} filename templating; frames are captured synchronously on the
        # eval thread at the on_frame point (render is read-only over mjData),
        # so recording never perturbs the bit-stable spec-path rollout.
        video_paths: list[str] = []
        current_vwriter: _RolloutVideoWriter | None = None

        stopped_early = False
        try:
            for ep in range(n_episodes):
                self.sim.reset()
                # Per-episode MP4 (foo_ep{i}.mp4). Path/camera validation +
                # probe render happen here; a bad path/camera fails the eval
                # up-front (on ep 0) instead of running N episodes and writing
                # nothing. No-op (returns None) when video is unset.
                ep_vcfg = self.sim._episode_video_config(video, ep)
                current_vwriter, _video_err = _RolloutVideoWriter.open(self.sim, ep_vcfg, control_frequency)
                if _video_err is not None:
                    return _video_err
                # Per-episode seeded RNG - deterministic given the master seed
                # and the episode index.
                episode_seed = master_rng.randint(0, 2**31 - 1)
                episode_rng = random.Random(episode_seed)

                # Re-seed at the start of EACH episode (not just once before
                # the loop) so episode N always starts from the same RNG state
                # regardless of how many draws episodes 0..N-1 consumed -
                # otherwise stochastic policies are not bit-stable across
                # re-runs at the same seed.
                set_eval_seed(episode_seed)

                # For service-mode policies, set_eval_seed only seeds the
                # client process; forward the per-episode seed via
                # policy.reset(seed=...) so server-side RNG state can be
                # re-initialised (default Policy.reset is a no-op).
                try:
                    policy.reset(seed=episode_seed)
                except Exception as e:  # noqa: BLE001 - reset is best-effort
                    logger.warning(
                        "policy.reset(seed=%d) raised %s; continuing without per-episode reset",
                        episode_seed,
                        e,
                    )

                try:
                    spec.on_episode_start(self.sim, episode_rng)
                except BenchmarkCompatibilityError as e:
                    # Surface the structured error with the supported list -
                    # agents can fix this without retrying.
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"Benchmark compatibility error: robot '{e.robot_name}' "
                                    f"has data_config={e.data_config!r}, but benchmark "
                                    f"{spec_name} supports {e.supported}."
                                )
                            }
                        ],
                    }
                except Exception as e:  # noqa: BLE001 - surface as structured error
                    logger.exception("on_episode_start failed")
                    return {
                        "status": "error",
                        "content": [{"text": f"on_episode_start failed in {spec_name}: {e}"}],
                    }

                success = False
                failure = False
                steps = 0
                cumulative_reward = 0.0
                last_info: dict[str, Any] = {}

                # Bound on APPLIED STEPS, not on iterations. `for _ in
                # range(max_steps)` re-queried the policy once per iteration even
                # though the inner loop drains `_chunk` actions and increments
                # `steps` for each - so N steps cost N inferences and every action
                # but the first of each chunk was thrown away. Measured with
                # chunk=4/max_steps=20: 20 inference calls where 5 suffice; on a
                # SmolVLA-realistic chunk=50 at 80ms it was 294 of 300 inferences
                # wasted, 23.5s of a 28.5s episode. It also silently reinstated
                # the closed-loop `action_horizon=1` behaviour that this method's
                # own docstring records as a contributing factor to
                # success_rate=0. The inner loop already breaks on
                # `steps >= max_steps`, so the `while` form terminates identically.
                # Matches the legacy `evaluate()` loop at :1929.
                while steps < max_steps:
                    observation = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
                    # Benchmarks may bridge the sim's observation schema to
                    # what the policy was trained on (default is identity).
                    # Failures surface as structured errors - "policy got the
                    # wrong obs schema" is a common bug source.
                    try:
                        observation = spec.augment_observation(self.sim, observation)
                    except Exception as e:  # noqa: BLE001
                        logger.exception("augment_observation failed in %s", spec_name)
                        return {
                            "status": "error",
                            "content": [{"text": f"augment_observation failed in {spec_name}: {e}"}],
                        }
                    actions = _spec_query_chunk(observation)

                    # Consume up to the resolved chunk length per inference
                    # (open-loop chunk replay matching how chunk-emitting
                    # models are trained; set action_horizon=1 for closed-loop
                    # receding-horizon control). ``on_step`` and the
                    # success/failure checks run after EACH applied action so
                    # per-step rewards / early termination work at any horizon.
                    action_applied: dict[str, Any] = {}
                    stop_episode = False
                    if not actions:
                        # Degenerate policy - advance physics. The step is COUNTED
                        # further down, in the `if not actions:` block that also
                        # runs on_step and the reward bookkeeping, so incrementing
                        # `steps` here as well would double-count it (measured: an
                        # 8-step episode reported 4 steps' worth of reward). That
                        # later increment is what keeps the now-`steps`-bounded
                        # outer loop terminating on an empty-chunk policy.
                        self.sim.step(n_steps=1)
                    else:
                        # Already truncated to resolve_chunk_length by
                        # _spec_query_chunk; re-slicing here would be a second
                        # source of truth for the same bound.
                        for action_in_chunk in actions:
                            if steps >= max_steps:
                                break
                            action_applied = dict(action_in_chunk)
                            self.sim.send_action(action_applied, robot_name=robot_name, n_substeps=n_substeps)
                            # Video frame + on_frame hook fire synchronously on
                            # the eval thread, after send_action and before
                            # on_step's reward bookkeeping (a daemon-thread
                            # recorder would race mjData mutations). Video
                            # capture is independent of the user hook;
                            # ``steps`` is the pre-increment ep-local index.
                            if current_vwriter is not None:
                                current_vwriter.capture(steps)
                            if on_frame is not None:
                                try:
                                    on_frame(global_step, observation, action_applied)
                                except CooperativeStop:
                                    # Documented graceful early-stop; propagate
                                    # to the episode loop instead of swallowing.
                                    raise
                                except Exception as e:  # noqa: BLE001 - hook is best-effort
                                    logger.warning(
                                        "on_frame hook failed at global_step=%d (ep=%d, ep_step=%d): %s",
                                        global_step,
                                        ep,
                                        steps,
                                        e,
                                    )
                            steps += 1
                            global_step += 1
                            try:
                                info = spec.on_step(self.sim, observation, action_applied)
                            except Exception as e:  # noqa: BLE001
                                logger.exception("on_step failed in %s", spec_name)
                                return {
                                    "status": "error",
                                    "content": [{"text": f"on_step failed in {spec_name}: {e}"}],
                                }
                            cumulative_reward += float(info.reward)
                            last_info = dict(info.info) if info.info else {}
                            if info.done:
                                stop_episode = True
                                break
                            if spec.is_failure(self.sim):
                                failure = True
                                stop_episode = True
                                break
                            if spec.is_success(self.sim):
                                success = True
                                stop_episode = True
                                break
                    if stop_episode:
                        break
                    if not actions:
                        # Degenerate-policy branch already advanced steps via
                        # sim.step(n_steps=1); count it like an applied step
                        # so the outer loop terminates.
                        steps += 1
                        global_step += 1
                        try:
                            info = spec.on_step(self.sim, observation, action_applied)
                        except Exception as e:  # noqa: BLE001
                            logger.exception("on_step failed in %s", spec_name)
                            return {
                                "status": "error",
                                "content": [{"text": f"on_step failed in {spec_name}: {e}"}],
                            }
                        cumulative_reward += float(info.reward)
                        last_info = dict(info.info) if info.info else {}
                        if info.done:
                            break
                        if spec.is_failure(self.sim):
                            failure = True
                            break
                        if spec.is_success(self.sim):
                            success = True
                            break

                results.append(
                    {
                        "episode": ep,
                        "steps": steps,
                        "success": success,
                        "failure": failure,
                        "cumulative_reward": round(cumulative_reward, 4),
                        "seed": episode_seed,
                        "info": last_info,
                    }
                )
                # Same per-episode recorder boundary as evaluate().
                self._finalize_recorder_episode()

                if current_vwriter is not None:
                    current_vwriter.close()
                    if current_vwriter.frame_count > 0 and os.path.exists(current_vwriter.path):
                        video_paths.append(current_vwriter.path)
                    else:
                        logger.warning(
                            "evaluate_benchmark episode %d: video requested but wrote 0 frames to %s",
                            ep,
                            current_vwriter.path,
                        )
                    current_vwriter = None

        except CooperativeStop:
            # A user/backend on_frame hook requested a graceful stop (the
            # same signal run() honors). End the benchmark over the episodes
            # completed so far instead of crashing with an uncaught
            # BaseException. Close any in-progress episode video cleanly.
            if current_vwriter is not None:
                current_vwriter.close()
            stopped_early = True
            logger.info(
                "on_frame requested a cooperative stop; ending benchmark after %d completed episode(s)",
                len(results),
            )
        n_completed = len(results)
        n_success = sum(1 for r in results if r["success"])
        n_failure = sum(1 for r in results if r["failure"])
        success_rate = n_success / max(n_completed, 1)
        avg_steps = sum(r["steps"] for r in results) / max(n_completed, 1)
        avg_reward = sum(r["cumulative_reward"] for r in results) / max(n_completed, 1)

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Benchmark: {spec_name} | policy {type(policy).__name__} on '{robot_name}'\n"
                        f"Episodes: {n_completed}"
                        + (f" of {n_episodes} (stopped early)" if stopped_early else "")
                        + f" | Success: {n_success} | Failure: {n_failure} ({success_rate:.1%} success)\n"
                        f"Avg reward: {avg_reward:.2f} | Avg steps: {avg_steps:.0f}/{max_steps}"
                    )
                },
                {
                    "json": {
                        "success_rate": round(success_rate, 4),
                        "success_measured": True,
                        "n_episodes": n_episodes,
                        "episodes_completed": n_completed,
                        "stopped_early": stopped_early,
                        "n_success": n_success,
                        "n_failure": n_failure,
                        "avg_steps": round(avg_steps, 1),
                        "avg_reward": round(avg_reward, 4),
                        "max_steps": max_steps,
                        "seed": seed,
                        "benchmark_class": spec_name,
                        "policy_load_time_s": round(float(getattr(policy, "load_time_s", 0.0)), 3),
                        "policy_load_cache_hit": bool(getattr(policy, "load_cache_hit", False)),
                        "positional_fallback_used": bool(getattr(policy, "positional_fallback_used", False)),
                        "generic_state_keys_used": bool(getattr(policy, "generic_state_keys_used", False)),
                        "missing_state_keys_used": bool(getattr(policy, "missing_state_keys_used", False)),
                        "policy_resident_rss_mb": process_rss_mb(),
                        "episodes": results,
                        "video_paths": video_paths,
                    }
                },
            ],
        }

    # Helpers

    def _maybe_sim_time(self) -> float | None:
        """Best-effort read of sim time (seconds) from any backend that exposes it.

        Tries ``sim._world.sim_time`` first, then a ``sim_time`` key in
        ``sim.get_state()``'s top level or ``json`` block. ``None`` when
        unavailable.
        """
        world = getattr(self.sim, "_world", None)
        if world is not None:
            t = getattr(world, "sim_time", None)
            if isinstance(t, (int, float)):
                return float(t)

        get_state = getattr(self.sim, "get_state", None)
        if get_state is None:
            return None
        try:
            state = get_state()
        except Exception:
            return None
        if isinstance(state, dict):
            if "sim_time" in state:
                return float(state["sim_time"])
            for blk in state.get("content", []):
                if isinstance(blk, dict) and isinstance(blk.get("json"), dict):
                    t = blk["json"].get("sim_time")
                    if isinstance(t, (int, float)):
                        return float(t)
        return None

    def _require_default_robot(self) -> str:
        robots = self.sim.list_robots()
        if not robots:
            raise ValueError("No robots in sim. Add one first.")
        return robots[0]

    def _resolve_success_fn(self, success_fn: SuccessFn | str | None) -> SuccessFn | None:
        if success_fn is None:
            return None
        if callable(success_fn):
            return success_fn
        if success_fn == "contact":
            # Delegate to the predicate DSL's implementation instead of keeping a
            # second copy. The copy that lived here read ``n_contacts`` /
            # ``contacts`` off the TOP LEVEL of the get_contacts() return, but
            # every backend returns the standard agent-tool envelope
            # ``{"status": ..., "content": [{"text": ...}, {"json": {...}}]}`` -
            # only ``status`` and ``content`` exist at the top level, so both
            # lookups were unconditionally None/0 and the predicate returned
            # False no matter what the robot touched. ``eval_policy`` then
            # reported success_rate=0.0 with success_measured=True, i.e. it
            # presented "never succeeded" as a real measurement.
            #
            # ``predicates._contact_any`` unwraps the envelope via
            # ``_extract_json`` and additionally filters to load-bearing contacts,
            # and its own docstring already claims to match this path - so
            # delegating makes that claim true and removes the divergence.
            from strands_robots.simulation.predicates import _contact_any

            sim = self.sim
            contact_predicate = _contact_any()

            def _contact_check(_obs: dict[str, Any]) -> bool:
                return contact_predicate(sim)

            return _contact_check
        raise ValueError(f"Unknown success_fn string: {success_fn!r}")


__all__ = ["PolicyRunner", "OnFrame", "SuccessFn", "CooperativeStop", "TrajectoryStep", "set_eval_seed"]
