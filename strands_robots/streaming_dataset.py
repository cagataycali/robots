"""Streaming read-back for LeRobotDataset - read frames directly from the Hub.

Primary use: in-process eval / replay / notebooks / agent loops. Streamed
*training* does not need this module: ``python -m lerobot.scripts.lerobot_train
--dataset.streaming=true`` already uses ``StreamingLeRobotDataset`` via
``lerobot.datasets.factory.make_dataset``.

lerobot is never imported at module top-level (numpy/pandas ABI safety on
Jetson; see the :mod:`strands_robots.dataset_recorder` header). The ``[lerobot]``
extra floors lerobot at ``BUCKET_STREAMING_MIN_LEROBOT``, whose constructor
accepts every keyword :meth:`StreamingDatasetReader.open` forwards.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Any

from strands_robots.dataset_recorder import local_dataset_dir
from strands_robots.utils import (
    boolean_flag_error,
    finite_number_error,
    non_negative_count_error,
    positive_count_error,
    refusal_repr,
)

logger = logging.getLogger(__name__)

# The lerobot that first accepts ``repo_type: Literal["dataset", "bucket"]``;
# every lerobot-bearing extra in pyproject floors at or above it.
BUCKET_STREAMING_MIN_LEROBOT = "0.6.1"

# Only a POSITIVE probe is cached: a transient import failure (slow import, a
# test's monkeypatched ``sys.modules``) must not disable streaming for the rest
# of the process.
_HAS_STREAMING_DATASET: list[bool] = []


def has_streaming_dataset() -> bool:
    """Return True if lerobot's ``StreamingLeRobotDataset`` is importable."""
    if _HAS_STREAMING_DATASET:
        return True
    try:
        from lerobot.datasets import StreamingLeRobotDataset  # noqa: F401
    except (ImportError, ValueError, RuntimeError) as exc:
        logger.debug("StreamingLeRobotDataset unavailable: %s", exc)
        return False
    _HAS_STREAMING_DATASET.append(True)
    return True


def _get_streaming_cls() -> Any:
    """Return StreamingLeRobotDataset, honoring a test-injected module override."""
    mock_cls = getattr(sys.modules[__name__], "StreamingLeRobotDataset", None)
    if mock_cls is not None:
        return mock_cls
    try:
        from lerobot.datasets import StreamingLeRobotDataset

        return StreamingLeRobotDataset
    except (ImportError, ValueError, RuntimeError) as exc:
        raise ImportError(
            f"StreamingLeRobotDataset unavailable ({exc}). "
            "Install with: pip install 'strands-robots[lerobot]' "
            "(needs torchcodec for video keys; on aarch64/Jetson that means "
            "torch>=2.11 + torchcodec>=0.11). "
            "For proprio-only streaming without torchcodec, use drop_videos=True "
            "with a delta_timestamps covering the non-video keys you need."
        ) from exc


def _tolerance_error(value: Any) -> str | None:
    """Return why ``tolerance_s`` cannot be used as a grid-match half-width, else None."""
    if error := finite_number_error(value, "tolerance_s", "open"):
        return error
    if float(value) < 0.0:
        return (
            f"open: tolerance_s must be >= 0, got {refusal_repr(value)}. It is the half-width of the "
            "grid-match window, so a negative one is satisfied by no delta at all and "
            "refuses an on-grid delta_timestamps. Pass 0 to require an exact grid match."
        )
    return None


# Every numeric knob ``open`` forwards, with the domain it is checked against
# before lerobot sees it - so a bad value fails at open, not on the first frame.
_NUMERIC_DOMAINS: dict[str, Callable[[Any], str | None]] = {
    "tolerance_s": _tolerance_error,
    "buffer_size": lambda value: positive_count_error(value, "buffer_size", "open"),
    "max_num_shards": lambda value: positive_count_error(value, "max_num_shards", "open"),
    "seed": lambda value: non_negative_count_error(value, "seed", "open"),
}

# Every boolean flag ``open`` takes. A non-boolean is refused rather than read
# for its truthiness: ``shuffle="false"`` would otherwise shuffle.
_BOOLEAN_FLAGS: tuple[str, ...] = ("streaming", "shuffle", "return_uint8", "validate_deltas", "drop_videos")

_VIDEO_KEY_PREFIX = "observation.images."


class StreamingDatasetReader:
    """Thin facade over ``StreamingLeRobotDataset``: iterate frames, or wrap in a DataLoader.

    Build one with :meth:`open` (or the :func:`stream_dataset` alias). The
    underlying lerobot dataset is ``.dataset``.
    """

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    @classmethod
    def open(
        cls,
        repo_id: str,
        *,
        root: str | None = None,
        episodes: list[int] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        image_transforms: Callable | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        streaming: bool = True,
        buffer_size: int = 1000,
        max_num_shards: int = 16,
        seed: int = 42,
        shuffle: bool = True,
        return_uint8: bool = True,
        validate_deltas: bool = True,
        drop_videos: bool = False,
        repo_type: str = "dataset",
    ) -> StreamingDatasetReader:
        """Open ``repo_id`` for streaming.

        Args:
            repo_id: Hub dataset id (``org/name``). A local directory a recorder
                registered under that id is used as ``root`` when ``root`` is
                not given.
            root: Local dataset directory; overrides the Hub.
            episodes: Episode indices to stream; all when ``None``.
            delta_timestamps: Per-key time offsets (seconds) to stack per frame.
                Checked against the dataset's fps grid when ``validate_deltas``.
            image_transforms: Callable applied to every image tensor.
            tolerance_s: Half-width of the grid-match window (``>= 0``).
            revision: Hub revision (branch, tag or commit).
            streaming: ``False`` materializes the dataset instead.
            buffer_size: Reservoir the reader yields from, in frames
                (``> 0``); ``1`` is half of capture order - see Ordering.
            max_num_shards: Parquet shards interleaved (``> 0``); ``1`` is the
                other half of capture order - see Ordering.
            seed: Shuffle seed (``>= 0``).
            shuffle: Whether the reorder is reproducible ACROSS exhaustions -
                NOT whether it happens - see Ordering.
            return_uint8: Stream images as uint8 (a quarter of float32's
                bandwidth); policies normalize either.
            validate_deltas: Refuse ``delta_timestamps`` off the fps grid.
            drop_videos: Proprio-only streaming with no video decode (no
                torchcodec needed). Requires ``delta_timestamps`` naming at
                least one non-video key; camera keys in it are dropped.
            repo_type: ``"dataset"`` (the versioned Hub namespace) or
                ``"bucket"`` (Hub storage buckets).

        Ordering:
            ``shuffle=False`` alone still reads shuffled frames.
            ``StreamingLeRobotDataset`` reorders at two levels either way - it
            samples a shard at random per frame, then yields from a reservoir of
            ``buffer_size`` - and ``shuffle`` selects only which generator drives
            that reorder (one reseeded from ``seed`` on every exhaustion, versus
            the dataset's own advancing one), i.e. reproducibility ACROSS
            epochs. Capture order therefore needs ``buffer_size=1`` (a reservoir
            of one has nothing to reorder) together with ``max_num_shards=1`` (a
            single shard has nothing to interleave). Nothing reports a shuffled
            read, so an eval or replay loop that asked for order with ``shuffle``
            alone silently consumes frames out of order.

        Raises:
            ValueError: A flag that is not a boolean, a numeric knob outside
                its domain, or ``drop_videos=True`` without a usable
                ``delta_timestamps``.
            ImportError: lerobot's streaming dataset is not importable.
        """
        supplied_flags = {
            "streaming": streaming,
            "shuffle": shuffle,
            "return_uint8": return_uint8,
            "validate_deltas": validate_deltas,
            "drop_videos": drop_videos,
        }
        for flag in _BOOLEAN_FLAGS:
            if error := boolean_flag_error(supplied_flags[flag], flag, "open"):
                raise ValueError(error)
        supplied = {
            "tolerance_s": tolerance_s,
            "buffer_size": buffer_size,
            "max_num_shards": max_num_shards,
            "seed": seed,
        }
        for param, domain in _NUMERIC_DOMAINS.items():
            if error := domain(supplied[param]):
                raise ValueError(error)

        if root is None and (local := local_dataset_dir(repo_id)) is not None:
            root = str(local)

        if drop_videos:
            if delta_timestamps:
                delta_timestamps = {
                    k: v for k, v in delta_timestamps.items() if not k.startswith(_VIDEO_KEY_PREFIX)
                } or None
            if delta_timestamps is None:
                raise ValueError(
                    "drop_videos=True requires delta_timestamps with at least "
                    "one non-video key: without it, every feature (including "
                    "camera keys, via torchcodec decode) is streamed and "
                    "drop_videos has no effect. Pass e.g. "
                    "delta_timestamps={'observation.state': [0.0], 'action': [0.0]}."
                )

        StreamingCls = _get_streaming_cls()
        optional = dict(
            root=root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=revision,
        )
        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            **{k: v for k, v in optional.items() if v is not None},
            "tolerance_s": tolerance_s,
            "streaming": streaming,
            "buffer_size": buffer_size,
            "max_num_shards": max_num_shards,
            "seed": seed,
            "shuffle": shuffle,
            "return_uint8": return_uint8,
            "repo_type": repo_type,
        }
        logger.info(
            "Opening StreamingLeRobotDataset: %s (streaming=%s, buffer=%d, shards=%d)",
            repo_id,
            streaming,
            buffer_size,
            max_num_shards,
        )
        ds = StreamingCls(**kwargs)
        if drop_videos:
            _hide_video_features(ds)
        if delta_timestamps and validate_deltas:
            from lerobot.datasets.feature_utils import check_delta_timestamps

            check_delta_timestamps(delta_timestamps, ds.fps, tolerance_s, raise_value_error=True)
        return cls(ds)

    def dataloader(self, batch_size: int = 64, num_workers: int = 0, **kw: Any) -> Any:
        """Wrap the stream in a ``torch.utils.data.DataLoader``.

        ``shuffle`` is ignored: the stream shuffles internally (see
        :meth:`open`'s Ordering note). With ``num_workers > 0`` video decode
        parallelizes across the worker processes and must not ALSO run in the
        main process - lerobot documents a segfault when a second
        ``num_workers=0`` loader touches the same video reader. lerobot's own
        ``make_dataset`` couples ``max_num_shards = num_workers``; to match it,
        pass the same N to :meth:`open`'s ``max_num_shards`` and here.
        """
        import torch

        if kw.pop("shuffle", None):
            logger.warning("Ignoring shuffle=True: streaming shuffles internally.")
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers, **kw)

    @property
    def num_frames(self) -> Any:
        """Total frames in the dataset."""
        return self.dataset.num_frames

    @property
    def num_episodes(self) -> Any:
        """Total episodes in the dataset."""
        return self.dataset.num_episodes

    @property
    def fps(self) -> Any:
        """Frames per second the dataset was recorded at."""
        return self.dataset.fps

    @property
    def meta(self) -> Any:
        """The dataset's ``LeRobotDatasetMetadata``."""
        return self.dataset.meta

    def __iter__(self) -> Any:
        return iter(self.dataset)


def _hide_video_features(ds: Any) -> None:
    """Make ``drop_videos=True`` actually skip video decode.

    lerobot's ``make_frame`` decodes every key in ``meta.video_keys`` whether or
    not ``delta_timestamps`` names it, so a proprio-only reader without a
    working torchcodec opened fine and raised on its first frame. ``video_keys``
    is derived from the metadata's feature table, so hide the video features on
    this instance's metadata (memory only; nothing on disk changes).
    """
    meta = getattr(ds, "meta", None)
    info = getattr(meta, "info", None)
    if info is None:
        return
    is_mapping = isinstance(info, dict)
    features = info.get("features") if is_mapping else getattr(info, "features", None)
    if not isinstance(features, dict):
        return
    kept = {k: v for k, v in features.items() if not (isinstance(v, dict) and v.get("dtype") == "video")}
    if is_mapping:
        info["features"] = kept
    else:
        info.features = kept


def stream_dataset(repo_id: str, **kwargs: Any) -> StreamingDatasetReader:
    """Open a streaming reader for a LeRobotDataset without a simulator.

    Module-level alias for :meth:`StreamingDatasetReader.open`; every keyword
    is forwarded unchanged.
    """
    return StreamingDatasetReader.open(repo_id, **kwargs)
