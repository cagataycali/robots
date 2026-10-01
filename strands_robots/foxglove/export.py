"""MCAP as a sidecar and an export format: never the training format.

Two entry points:

* :func:`export_episode` reads one episode of a LeRobot v3 dataset through
  lerobot's own reader (video decode included) and writes it as an MCAP file
  Foxglove opens and scrubs: ``/observation/state`` and ``/action/state`` as
  ``lerobot.Scalars`` JSON (the shape lerobot's own Foxglove logger uses, so
  its layouts apply), one ``CompressedImage`` JPEG channel per camera, and one
  ``/lerobot/episode`` record naming the dataset, episode, task and fps.
  Timestamps come from the dataset's ``timestamp`` column, so the file is
  seekable.
* :func:`mcap_info` summarises any MCAP with the independent ``mcap`` reader:
  channels, message counts, time span, schema names.

The live sidecar (``Robot(foxglove=True, foxglove_mcap=path)``) is in
:mod:`strands_robots.foxglove.bridge`. Measured on the spike: an MCAP is about
3.7 times the size of the same frames as LeRobot v3, which is why export and
sidecar are both opt-in and neither replaces the dataset.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from strands_robots.foxglove.options import foxglove_mcap_error
from strands_robots.utils import require_optional

#: JPEG quality for exported camera frames.
EXPORT_JPEG_QUALITY = 80

SCALARS_SCHEMA = {
    "type": "object",
    "title": "lerobot.Scalars",
    "properties": {
        "scalars": {
            "type": "array",
            "items": {"type": "object", "properties": {"label": {"type": "string"}, "value": {"type": "number"}}},
        }
    },
}

EPISODE_SCHEMA = {
    "type": "object",
    "title": "lerobot.Episode",
    "properties": {
        "repo_id": {"type": "string"},
        "episode": {"type": "integer"},
        "task": {"type": "string"},
        "fps": {"type": "number"},
        "frames": {"type": "integer"},
    },
}


def _to_hwc_uint8(tensor: Any) -> np.ndarray:
    """A lerobot image (CHW float in 0-1 or HWC uint8) as an HWC uint8 array."""
    array = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[2] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def _jpeg(array: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=EXPORT_JPEG_QUALITY)
    return buffer.getvalue()


def _episode_count_error(dataset: Any, episode_index: Any) -> str | None:
    from strands_robots.utils import refusal_repr

    total = int(getattr(dataset.meta, "total_episodes", 0) or 0)
    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
        return f"export_episode: episode_index must be a non-negative int, got {refusal_repr(episode_index)}."
    if total and episode_index >= total:
        return (
            f"export_episode: episode_index {refusal_repr(episode_index)} is past the last episode "
            f"({total - 1}) of this dataset."
        )
    return None


def export_episode(
    dataset_root: str | os.PathLike[str],
    episode_index: int,
    out_path: str | os.PathLike[str],
    *,
    repo_id: str | None = None,
) -> dict[str, Any]:
    """Write one LeRobot v3 episode as an MCAP file Foxglove can open.

    Args:
        dataset_root: The dataset directory (``meta/info.json`` lives in it).
        episode_index: Which episode, zero-based.
        out_path: The MCAP to create. An existing file is refused.
        repo_id: The dataset's repo id, recorded in the file; defaults to the
            one in ``meta/info.json`` or the directory name.

    Returns:
        ``{"path", "frames", "fps", "cameras", "channels", "bytes", "seconds"}``.

    Raises:
        ImportError: The ``[foxglove]`` or ``[lerobot]`` extra is missing.
        ValueError: ``out_path`` exists, or ``episode_index`` is out of range.
    """
    require_optional(
        "foxglove", pip_install="foxglove-sdk", extra="foxglove", purpose="writing an MCAP (export_episode)"
    )
    require_optional("lerobot", extra="lerobot", purpose="reading a LeRobot dataset (export_episode)")
    import foxglove as fox
    from foxglove.channels import CompressedImageChannel
    from foxglove.mcap import MCAPCompression, MCAPWriteOptions
    from foxglove.messages import CompressedImage, Timestamp
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if error := foxglove_mcap_error(out_path, "out_path", "export_episode"):
        raise ValueError(error)
    root = Path(dataset_root)
    repo = repo_id or _repo_id_of(root)
    probe = LeRobotDataset(repo, root=str(root))
    if error := _episode_count_error(probe, episode_index):
        raise ValueError(error)
    dataset = LeRobotDataset(repo, root=str(root), episodes=[episode_index])
    meta = dataset.meta
    camera_keys = list(meta.camera_keys)
    names = {k: meta.features[k].get("names") if k in meta.features else None for k in ("observation.state", "action")}
    frames = len(dataset)

    started = time.perf_counter()
    options = MCAPWriteOptions(compression=MCAPCompression.Zstd, profile="strands-robots")
    context = fox.Context()
    writer = fox.open_mcap(str(out_path), context=context, writer_options=options)
    try:
        scalar_topics = {
            key: topic
            for key, topic in (("observation.state", "/observation/state"), ("action", "/action/state"))
            if key in meta.features
        }
        image_topics = {key: f"/observation/images/{key.rsplit('.', 1)[-1]}" for key in camera_keys}
        scalar_channels = {
            key: fox.Channel(topic, schema=SCALARS_SCHEMA, message_encoding="json", context=context)
            for key, topic in scalar_topics.items()
        }
        image_channels: dict[str, Any] = {
            key: CompressedImageChannel(topic=topic, context=context) for key, topic in image_topics.items()
        }
        episode_channel = fox.Channel(
            "/lerobot/episode", schema=EPISODE_SCHEMA, message_encoding="json", context=context
        )
        base_ns = time.time_ns()
        for i in range(frames):
            sample = dataset[i]
            t_ns = base_ns + int(round(float(sample["timestamp"]) * 1e9))
            if i == 0:
                episode_channel.log(
                    {
                        "repo_id": repo,
                        "episode": int(episode_index),
                        "task": str(sample.get("task", "")),
                        "fps": float(meta.fps),
                        "frames": int(frames),
                    },
                    log_time=t_ns,
                )
            for key, channel in scalar_channels.items():
                vector = np.asarray(sample[key]).reshape(-1).tolist()
                labels = names[key] or [f"{key}.{j}" for j in range(len(vector))]
                channel.log(
                    {
                        "scalars": [
                            {"label": str(label), "value": float(v)} for label, v in zip(labels, vector, strict=False)
                        ]
                    },
                    log_time=t_ns,
                )
            stamp = Timestamp(sec=t_ns // 1_000_000_000, nsec=t_ns % 1_000_000_000)
            for key, image_channel in image_channels.items():
                image_channel.log(
                    CompressedImage(
                        timestamp=stamp,
                        frame_id=key.rsplit(".", 1)[-1],
                        data=_jpeg(_to_hwc_uint8(sample[key])),
                        format="jpeg",
                    ),
                    log_time=t_ns,
                )
    finally:
        writer.close()
    size = Path(out_path).stat().st_size
    return {
        "path": str(out_path),
        "frames": int(frames),
        "fps": float(meta.fps),
        "cameras": camera_keys,
        "channels": sorted([*scalar_topics.values(), *image_topics.values(), "/lerobot/episode"]),
        "bytes": int(size),
        "seconds": round(time.perf_counter() - started, 3),
    }


def _repo_id_of(root: Path) -> str:
    import json

    info = root / "meta" / "info.json"
    if info.is_file():
        try:
            repo = json.loads(info.read_text(encoding="utf-8")).get("repo_id")
        except (OSError, ValueError):
            repo = None
        if isinstance(repo, str) and repo:
            return repo
    return f"local/{root.name}"


def mcap_info(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Summarise an MCAP file with the independent ``mcap`` reader.

    Args:
        path: The file to read.

    Returns:
        ``{"path", "bytes", "messages", "channels": {topic: {"schema", "encoding", "messages"}},
        "start_ns", "end_ns", "seconds"}``. A file with no summary section
        still reports its channels, with message counts from a full pass.

    Raises:
        ImportError: The ``[foxglove]`` extra (which carries ``mcap``) is missing.
        FileNotFoundError: ``path`` does not exist.
    """
    require_optional("mcap", extra="foxglove", purpose="reading an MCAP summary (mcap_info)")
    from mcap.reader import make_reader

    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"mcap_info: {file} is not a file.")
    with file.open("rb") as handle:
        reader = make_reader(handle)
        summary = reader.get_summary()
        channels: dict[str, dict[str, Any]] = {}
        start_ns: int | None = None
        end_ns: int | None = None
        total = 0
        if summary is not None and summary.statistics is not None and summary.channels:
            stats = summary.statistics
            for channel_id, channel in summary.channels.items():
                schema = summary.schemas.get(channel.schema_id)
                channels[channel.topic] = {
                    "schema": schema.name if schema is not None else "",
                    "encoding": channel.message_encoding,
                    "messages": int(stats.channel_message_counts.get(channel_id, 0)),
                }
            total = int(stats.message_count)
            start_ns, end_ns = int(stats.message_start_time), int(stats.message_end_time)
        else:
            for schema, channel, message in reader.iter_messages():
                entry = channels.setdefault(
                    channel.topic,
                    {
                        "schema": schema.name if schema is not None else "",
                        "encoding": channel.message_encoding,
                        "messages": 0,
                    },
                )
                entry["messages"] += 1
                total += 1
                start_ns = message.log_time if start_ns is None else min(start_ns, message.log_time)
                end_ns = message.log_time if end_ns is None else max(end_ns, message.log_time)
    seconds = (end_ns - start_ns) / 1e9 if start_ns is not None and end_ns is not None else 0.0
    return {
        "path": str(file),
        "bytes": file.stat().st_size,
        "messages": total,
        "channels": dict(sorted(channels.items())),
        "start_ns": start_ns,
        "end_ns": end_ns,
        "seconds": round(seconds, 3),
    }
