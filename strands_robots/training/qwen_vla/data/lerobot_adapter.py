"""LeRobotDataset -> Qwen-VLA training tensors adapter (section 2.4).

Converts a LeRobot-style frame (the schema ``DatasetRecorder`` emits) into the
unified Qwen-VLA training sample:

    (video, state, language, Y[H x K], mask[H x K])

where ``Y`` is the quantile-normalized, zero-padded action chunk and ``mask``
excludes the padding from the loss. The embodiment prompt (``language``) is
rendered from the shared :class:`EmbodimentTag` so it matches inference exactly.

The transform core is pure NumPy (uses the inference ``normalize`` /
``pad_to_width`` / ``build_channel_mask`` helpers), so it unit-tests without
torch or a real LeRobotDataset. A thin ``to_torch`` converts the resulting
sample dict to tensors behind the ``qwen-vla-train`` extra.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from strands_robots.policies.qwen_vla.normalize import build_channel_mask, normalize, pad_to_width
from strands_robots.training.qwen_vla.data.embodiment_tags import EmbodimentTag


@dataclass
class QwenVlaSample:
    """One adapted training sample.

    Attributes:
        video: ``{view_tag: ndarray(H_img, W, C) uint8}`` per camera view.
        state: ``(D_state,)`` float32 proprioceptive vector.
        language: The rendered embodiment prompt (section 2.3).
        action: ``(H, K)`` quantile-normalized, zero-padded action chunk.
        mask: ``(H, K)`` binary validity mask (1 valid, 0 padding).
        c: Number of valid action channels.
        h_task: Number of valid timesteps.
    """

    video: dict[str, np.ndarray]
    state: np.ndarray
    language: str
    action: np.ndarray
    mask: np.ndarray
    c: int
    h_task: int


class LeRobotAdapter:
    """Adapts LeRobot frames to Qwen-VLA training samples for one embodiment.

    Args:
        embodiment: The :class:`EmbodimentTag` (defines prompt + chunk_size H).
        video_keys: Ordered LeRobot frame keys for cameras (e.g.
            ``["observation.images.front", "observation.images.wrist"]``).
        view_tags: Per-``video_key`` Qwen-VLA view tag (``ego`` / wrist tags).
        state_keys: Ordered LeRobot frame keys whose values concatenate into
            the proprioceptive state vector.
        action_dim: Fixed unified channel width K (zero-pad target).
        quantile_stats: Optional ``{"q_low", "q_high"}`` per-channel stats for
            normalization. When ``None``, actions pass through unnormalized
            (caller must opt out explicitly - no silent mis-normalization).
    """

    def __init__(
        self,
        *,
        embodiment: EmbodimentTag,
        video_keys: list[str],
        view_tags: dict[str, str],
        state_keys: list[str],
        action_dim: int = 32,
        quantile_stats: dict[str, np.ndarray] | None = None,
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        self.embodiment = embodiment
        self.video_keys = video_keys
        self.view_tags = view_tags
        self.state_keys = state_keys
        self.action_dim = action_dim
        self.quantile_stats = quantile_stats
        self.h = embodiment.chunk_size

    def _concat_state(self, frame: dict[str, Any]) -> np.ndarray:
        """Concatenate the configured state keys into one float32 vector."""
        parts: list[np.ndarray] = []
        for key in self.state_keys:
            if key not in frame:
                raise ValueError(f"state key '{key}' missing from frame")
            parts.append(np.atleast_1d(np.asarray(frame[key], dtype=np.float32)))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def _build_action_chunk(self, action_chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Normalize + zero-pad an ``(H, c)`` chunk to ``(H, K)`` + mask.

        Returns ``(padded_action, mask, c, h_task)``.
        """
        arr = np.asarray(action_chunk, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"action_chunk must be 2-D (H, c), got {arr.shape}")
        h_task, c = arr.shape
        if h_task > self.h:
            raise ValueError(f"action chunk horizon {h_task} exceeds embodiment H={self.h}")

        if self.quantile_stats is not None:
            arr = normalize(arr, self.quantile_stats)

        # Pad the time axis up to H if the chunk is short (e.g. nav waypoints).
        if h_task < self.h:
            padded_time = np.zeros((self.h, c), dtype=np.float32)
            padded_time[:h_task] = arr
            arr = padded_time

        padded = pad_to_width(arr, self.action_dim)
        mask = build_channel_mask(c=c, k=self.action_dim, h_task=h_task, h=self.h)
        return padded, mask, c, h_task

    def adapt(self, frame: dict[str, Any], action_chunk: np.ndarray, instruction: str) -> QwenVlaSample:
        """Convert a LeRobot frame + action chunk into a :class:`QwenVlaSample`.

        Args:
            frame: LeRobot frame dict (cameras + state keys).
            action_chunk: ``(H_task, c)`` raw action chunk for this frame.
            instruction: Task instruction (embodiment prompt is rendered from it).

        Returns:
            A fully-populated :class:`QwenVlaSample`.

        Raises:
            ValueError: If a configured key is missing or shapes are invalid.
        """
        video: dict[str, np.ndarray] = {}
        for key in self.video_keys:
            if key not in frame:
                raise ValueError(f"video key '{key}' missing from frame")
            tag = self.view_tags.get(key, key)
            video[tag] = np.asarray(frame[key], dtype=np.uint8)

        state = self._concat_state(frame)
        action, mask, c, h_task = self._build_action_chunk(action_chunk)
        language = self.embodiment.render_prompt(instruction)

        return QwenVlaSample(video=video, state=state, language=language, action=action, mask=mask, c=c, h_task=h_task)


def to_torch(sample: QwenVlaSample) -> dict[str, Any]:
    """Convert a :class:`QwenVlaSample` to a torch tensor dict (train loop).

    Import-guarded behind the ``qwen-vla-train`` extra.
    """
    from strands_robots.utils import require_optional

    torch: Any = require_optional("torch", extra="qwen-vla-train", purpose="Qwen-VLA training tensors")
    return {
        "video": {k: torch.from_numpy(v) for k, v in sample.video.items()},
        "state": torch.from_numpy(sample.state),
        "language": sample.language,
        "action": torch.from_numpy(sample.action),
        "mask": torch.from_numpy(sample.mask),
    }


__all__ = ["QwenVlaSample", "LeRobotAdapter", "to_torch"]
