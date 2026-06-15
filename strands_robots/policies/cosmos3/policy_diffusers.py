"""In-process Cosmos 3 backend via ``strands-diffusers`` (no policy server).

Parallel to :mod:`client` (the WebSocket *service* backend). Where
:class:`~strands_robots.policies.cosmos3.client.Cosmos3WebsocketClient` talks
msgpack+NumPy to a running ``cosmos_framework`` RoboLab policy server, this
backend loads Cosmos 3 **in-process** through ``strands-diffusers``'
``use_diffusers`` entry point (which wraps ``diffusers.from_pretrained`` for the
``Cosmos3OmniPipeline``). One forward pass returns the predicted world video,
optional sound, *and* the robot action chunk in a single call.

The backend exposes the same ``infer(observation) -> dict`` contract the service
client does so :class:`~strands_robots.policies.cosmos3.policy.Cosmos3Policy` is
backend-agnostic downstream::

    {"action": np.ndarray[T, D], "video": str | np.ndarray | None, "sound": ...}

Action modes (``CosmosActionCondition.mode``):

* ``policy`` (default) - first frame + task prompt -> future video + actions.
  The 1:1 match for the robots policy contract.
* ``forward_dynamics`` - first frame + given ``raw_actions`` -> future video.
  Predicts the world; yields no action chunk (surface the video via
  ``Cosmos3Policy.last_rollout``).
* ``inverse_dynamics`` - an observed video -> the actions between frames.

Why ``strands-diffusers`` is optional/lazy: it pulls ``diffusers`` + ``torch``
(a heavy GPU stack). The service backend stays the default so a plain
``pip install strands-robots[cosmos3-service]`` (msgpack + websockets only)
keeps working. ``strands-diffusers`` + its ``diffusers`` pin compose with
``numpy>=2`` (verified against lerobot 0.5.x in the same env), so the
``cosmos3-diffusers`` extra is co-installable with ``cosmos3-service`` and
``lerobot``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .embodiments import Cosmos3Embodiment

logger = logging.getLogger(__name__)

# CosmosActionCondition modes that produce an action chunk consumable by
# Cosmos3Policy.get_actions. ``forward_dynamics`` predicts world video only.
ACTION_PRODUCING_MODES = ("policy", "inverse_dynamics")
ALL_MODES = ("policy", "forward_dynamics", "inverse_dynamics")

_DEFAULT_MODEL = "nvidia/Cosmos3-Nano"


def _install_hint() -> str:
    """Actionable message when ``strands-diffusers`` is not importable."""
    return (
        "Cosmos3Policy(backend='diffusers') needs the optional 'strands-diffusers' "
        "package, which was not importable. Install it (and the diffusers-from-source "
        "pin that ships Cosmos3OmniPipeline):\n"
        "  uv pip install strands-robots[cosmos3-diffusers]\n"
        "  # or directly:\n"
        "  uv pip install strands-diffusers 'diffusers @ git+https://github.com/huggingface/diffusers'\n"
        "Then retry. Or use the service backend (no in-process GPU load): "
        "Cosmos3Policy(backend='service', host=..., port=...)."
    )


def _image_keys(server_key_iter: Any) -> list[str]:
    """Filter OpenPI observation keys down to image keys."""
    out = []
    for k in server_key_iter:
        low = str(k).lower()
        if "image" in low or "rgb" in low or "cam" in low:
            out.append(k)
    return out


class Cosmos3DiffusersBackend:
    """In-process Cosmos 3 inference via ``strands-diffusers``' ``use_diffusers``.

    Args:
        embodiment: Active :class:`Cosmos3Embodiment` (provides ``domain_name``,
            ``action_chunk_size``, ``fps``, ``camera_keys``).
        model: HF repo id or local path of the Cosmos 3 omni checkpoint
            (default ``"nvidia/Cosmos3-Nano"``).
        mode: One of :data:`ALL_MODES`. ``policy`` is the control default.
        resolution_tier: Cosmos conditioning resolution tier (e.g. ``480``).
        view_point: Optional Cosmos ``view_point`` tag (e.g. ``"ego_view"``).
        device: ``"cuda"`` / ``"cpu"`` / ``"auto"`` (``None`` lets
            ``use_diffusers`` pick).
        dtype: Torch dtype string (default ``"bfloat16"``).
        num_inference_steps: Diffusion sampling steps for the pipeline run.
        guidance_scale: Classifier-free guidance scale.
        use_diffusers_fn: Dependency injection for tests. When ``None`` the real
            ``strands_diffusers.use_diffusers`` is imported lazily at
            construction; a missing package raises :class:`ImportError` with an
            actionable install hint (no silent default, per AGENTS.md #6).

    Notes:
        Cosmos3OmniPipeline emits an action tensor of shape
        ``[num_chunks, T, action_dim]`` normalized to ``[-1, 1]``. :meth:`infer`
        returns the **first** chunk reshaped to ``[T, D]`` so the policy's
        ``_unpack_actions`` (shared with the service backend) consumes it
        unchanged. The column order matches the embodiment ``action_layouts``
        (DROID ``joint_pos`` = ``[joint_0..joint_6, gripper]``); the ``[-1, 1]``
        normalization is preserved verbatim (the consumer denormalizes to its
        actuator range, exactly as for the service action chunk).
    """

    def __init__(
        self,
        embodiment: Cosmos3Embodiment,
        model: str | None = None,
        mode: str = "policy",
        resolution_tier: int = 480,
        view_point: str | None = None,
        device: str | None = None,
        dtype: str = "bfloat16",
        num_inference_steps: int = 30,
        guidance_scale: float = 1.0,
        use_diffusers_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        if mode not in ALL_MODES:
            raise ValueError(f"Unknown Cosmos 3 action mode {mode!r}. Available: {list(ALL_MODES)}")
        self.embodiment = embodiment
        self.model = model or _DEFAULT_MODEL
        self.mode = mode
        self.resolution_tier = resolution_tier
        self.view_point = view_point
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        if use_diffusers_fn is None:
            try:
                from strands_diffusers import use_diffusers as _imported
            except ImportError as e:
                raise ImportError(_install_hint()) from e
            use_diffusers_fn = _imported
        self._use_diffusers: Callable[..., dict[str, Any]] = use_diffusers_fn
        logger.info(
            "Cosmos3DiffusersBackend ready [model=%s domain=%s mode=%s tier=%d dtype=%s]",
            self.model,
            self.embodiment.domain_name,
            self.mode,
            self.resolution_tier,
            self.dtype,
        )

    def _first_frame(self, observation: dict[str, Any]) -> np.ndarray:
        """Pick the first available camera frame from the OpenPI observation."""
        # Prefer the embodiment's declared camera keys, then any image-like key.
        candidates = list(self.embodiment.camera_keys) + _image_keys(observation)
        for key in candidates:
            val = observation.get(key)
            if val is None:
                continue
            arr = np.asarray(val)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                return arr
        raise ValueError(
            "Cosmos3DiffusersBackend requires at least one camera frame in the "
            f"observation; none of {candidates} held an (H, W, 3) array. "
            f"Observation keys: {sorted(observation)}"
        )

    def _check(self, result: dict[str, Any], stage: str) -> None:
        """Raise with the tool error text when a use_diffusers call failed."""
        if not isinstance(result, dict) or result.get("status") != "success":
            text = ""
            try:
                text = result["content"][0]["text"]
            except (KeyError, IndexError, TypeError):
                text = str(result)
            raise RuntimeError(f"Cosmos 3 diffusers {stage} failed: {text}")

    def infer(self, observation: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Run Cosmos 3 in-process and return action + world video/sound.

        Args:
            observation: OpenPI-shaped observation dict (same one the service
                backend sends): a ``prompt`` string plus ``observation/<cam>``
                image arrays and state keys.
            **kwargs: ``raw_actions`` (required for ``mode="forward_dynamics"``)
                and ``video`` (an observed video for ``mode="inverse_dynamics"``)
                may be passed through from the caller.

        Returns:
            ``{"action": np.ndarray[T, D] | None, "video": ..., "sound": ...}``.
            ``action`` is ``None`` only for ``forward_dynamics`` (world-only).
        """
        prompt = observation.get("prompt", "")

        cond_params: dict[str, Any] = {
            "mode": self.mode,
            "chunk_size": self.embodiment.action_chunk_size,
            "domain_name": self.embodiment.domain_name,
            "resolution_tier": self.resolution_tier,
        }
        if self.view_point is not None:
            cond_params["view_point"] = self.view_point

        if self.mode == "inverse_dynamics":
            video = kwargs.get("video")
            if video is None:
                raise ValueError(
                    "Cosmos 3 mode='inverse_dynamics' needs an observed video; "
                    "pass video=<path|ndarray> to get_actions (recovers the actions "
                    "between frames)."
                )
            cond_params["video"] = video
        elif self.mode == "forward_dynamics":
            raw_actions = kwargs.get("raw_actions")
            if raw_actions is None:
                raise ValueError(
                    "Cosmos 3 mode='forward_dynamics' needs raw_actions to roll the "
                    "world forward; pass raw_actions=<array> to get_actions."
                )
            cond_params["image"] = self._first_frame(observation)
            cond_params["raw_actions"] = raw_actions
        else:  # policy
            cond_params["image"] = self._first_frame(observation)

        cache_key = f"cosmos3_cond_{id(self):x}"
        cond = self._use_diffusers(
            action="call",
            target="CosmosActionCondition",
            parameters=cond_params,
            cache_key=cache_key,
            device=self.device,
            dtype=self.dtype,
        )
        self._check(cond, "CosmosActionCondition")

        run = self._use_diffusers(
            action="run",
            pipeline="Cosmos3OmniPipeline",
            model=self.model,
            parameters={
                "prompt": prompt,
                "action": f"cached:{cache_key}",
                "fps": self.embodiment.fps,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
            },
            device=self.device,
            dtype=self.dtype,
        )
        self._check(run, "Cosmos3OmniPipeline")

        data = run.get("data") or {}
        artifacts = run.get("artifacts") or []
        return {
            "action": _extract_action(data, self.mode),
            "video": _extract_video(data, artifacts),
            "sound": _extract_sound(data, artifacts),
        }

    def reset(self) -> None:
        """Per-episode reset - free the cached conditioning object, best-effort."""
        try:
            self._use_diffusers(action="clear_cache", cache_key=f"cosmos3_cond_{id(self):x}")
        except Exception as e:  # noqa: BLE001 - reset is best-effort
            logger.info("Cosmos3DiffusersBackend.reset best-effort clear failed: %s", e)


def _extract_action(data: dict[str, Any], mode: str) -> np.ndarray | None:
    """Pull the ``[T, D]`` first action chunk out of a serialized run result.

    ``strands-diffusers`` serializes a ``Cosmos3OmniPipelineOutput`` action field
    to ``{"type": "action", "data": [[...]], "chunk_shape": [T, D],
    "num_chunks": N}`` (full nested lists, normalized to ``[-1, 1]``). We return
    the first chunk as ``np.ndarray[T, D]``.
    """
    action_field = data.get("action") if isinstance(data, dict) else None
    if not action_field:
        if mode == "forward_dynamics":
            return None  # world-only mode produces no action chunk
        raise RuntimeError(
            f"Cosmos 3 diffusers run (mode={mode!r}) returned no action field. "
            "Expected a Cosmos3OmniPipelineOutput with an 'action' tensor."
        )
    chunks = action_field["data"] if isinstance(action_field, dict) else action_field
    arr = np.asarray(chunks, dtype=np.float32)
    # Cosmos emits [num_chunks, T, D]; take the first chunk -> [T, D].
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def _extract_video(data: dict[str, Any], artifacts: list[str]) -> str | None:
    """Return the predicted world-video artifact path, if any."""
    for a in artifacts:
        if str(a).lower().endswith((".mp4", ".gif", ".webm")):
            return a
    if isinstance(data, dict) and data.get("video") is not None:
        return data["video"]
    return None


def _extract_sound(data: dict[str, Any], artifacts: list[str]) -> str | None:
    """Return the predicted sound artifact path, if any."""
    for a in artifacts:
        if str(a).lower().endswith((".wav", ".flac", ".mp3")):
            return a
    if isinstance(data, dict) and data.get("sound") is not None:
        return data["sound"]
    return None
