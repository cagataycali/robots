"""In-process Cosmos 3 backend via native ``diffusers`` (no policy server).

Parallel to :mod:`client` (the WebSocket *service* backend). Where
:class:`~strands_robots.policies.cosmos3.client.Cosmos3WebsocketClient` talks
msgpack+NumPy to a running ``cosmos_framework`` RoboLab policy server, this
backend loads Cosmos 3 **in-process** directly through Hugging Face
``diffusers`` - the upstream :class:`diffusers.Cosmos3OmniPipeline` (loaded with
``Cosmos3OmniPipeline.from_pretrained``) driven by a
:class:`diffusers.CosmosActionCondition`. One forward pass returns the predicted
world video, optional sound, *and* the robot action chunk in a single call.

The backend exposes the same ``infer(observation) -> dict`` contract the service
client does so :class:`~strands_robots.policies.cosmos3.policy.Cosmos3Policy` is
backend-agnostic downstream::

    {"action": np.ndarray[T, D] | None, "video": np.ndarray | None, "sound": ...}

Action modes (``CosmosActionCondition.mode``):

* ``policy`` (default) - first frame + task prompt -> future video + actions.
  The 1:1 match for the robots policy contract.
* ``forward_dynamics`` - first frame + given ``raw_actions`` -> future video.
  Predicts the world; yields no action chunk (surface the video via
  ``Cosmos3Policy.last_rollout``).
* ``inverse_dynamics`` - an observed video -> the actions between frames.

Action width: :class:`Cosmos3OmniPipeline` emits the model's **raw unified
action** of width ``embodiment.raw_action_dim`` (e.g. ``droid_lerobot`` = 10 =
9D end-effector pose + 1D gripper), NOT the service server's post-processed
``joint_pos`` (8D) layout. The columns are named by
``embodiment.raw_action_layout`` accordingly; no IK conversion to joint targets
is fabricated (that is the RoboLab server's job - use ``backend="service"`` for
joint commands).

Why ``diffusers`` is optional/lazy: it pulls ``diffusers`` + ``torch`` +
``transformers`` (a heavy GPU stack). The service backend stays the default so a
plain ``pip install strands-robots[cosmos3-service]`` (msgpack + websockets
only) keeps working. ``diffusers`` composes with ``numpy>=2`` (the same env as
``lerobot`` dataset recording), so the ``cosmos3-diffusers`` extra is
co-installable with ``cosmos3-service`` and ``lerobot``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from .embodiments import Cosmos3Embodiment

logger = logging.getLogger(__name__)

# CosmosActionCondition modes that produce an action chunk consumable by
# Cosmos3Policy.get_actions. ``forward_dynamics`` predicts world video only.
ACTION_PRODUCING_MODES = ("policy", "inverse_dynamics")
ALL_MODES = ("policy", "forward_dynamics", "inverse_dynamics")

_DEFAULT_MODEL = "nvidia/Cosmos3-Nano"


class _PipelineCallable(Protocol):
    """Minimal structural type for the injectable Cosmos3OmniPipeline."""

    def __call__(self, **kwargs: Any) -> Any:
        """Run one forward pass; returns a Cosmos3OmniPipelineOutput-like object."""


def _install_hint() -> str:
    """Actionable message when native ``diffusers`` is not importable."""
    return (
        "Cosmos3Policy(backend='diffusers') needs the optional native 'diffusers' "
        "stack (diffusers + torch + transformers), which was not importable. "
        "Cosmos3OmniPipeline ships only in diffusers-from-source (>0.38), so install "
        "the git pin alongside the extra:\n"
        "  uv pip install strands-robots[cosmos3-diffusers] "
        "'diffusers @ git+https://github.com/huggingface/diffusers'\n"
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


def _resolve_torch_dtype(torch_mod: Any, dtype: str) -> Any:
    """Map a dtype string (``"bfloat16"``) to the matching ``torch`` dtype."""
    resolved = getattr(torch_mod, dtype, None)
    if resolved is None:
        raise ValueError(f"Unknown torch dtype {dtype!r}. Use e.g. 'bfloat16', 'float16', or 'float32'.")
    return resolved


class Cosmos3DiffusersBackend:
    """In-process Cosmos 3 inference via native :class:`diffusers.Cosmos3OmniPipeline`.

    Args:
        embodiment: Active :class:`Cosmos3Embodiment` (provides ``domain_name``,
            ``action_chunk_size``, ``fps``, ``camera_keys``, ``raw_action_dim``).
        model: HF repo id or local path of the Cosmos 3 omni checkpoint
            (default ``"nvidia/Cosmos3-Nano"``).
        mode: One of :data:`ALL_MODES`. ``policy`` is the control default.
        resolution_tier: Cosmos conditioning resolution tier (256/480/704/720).
        view_point: Cosmos ``view_point`` tag (``"ego_view"`` default;
            ``"third_person_view"`` / ``"wrist_view"`` / ``"concat_view"``).
        device: ``"cuda"`` / ``"cpu"`` (``None`` -> ``"cuda"`` when available,
            else ``"cpu"``). Only used when loading the pipeline natively.
        dtype: Torch dtype string (default ``"bfloat16"``).
        num_inference_steps: Diffusion sampling steps for the pipeline run.
        guidance_scale: Classifier-free guidance scale.
        enable_sound: Decode the audio waveform alongside the video.
        enable_safety_checker: Build the pipeline's ``CosmosSafetyChecker``
            (requires the heavy optional ``cosmos_guardrail`` extra). Default
            ``False`` so the pipeline loads without that extra; the load passes
            ``enable_safety_checker=False`` to ``from_pretrained`` to skip the
            checker. Set ``True`` only when ``cosmos_guardrail`` is installed.
        pipeline: Pre-built / injected ``Cosmos3OmniPipeline`` callable
            (dependency injection for tests). When ``None`` the real pipeline is
            loaded lazily via ``Cosmos3OmniPipeline.from_pretrained``; a missing
            ``diffusers`` raises :class:`ImportError` with an actionable install
            hint (no silent default, per AGENTS.md #6).
        condition_cls: Injected :class:`diffusers.CosmosActionCondition` factory
            (dependency injection for tests). When ``None`` it is imported lazily
            from ``diffusers``.

    Notes:
        :class:`Cosmos3OmniPipeline` returns a ``Cosmos3OmniPipelineOutput`` whose
        ``action`` field is a ``list[torch.Tensor]`` (one ``[T, raw_action_dim]``
        chunk). :meth:`infer` returns the first chunk as ``np.ndarray[T, D]`` so
        the policy's ``_unpack_actions`` (shared with the service backend)
        consumes it unchanged. The width is the embodiment's raw unified action
        width (``raw_action_dim``), named by ``raw_action_layout`` - not the
        service ``joint_pos`` layout.
    """

    def __init__(
        self,
        embodiment: Cosmos3Embodiment,
        model: str | None = None,
        mode: str = "policy",
        resolution_tier: int = 480,
        view_point: str = "ego_view",
        device: str | None = None,
        dtype: str = "bfloat16",
        num_inference_steps: int = 35,
        guidance_scale: float = 6.0,
        enable_sound: bool = False,
        enable_safety_checker: bool = False,
        pipeline: _PipelineCallable | None = None,
        condition_cls: Any | None = None,
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
        self.enable_sound = enable_sound
        self.enable_safety_checker = enable_safety_checker

        # Pipeline (callable) + CosmosActionCondition factory. Injected for
        # tests; otherwise imported/loaded lazily from native diffusers.
        self._pipeline: _PipelineCallable = pipeline if pipeline is not None else self._load_pipeline()
        self._condition_cls: Any = condition_cls if condition_cls is not None else self._import_condition_cls()
        logger.info(
            "Cosmos3DiffusersBackend ready [model=%s domain=%s mode=%s tier=%d dtype=%s]",
            self.model,
            self.embodiment.domain_name,
            self.mode,
            self.resolution_tier,
            self.dtype,
        )

    def _load_pipeline(self) -> _PipelineCallable:
        """Load the native ``Cosmos3OmniPipeline`` (heavy GPU import, lazy)."""
        try:
            import torch
            from diffusers import Cosmos3OmniPipeline  # type: ignore[attr-defined]
        except ImportError as e:
            raise ImportError(_install_hint()) from e
        torch_dtype = _resolve_torch_dtype(torch, self.dtype)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # ``Cosmos3OmniPipeline.__init__`` builds a ``CosmosSafetyChecker`` which
        # hard-raises ``ImportError: cosmos_guardrail is not installed`` unless the
        # heavy optional ``cosmos_guardrail`` extra is present. Disable it by
        # default so the in-process backend loads without that extra; callers who
        # installed ``cosmos_guardrail`` can opt back in via
        # ``enable_safety_checker=True``.
        from_pretrained_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        if not self.enable_safety_checker:
            from_pretrained_kwargs["enable_safety_checker"] = False
        pipe = Cosmos3OmniPipeline.from_pretrained(self.model, **from_pretrained_kwargs)
        pipe = pipe.to(device)
        self.device = device
        return pipe

    @staticmethod
    def _import_condition_cls() -> Any:
        """Import :class:`diffusers.CosmosActionCondition` (lazy)."""
        try:
            from diffusers import CosmosActionCondition  # type: ignore[attr-defined]
        except ImportError as e:
            raise ImportError(_install_hint()) from e
        return CosmosActionCondition

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

    def _build_condition(self, observation: dict[str, Any], **kwargs: Any) -> Any:
        """Build a :class:`CosmosActionCondition` for the active mode."""
        cond_params: dict[str, Any] = {
            "mode": self.mode,
            "chunk_size": self.embodiment.action_chunk_size,
            "domain_name": self.embodiment.domain_name,
            "resolution_tier": self.resolution_tier,
            "view_point": self.view_point,
        }
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
            cond_params["raw_actions"] = self._as_action_tensor(raw_actions)
        else:  # policy
            cond_params["image"] = self._first_frame(observation)
        return self._condition_cls(**cond_params)

    @staticmethod
    def _as_action_tensor(raw_actions: Any) -> Any:
        """Coerce ``raw_actions`` to a ``torch.Tensor[T, D]`` for the condition."""
        try:
            import torch
        except ImportError as e:
            raise ImportError(_install_hint()) from e
        if isinstance(raw_actions, torch.Tensor):
            return raw_actions.to(torch.float32)
        return torch.as_tensor(np.asarray(raw_actions, dtype=np.float32))

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
            ``video`` is the predicted world video (``np.ndarray[T, H, W, C]``,
            ``output_type="np"``); ``sound`` is the decoded waveform or ``None``.
        """
        prompt = observation.get("prompt", "")
        cond = self._build_condition(observation, **kwargs)

        output = self._pipeline(
            prompt=prompt,
            action=cond,
            fps=self.embodiment.fps,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            enable_sound=self.enable_sound,
            output_type="np",
        )
        return {
            "action": _extract_action(getattr(output, "action", None), self.mode),
            "video": getattr(output, "video", None),
            "sound": getattr(output, "sound", None),
        }

    def reset(self) -> None:
        """Per-episode reset hook (no server-side cache to clear in-process)."""
        logger.debug("Cosmos3DiffusersBackend.reset (in-process; no remote state)")


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor / array-like action chunk to ``np.ndarray``.

    Cosmos 3 runs in ``bfloat16`` by default, so the pipeline output action
    tensors are ``bfloat16`` (or ``float16``). ``np.asarray`` cannot read those
    dtypes directly (``TypeError: Got unsupported ScalarType BFloat16``), so we
    up-cast a half-precision tensor to ``float32`` on the torch side before
    handing it to NumPy.
    """
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu()
        try:
            import torch
        except ImportError:
            pass
        else:
            if isinstance(value, torch.Tensor) and value.dtype in (torch.bfloat16, torch.float16):
                value = value.to(torch.float32)
    return np.asarray(value, dtype=np.float32)


def _extract_action(action_field: Any, mode: str) -> np.ndarray | None:
    """Pull the ``[T, D]`` first action chunk out of a pipeline output.

    :class:`Cosmos3OmniPipelineOutput.action` is a ``list[torch.Tensor]`` (one
    ``[T, raw_action_dim]`` chunk) or ``None``. We return the first chunk as
    ``np.ndarray[T, D]``. ``forward_dynamics`` predicts world video only and so
    has no action chunk (returns ``None``).
    """
    if action_field is None or (isinstance(action_field, (list, tuple)) and not action_field):
        if mode == "forward_dynamics":
            return None  # world-only mode produces no action chunk
        raise RuntimeError(
            f"Cosmos 3 diffusers run (mode={mode!r}) returned no action field. "
            "Expected a Cosmos3OmniPipelineOutput with a non-empty 'action' list."
        )
    first = action_field[0] if isinstance(action_field, (list, tuple)) else action_field
    arr = _to_numpy(first)
    # Cosmos may emit [num_chunks, T, D]; take the first chunk -> [T, D].
    if arr.ndim == 3:
        arr = arr[0]
    return arr
