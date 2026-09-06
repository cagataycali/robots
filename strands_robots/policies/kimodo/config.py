"""KimodoConfig - configuration for the Kimodo text-to-motion diffusion policy.

Kimodo (``nvidia/Kimodo-G1-RP-v1``) is a text-conditioned kinematic motion
diffusion model: prompt in, per-frame G1 ``qpos`` out. This module captures the
knobs a user cares about (prompt, sampling steps, guidance, FPS, and the HF
model id) as a frozen :class:`KimodoConfig` dataclass so a bad value surfaces
at construction time with a clear message rather than deep inside the sampler.

Numeric domains follow the shared helpers in :mod:`strands_robots.utils`:

* ``diffusion_steps`` - positive integer (Kimodo default 100, 25-200 useful range)
* ``guidance_scale`` - positive finite number (Kimodo default 7.5)
* ``fps`` - positive integer (Kimodo emits at 30Hz native, upsampled to 50Hz
  for the G1 tracker via SLERP downstream)
* ``num_frames`` - positive integer bounded by the model's max sequence length
  (196 for RP-v1)
* ``transition_frames`` - positive integer, at least 1, matching the domain the
  Kimodo sampler applies to its own ``num_transition_frames``
* ``seed`` - whole number or ``None``, via :func:`sampling_seed_error`, which is
  also the domain :meth:`KimodoPolicy.reset` applies to a per-episode reseed and
  :meth:`KimodoPolicy.get_actions` to a per-call override

The default ``model_id`` targets the RP-v1 checkpoint. Alternate model ids are
accepted verbatim; the loader defers validation to ``from_pretrained``. Note
that ``nvidia/Kimodo-G1-RP-v1`` publishes bare weights rather than a diffusers
pipeline, so a real-model run supplies its sampler through ``motion_agent=``.
"""

from __future__ import annotations

import json
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands_robots.utils import positive_finite_number_error, positive_whole_number_error

_KIMODO_DEFAULT_MODEL_ID = "nvidia/Kimodo-G1-RP-v1"
_KIMODO_MAX_FRAMES = 196
_KIMODO_NATIVE_FPS = 30
_KIMODO_TRACKER_FPS = 50
# Kimodo's own sampler blends a multi-prompt sequence over its last
# ``num_transition_frames`` frames and defaults that to 5 (and refuses < 1), so
# a chained rollout here adopts the same length and domain rather than inventing
# a second convention for the same transition.
_KIMODO_TRANSITION_FRAMES = 5


def _positive_int(name: str, value: int, upper: int | None = None) -> int:
    """Validate a positive whole knob through the shared domain, plus an upper bound.

    Delegates the sign, integrality, ``bool`` and float-range rules to
    :func:`~strands_robots.utils.positive_whole_number_error` so this provider
    cannot drift from every other counted knob in the tree; only the
    provider-specific ceiling is applied here.

    Args:
        name: Parameter name, used in the message.
        value: The caller-supplied value.
        upper: Inclusive ceiling, when the field has one.

    Returns:
        The validated value.

    Raises:
        ValueError: If the value is outside the domain or above ``upper``.
    """
    if error := positive_whole_number_error(value, name, "KimodoConfig"):
        raise ValueError(error)
    if upper is not None and value > upper:
        raise ValueError(f"KimodoConfig: {name} must be <= {upper}, got {value}.")
    return int(value)


def _positive_float(name: str, value: float) -> float:
    """Validate a positive continuous knob through the shared domain.

    Delegates to :func:`~strands_robots.utils.positive_finite_number_error`,
    which rejects ``bool``, ``nan``, ``inf`` and values past the float64 range -
    the cases a hand-rolled ``value != value`` check reads as a NaN test but
    states less precisely.

    Args:
        name: Parameter name, used in the message.
        value: The caller-supplied value.

    Returns:
        The validated value as a float.

    Raises:
        ValueError: If the value is not a positive finite number.
    """
    if error := positive_finite_number_error(value, name, "KimodoConfig"):
        raise ValueError(error)
    return float(value)


def sampling_seed_error(value: Any, context: str) -> str | None:
    """Return why a value cannot seed a Kimodo sampler run.

    The single owner of this domain. Three surfaces set the sampling seed -
    :class:`KimodoConfig` (directly, through :meth:`KimodoConfig.from_dict` /
    :meth:`KimodoConfig.from_json`, or through a ``KimodoPolicy(seed=...)``
    override); :meth:`KimodoPolicy.reset`, which stores a per-episode reseed
    on the frozen config with ``object.__setattr__`` and so does not re-enter
    :meth:`__post_init__`; and :meth:`KimodoPolicy.get_actions`, whose
    documented per-call ``seed=`` override is read straight from ``kwargs`` and
    reaches both the sampler and the buffered-motion key without passing
    through the config at all. All three consult this function, so one value
    gets one verdict whichever way it is spelled.

    A seed has to survive being used twice, and that is what the domain here
    protects. It is handed to the sampler, and it is part of the key
    ``KimodoPolicy`` identifies the buffered motion by - a key built by
    coercing the seed with ``int()``. A seed that is not already whole
    therefore names a different sample than the one it produces: ``2.5`` and
    ``2.9`` reach the sampler as themselves and key as ``2``, so a reseeded
    episode reads as a cache hit and silently replays the previous episode's
    motion rather than sampling its own. ``nan`` and ``inf`` do not survive
    the coercion at all and raised out of the key builder - past the
    construction-time reporting this module exists to give, and on a rollout
    that had already reported the reseed as applied. ``inf`` is reachable from
    a JSON config file: ``1e400`` is well-formed JSON and parses to it.

    ``bool`` is rejected explicitly. It is an ``int`` subclass, so ``True``
    would pass an integrality test and then key as ``1``, silently sharing a
    motion with ``seed=1``.

    Sign is not part of this domain. Kimodo's seed is applied with
    ``torch.manual_seed`` (or a ``Generator``'s), which honors a negative seed,
    and the key holds it unchanged - so a negative seed round-trips, and
    refusing it would reject a value that works. The rollout facades'
    :func:`~strands_robots.simulation.base.randomization_seed_error` is
    narrower for the reason it states itself: it also reseeds ``numpy.random``,
    which refuses a negative seed and a value above ``2**32 - 1``. Their
    appliers are not equally wide, so their domains are not either.

    Magnitude is not part of it either, and the distinction is where the
    failure surfaces. Both torch appliers accept ``[-2**63, 2**64-1]`` and
    refuse a wider seed with a ``ValueError`` naming the overflow, at the call
    that applies it - so an outsized seed is already reported, by the applier
    that owns the bound, and the sampler is a pluggable ``motion_agent=`` whose
    bound is not this module's to state. What this function refuses is the
    complementary set: seeds that fail somewhere nobody is looking - inside a
    private key builder, with no mention of a seed - or that do not fail at
    all and simply name the wrong sample.

    Args:
        value: The candidate seed (``None`` selects fresh entropy per call).
        context: Surface that received it, used to prefix the message - the
            class name for a constructor parameter, the method name otherwise.

    Returns:
        ``None`` when the seed is usable, otherwise the reason as a string.
    """
    if value is None:
        return None
    prefix = f"{context}: seed must be a whole number or None, got {value!r}"
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return f"{prefix} (None draws fresh entropy for every sample)."
    try:
        whole = int(value)
    except (OverflowError, TypeError, ValueError):
        # nan and inf reach here. They cannot key a sample at all, so they
        # would raise out of the key builder mid-rollout instead of here.
        return f"{prefix}: it cannot be coerced to the whole number the buffered-motion key is built from."
    if whole != value:
        return (
            f"{prefix}: the buffered-motion key rounds a seed to a whole number, so "
            f"{value!r} would name the sample keyed by {whole!r} and replay it instead of sampling its own."
        )
    return None


@dataclass(frozen=True)
class KimodoConfig:
    """Frozen configuration for :class:`KimodoPolicy`.

    Attributes:
        model_id: HuggingFace model id. Defaults to ``nvidia/Kimodo-G1-RP-v1``.
        diffusion_steps: Number of denoising steps (25-200, default 100).
        guidance_scale: Classifier-free-guidance weight (default 7.5).
        num_frames: Motion length in Kimodo native frames (30 FPS, max 196).
        transition_frames: Number of native frames over which a newly sampled
            motion is eased off the pose last commanded, when a prompt (or any
            other sampler input) changes mid-rollout. A fresh sample begins at
            its own canonical start pose, which is unrelated to wherever the
            previous motion left the robot, so emitting it directly steps every
            joint at once. Defaults to 5, the same length Kimodo's sampler uses
            for its own ``num_transition_frames``. Only the first sample of a
            rollout is unaffected: with no previously commanded pose there is no
            seam to ease.
        native_fps: Kimodo sampler output rate (30 Hz native, do not change
            unless targeting a retrained checkpoint).
        tracker_fps: Rate the G1 tracker consumes at (50 Hz standard). Frames
            are SLERP-interpolated from ``native_fps`` to ``tracker_fps`` by
            :class:`KimodoPolicy`.
        device: torch device string (``"cuda"``, ``"cuda:0"``, ``"cpu"``).
            ``None`` means auto-select CUDA if available.
        dtype: Sampler dtype string. ``"fp16"`` recommended on GPU for speed;
            ``"fp32"`` for reproducibility.
        cache_dir: Optional local HF cache override; ``None`` uses
            ``$HF_HOME``.
        trust_remote_code: Whether ``from_pretrained`` may execute code that
            ships inside the checkpoint repository. Defaults to ``False``, so
            loading a checkpoint does not run its code unless the caller asks
            for it. ``STRANDS_TRUST_REMOTE_CODE`` is a separate and coarser
            gate: it decides whether this provider may be constructed at all
            and never sets this field, so opting in to the provider does not
            also opt in to executing a repository's code. Only a ``model_id``
            published in diffusers pipeline layout reaches the flag at all -
            NVIDIA's own Kimodo checkpoints are not, and are refused at load
            time in favour of a ``motion_agent=`` sampler.
        seed: RNG seed for reproducible sampling. A whole number, of either
            sign and any width, or ``None`` for fresh entropy on every sample.
            :func:`sampling_seed_error` owns the domain and
            :meth:`KimodoPolicy.reset` applies the same one to a per-episode
            reseed.
    """

    model_id: str = _KIMODO_DEFAULT_MODEL_ID
    diffusion_steps: int = 100
    guidance_scale: float = 7.5
    num_frames: int = 120
    transition_frames: int = _KIMODO_TRANSITION_FRAMES
    native_fps: int = _KIMODO_NATIVE_FPS
    tracker_fps: int = _KIMODO_TRACKER_FPS
    device: str | None = None
    dtype: str = "fp16"
    cache_dir: str | None = None
    trust_remote_code: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        # Validate at construction so bad values fail loud.
        _positive_int("diffusion_steps", self.diffusion_steps, upper=500)
        _positive_float("guidance_scale", self.guidance_scale)
        _positive_int("num_frames", self.num_frames, upper=_KIMODO_MAX_FRAMES)
        _positive_int("transition_frames", self.transition_frames, upper=_KIMODO_MAX_FRAMES)
        _positive_int("native_fps", self.native_fps)
        _positive_int("tracker_fps", self.tracker_fps)
        if self.dtype not in ("fp16", "bf16", "fp32"):
            raise ValueError(f"dtype must be one of 'fp16'/'bf16'/'fp32', got {self.dtype!r}")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if error := sampling_seed_error(self.seed, "KimodoConfig"):
            raise ValueError(error)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KimodoConfig:
        """Build a config from a plain dict.

        Only recognised keys are consumed; unknown keys are ignored for forward
        compatibility, and no warning is emitted for a dropped key - the policy
        :mod:`strands_robots.policies.motionbricks.config` and
        :mod:`strands_robots.policies.wbc.config` state for their own
        ``from_dict``.

        Args:
            data: Mapping of field name to value. Keys that are not fields of
                this class are dropped.

        Returns:
            The config, validated by :meth:`__post_init__`.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str | Path) -> KimodoConfig:
        """Load a config from a JSON file on disk.

        A file that cannot supply fields is reported by name rather than
        reaching :meth:`from_dict`, which is the reporting the sibling
        policy-config file loaders in
        :mod:`strands_robots.policies.motionbricks.config` and
        :mod:`strands_robots.policies.wbc.config` already give. ``~`` in
        ``path`` is expanded.

        The extension is deliberately not checked, unlike those two loaders: a
        JSON object stored under any name loads here today, and refusing one
        would stop a payload that currently works. Every refusal below names an
        input that already fails.

        Args:
            path: Path to a file holding a JSON object of config fields.

        Returns:
            The config, validated by :meth:`__post_init__`.

        Raises:
            FileNotFoundError: If ``path`` does not name a file.
            ValueError: If the file is not valid JSON, or holds a JSON value
                that is not an object. A value inside the object keeps the
                domain :meth:`__post_init__` applies to it, so the loader adds
                no second domain of its own.
        """
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"KimodoConfig file not found: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"KimodoConfig file {p} is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(f"KimodoConfig file {p} must contain a mapping, got {type(data).__name__}")
        return cls.from_dict(data)
