"""MotionBricksConfig - configuration for the MotionBricks kinematic motion policy.

MotionBricks (NVlabs/GR00T-WholeBodyControl ``motionbricks/``) is a *generative
kinematic* motion model: given a style (a clip mode) and a movement/facing
command it synthesises per-frame full-body ``qpos`` for the Unitree G1. The
canonical upstream runner is ``motionbricks/scripts/interactive_demo_g1.py``,
which is config-driven through a handful of paths (checkpoint ``result_dir``,
the G1 skeleton/scene XML, the clip set) plus a few synthesis knobs
(``generate_dt``, ``fps``, ``speed_scale``).

This module captures that contract as a frozen :class:`MotionBricksConfig`
dataclass plus loaders that read it from a dict or a JSON file. Keeping it a
typed dataclass (rather than a raw dict) means a bad path or an out-of-range
synthesis knob surfaces at construction with a clear message rather than as an
opaque failure deep inside the generator.

The numeric knobs are held to the shared domains in
:mod:`strands_robots.utils` (:func:`~strands_robots.utils.positive_whole_number_error`
for ``fps``, :func:`~strands_robots.utils.positive_finite_number_error` for
``generate_dt`` and each ``speed_scale`` component) rather than to a local
comparison, because a comparison is not a domain: ``fps < 1`` and
``generate_dt <= 0`` are both ``False`` for ``nan``, and every value they let
through reached :attr:`MotionBricksConfig.controller_dt` and from there the
generator's ``generate_new_frames``.

The three path fields (``result_dir``, ``skeleton_xml``, ``scene_xml``) take a
domain of their own rather than a share of that one, because what they must
refuse is not out of range - it is not a path. Each is held to "a value a path
can be read from" (a ``str`` or an :class:`os.PathLike`) and normalised to the
``str`` the field declares, the same shape as the ``speed_scale`` normalisation
below. The two optional ones additionally keep ``None`` first-class, because
``None`` is how a caller asks the builder to derive the path from the package
install - and it is the *only* way, which is why an empty string is refused
here rather than read as a second spelling of it.

That rule is one module-local function, :func:`_normalised_path_field`, rather
than a shared ``non_empty_string_error`` beside the numeric guards: those have
between 5 and 123 callers in this tree and this has three, all in this file,
since the only other ``if not self.<path field>`` in the library
(:mod:`strands_robots.training._inproc`) is a branch that skips logging rather
than a validation. Lift it when a second config needs it.

The remaining identity fields (``device``, ``clips``, ``exp``) are enumerations,
for which a type check is not the useful rule - it would refuse ``device=5`` and
accept ``clips="g1"`` - and ``style`` is already refused for a bool and
range-checked against the live clip list where the mode is resolved. See #2010.

No checkpoints are bundled: ``result_dir`` must point at the upstream ``out/``
checkpoint tree (``motionbricks_pose`` / ``motionbricks_root`` /
``motionbricks_vqvae`` + ``G1-clip.ckpt``), fetched with git-LFS under the
NVIDIA license.
"""

from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands_robots.utils import (
    positive_finite_number_error,
    positive_whole_number_error,
    sequence_length,
)

# Upstream defaults from ``interactive_demo_g1.py`` + the demo controllers.
# The reference controller regenerates motion every ``NUM_REGEN_FRAMES`` (8)
# frames at ``DEFAULT_FPS`` (30) Hz, scaled by ``generate_dt`` (2.0); the
# controller dt the generator integrates is ``(8 / fps) * generate_dt``.
_DEFAULT_FPS = 30
_DEFAULT_GENERATE_DT = 2.0
_DEFAULT_CLIPS = "G1"
_DEFAULT_EXP = "default"
_DEFAULT_STYLE = "walk"
# Number of frames the reference controller advances before re-querying the
# generator (upstream ``base_controller._CONTROLLER_DT = 8 / FPS``).
NUM_REGEN_FRAMES = 8

# The fields that name a location, with what each names and where an unusable
# value would otherwise have surfaced. ``result_dir`` is required; the other two
# take ``None`` to mean "derive it from the package install".
_PATH_FIELDS: tuple[tuple[str, str, str], ...] = (
    (
        "result_dir",
        "path to the 'out/' checkpoint tree",
        "it is read as Path(result_dir) when the generator is built, which raises TypeError there rather than here.",
    ),
    (
        "skeleton_xml",
        "path to the G1 skeleton MuJoCo XML",
        "it is handed to the upstream generator as skeleton_xml, which reads it as a file there rather than here.",
    ),
    (
        "scene_xml",
        "path to the G1 scene MuJoCo XML",
        "it is read as MjModel.from_xml_path(scene_xml) when the scene is loaded, which raises there rather than here.",
    ),
)


def _normalised_path_field(value: Any, field: str, *, names: str, consequence: str, optional: bool) -> str | None:
    """Return ``value`` as the ``str`` a path field declares, or raise.

    The domain of a field that names a location is path-ness, not truthiness:
    ``if not value`` sorts candidates by a property the field does not have, so
    it accepts ``123`` and ``["out"]`` for being truthy and rejects ``0`` - with
    a message about an empty path, about a number - for being falsy. A
    :class:`~pathlib.Path`, meanwhile, is what a caller is most likely to be
    holding, so the rule normalises rather than merely admitting it: two configs
    naming the same file must compare and hash equal whichever spelling built
    them, and this dataclass is frozen, therefore hashable, so a ``list`` here
    would break ``hash()`` outright.

    Args:
        value: The caller-supplied field value, of any type.
        field: Field name, for the refusal message.
        names: What the path names, e.g. ``"path to the G1 skeleton MuJoCo
            XML"``; read into both refusal messages.
        consequence: Where an unusable value would have failed instead, so the
            refusal says what it saved the caller from.
        optional: When ``True``, ``None`` is returned unchanged - the documented
            way to ask the builder to derive this path.

    Returns:
        ``os.fspath(value)`` for an :class:`os.PathLike`, ``value`` for a
        non-empty ``str``, or ``None`` for an omitted optional field.

    Raises:
        ValueError: If no path can be read from ``value``, or it is empty.
    """
    if optional and value is None:
        return None
    if isinstance(value, os.PathLike):
        # ``os.fspath`` raises when ``__fspath__`` returns a non-path, so leave
        # the value unconverted on that: it then falls to the refusal below,
        # which is the channel a bad value is reported on here.
        with suppress(TypeError):
            value = os.fspath(value)
    if not isinstance(value, str):
        raise ValueError(
            f"MotionBricksConfig.{field} must be a str or os.PathLike {names}, "
            f"got {value!r} ({type(value).__name__}); {consequence}"
        )
    if not value:
        derive = " or None to derive it" if optional else ""
        raise ValueError(f"MotionBricksConfig.{field} must be a non-empty {names}{derive}")
    return value


@dataclass(frozen=True)
class MotionBricksConfig:
    """Typed configuration for :class:`~strands_robots.policies.motionbricks.policy.MotionBricksPolicy`.

    Attributes:
        result_dir: Path to the upstream ``out/`` checkpoint tree (contains
            ``motionbricks_pose`` / ``motionbricks_root`` / ``motionbricks_vqvae``
            ``version_1/`` dirs + ``G1-clip.ckpt``). Required to build the real
            generator. Accepts a ``str`` or any :class:`os.PathLike` - a caller
            holding a :class:`~pathlib.Path` is doing what every consumer of
            this field does - and stores ``os.fspath`` of it, so two configs
            naming the same tree compare and hash equal. A value no path can be
            read from is refused here rather than at ``Path(result_dir)`` in the
            generator build. Existence is still not validated here (the stub
            seam builds a policy without checkpoints) - it is checked when the
            agent is constructed.
        skeleton_xml: Path to the G1 skeleton MuJoCo XML (upstream
            ``assets/skeletons/g1/g1.xml``). ``None`` - and only ``None`` - lets
            the builder derive it from the package install; an empty string is
            refused rather than read as a second spelling of that, because the
            builder cannot tell one from a caller whose path came out empty.
            Otherwise held to the same path domain as ``result_dir``: a ``str``
            or any :class:`os.PathLike`, stored as ``os.fspath`` of it so two
            configs naming the same skeleton compare and hash equal.
        scene_xml: Path to the G1 scene MuJoCo XML (upstream
            ``assets/skeletons/g1/scene_29dof.xml``), used for rendering /
            kinematic playback. ``None`` lets the builder derive it; same path
            domain as ``skeleton_xml``.
        clips: Clip set name (upstream ``--clips``; the only shipped set is
            ``"G1"``).
        style: Default motion style - either a clip mode index (``int``) or a
            clip mode name (``str``, e.g. ``"walk"``, ``"stealth_walk"``).
            Overridable per call via the ``style`` / ``mode`` kwarg.
        generate_dt: Synthesis horizon multiplier (upstream ``--generate_dt``).
            Larger values plan further ahead per regeneration. Must be a
            positive finite number: it is the multiplier of
            :attr:`controller_dt`, so ``0`` gives the generator no time to
            integrate, ``inf`` an unbounded horizon and ``nan`` a horizon that
            poisons every frame synthesised from it.
        fps: Motion frame rate (upstream model fps, 30). Must be a positive
            whole number - it is the divisor of :attr:`controller_dt`, so a
            fractional rate is not a frame count the generator can emit, and
            ``inf`` collapses the horizon to ``0`` rather than making it small.
        device: Torch device for the generator (``"cuda"`` or ``"cpu"``).
        speed_scale: ``(min, max)`` root-velocity perturbation range (upstream
            ``--speed_scale``). ``(1.0, 1.0)`` disables perturbation. Both
            components must be positive finite numbers with ``min <= max``;
            they multiply the synthesised root velocity, so a non-finite
            component scales it to a velocity no integrator can consume.
        exp: Upstream experiment key selecting the checkpoint layout
            (``"default"``).
        style_map: Optional overrides merged over the built-in
            ``locomotion_style`` -> clip-name map
            (:data:`~strands_robots.policies.motionbricks.observation.LOCOMOTION_STYLE_TO_G1_CLIP`),
            used to translate a caller-supplied ``locomotion_style`` goal kwarg
            to this generator's clip set. ``None`` uses the defaults.
    """

    result_dir: str
    skeleton_xml: str | None = None
    scene_xml: str | None = None
    clips: str = _DEFAULT_CLIPS
    style: int | str = _DEFAULT_STYLE
    generate_dt: float = _DEFAULT_GENERATE_DT
    fps: int = _DEFAULT_FPS
    device: str = "cuda"
    speed_scale: tuple[float, float] = (1.0, 1.0)
    exp: str = _DEFAULT_EXP
    style_map: dict[str, str] | None = None

    def __post_init__(self) -> None:
        # Fail-fast on bad synthesis knobs (AGENTS.md #5: raise on fatal config,
        # never carry a value that will misbehave deep inside the generator).
        # Every field that names a location goes through the same rule, because
        # its domain is path-ness: ``skeleton_xml`` and ``scene_xml`` had no
        # check at all, so a ``Path`` was stored unnormalised (a config built
        # from one compared unequal to the identical config built from a
        # ``str``), a ``list`` left this frozen dataclass unhashable, and ``123``
        # travelled to the generator to fail there. Worse than the truthiness
        # test ``result_dir`` used to carry: a falsy value was not refused at
        # all, and the builder's derived default silently took its place.
        for field, names, consequence in _PATH_FIELDS:
            object.__setattr__(
                self,
                field,
                _normalised_path_field(
                    getattr(self, field),
                    field,
                    names=names,
                    consequence=consequence,
                    optional=field != "result_dir",
                ),
            )
        # The synthesis knobs go through the shared numeric domains rather than a
        # local comparison, because a comparison is not a domain: ``fps < 1`` is
        # ``False`` for ``nan`` and for ``inf``, and ``True`` is an ``int``
        # subclass that reads as a 1 Hz frame rate. Each of those reached
        # ``controller_dt`` and, from there, ``generate_new_frames``.
        if error := positive_whole_number_error(self.fps, "fps", "MotionBricksConfig"):
            raise ValueError(error)
        if error := positive_finite_number_error(self.generate_dt, "generate_dt", "MotionBricksConfig"):
            raise ValueError(error)
        if not isinstance(self.style, (int, str)):
            raise ValueError(
                f"MotionBricksConfig.style must be an int mode index or a str mode name, got {self.style!r}"
            )
        # Arity is read with ``sequence_length`` rather than ``len(tuple(...))``
        # so a scalar is refused by this message instead of raising ``'float'
        # object is not iterable`` from the arity check itself.
        if sequence_length(self.speed_scale) != 2:
            raise ValueError(f"MotionBricksConfig.speed_scale must be a (min, max) pair, got {self.speed_scale!r}")
        lo_raw, hi_raw = tuple(self.speed_scale)
        for index, component in ((0, lo_raw), (1, hi_raw)):
            # Before the ``float()`` below, not after: that conversion is what
            # used to launder ``speed_scale=("1", "2")`` into ``(1.0, 2.0)``.
            if error := positive_finite_number_error(component, f"speed_scale[{index}]", "MotionBricksConfig"):
                raise ValueError(error)
        lo, hi = float(lo_raw), float(hi_raw)
        if hi < lo:
            raise ValueError(f"MotionBricksConfig.speed_scale must be 0 < min <= max, got ({lo}, {hi})")
        # Normalise speed_scale to a plain float tuple (frozen -> object.__setattr__).
        object.__setattr__(self, "speed_scale", (lo, hi))
        if self.style_map is not None and (
            not isinstance(self.style_map, dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in self.style_map.items())
        ):
            raise ValueError(
                "MotionBricksConfig.style_map must be a dict[str, str] mapping locomotion-style "
                f"name -> clip-mode name, got {self.style_map!r}"
            )

    @property
    def controller_dt(self) -> float:
        """Per-regeneration integration horizon the generator consumes.

        Mirrors the upstream controller's ``get_controller_dt() * generate_dt``
        (``(NUM_REGEN_FRAMES / fps) * generate_dt``).
        """
        return (NUM_REGEN_FRAMES / float(self.fps)) * float(self.generate_dt)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MotionBricksConfig:
        """Build a :class:`MotionBricksConfig` from a plain dict.

        Only recognised keys are consumed; unknown keys are ignored (forward
        compatibility). ``result_dir`` is required.
        """
        if "result_dir" not in data:
            raise ValueError("MotionBricksConfig requires a 'result_dir' entry")
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        if "speed_scale" in kwargs and kwargs["speed_scale"] is not None:
            kwargs["speed_scale"] = tuple(kwargs["speed_scale"])
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str | Path) -> MotionBricksConfig:
        """Load a :class:`MotionBricksConfig` from a JSON file.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file is not valid JSON, is not a mapping, or is
                missing ``result_dir``.
        """
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"MotionBricksConfig file not found: {p}")
        suffix = p.suffix.lower()
        if suffix != ".json":
            raise ValueError(f"MotionBricksConfig file {p} has unsupported extension {suffix!r}; use .json.")
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"MotionBricksConfig file {p} is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(f"MotionBricksConfig file {p} must contain a mapping, got {type(data).__name__}")
        return cls.from_dict(data)


__all__ = ["MotionBricksConfig", "NUM_REGEN_FRAMES"]
