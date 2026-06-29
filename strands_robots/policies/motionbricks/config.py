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

No checkpoints are bundled: ``result_dir`` must point at the upstream ``out/``
checkpoint tree (``motionbricks_pose`` / ``motionbricks_root`` /
``motionbricks_vqvae`` + ``G1-clip.ckpt``), fetched with git-LFS under the
NVIDIA license.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class MotionBricksConfig:
    """Typed configuration for :class:`~strands_robots.policies.motionbricks.policy.MotionBricksPolicy`.

    Attributes:
        result_dir: Path to the upstream ``out/`` checkpoint tree (contains
            ``motionbricks_pose`` / ``motionbricks_root`` / ``motionbricks_vqvae``
            ``version_1/`` dirs + ``G1-clip.ckpt``). Required to build the real
            generator; not validated for existence here (the stub seam builds a
            policy without checkpoints) - existence is checked when the agent is
            constructed.
        skeleton_xml: Path to the G1 skeleton MuJoCo XML (upstream
            ``assets/skeletons/g1/g1.xml``). ``None`` lets the builder derive it
            from the package install.
        scene_xml: Path to the G1 scene MuJoCo XML (upstream
            ``assets/skeletons/g1/scene_29dof.xml``), used for rendering /
            kinematic playback. ``None`` lets the builder derive it.
        clips: Clip set name (upstream ``--clips``; the only shipped set is
            ``"G1"``).
        style: Default motion style - either a clip mode index (``int``) or a
            clip mode name (``str``, e.g. ``"walk"``, ``"stealth_walk"``).
            Overridable per call via the ``style`` / ``mode`` kwarg.
        generate_dt: Synthesis horizon multiplier (upstream ``--generate_dt``).
            Larger values plan further ahead per regeneration.
        fps: Motion frame rate (upstream model fps, 30).
        device: Torch device for the generator (``"cuda"`` or ``"cpu"``).
        speed_scale: ``(min, max)`` root-velocity perturbation range (upstream
            ``--speed_scale``). ``(1.0, 1.0)`` disables perturbation.
        exp: Upstream experiment key selecting the checkpoint layout
            (``"default"``).
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

    def __post_init__(self) -> None:
        # Fail-fast on bad synthesis knobs (AGENTS.md #5: raise on fatal config,
        # never carry a value that will misbehave deep inside the generator).
        if not self.result_dir:
            raise ValueError("MotionBricksConfig.result_dir must be a non-empty path to the 'out/' checkpoint tree")
        if self.fps < 1:
            raise ValueError(f"MotionBricksConfig.fps must be >= 1, got {self.fps}")
        if self.generate_dt <= 0:
            raise ValueError(f"MotionBricksConfig.generate_dt must be > 0, got {self.generate_dt}")
        if not isinstance(self.style, (int, str)):
            raise ValueError(
                f"MotionBricksConfig.style must be an int mode index or a str mode name, got {self.style!r}"
            )
        scale = tuple(self.speed_scale)
        if len(scale) != 2:
            raise ValueError(f"MotionBricksConfig.speed_scale must be a (min, max) pair, got {self.speed_scale!r}")
        lo, hi = float(scale[0]), float(scale[1])
        if lo <= 0 or hi <= 0 or hi < lo:
            raise ValueError(f"MotionBricksConfig.speed_scale must be 0 < min <= max, got ({lo}, {hi})")
        # Normalise speed_scale to a plain float tuple (frozen -> object.__setattr__).
        object.__setattr__(self, "speed_scale", (lo, hi))

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
