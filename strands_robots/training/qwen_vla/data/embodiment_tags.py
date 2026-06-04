"""Embodiment-tag registry - single source of truth for prompt fields.

Maps every Qwen-VLA embodiment to the morphology fields that drive the
section-2.3 prompt. This is the SAME data used by the inference provider's
:class:`~strands_robots.policies.qwen_vla.data_config.QwenVlaDataConfig`, so we
derive directly from that registry rather than duplicating it - inference and
training therefore render byte-identical prompts (the property the paper relies
on for train/eval consistency).

Pure Python - no torch - so it imports without the training extras.
"""

from dataclasses import dataclass

from strands_robots.policies.qwen_vla.data_config import DATA_CONFIG_MAP
from strands_robots.policies.qwen_vla.prompt import build_embodiment_prompt


@dataclass(frozen=True)
class EmbodimentTag:
    """Prompt-relevant morphology for one embodiment (section 2.3).

    Attributes:
        name: Config / embodiment identifier.
        robot_tag: The ``{robot_tag}`` token in the prompt.
        arm_config: ``"single"`` or ``"dual"``.
        has_waist: Robot has a controllable waist.
        has_mobile_base: Robot has a mobile base.
        fps: Control frequency in Hz.
        chunk_size: Action horizon H.
    """

    name: str
    robot_tag: str
    arm_config: str
    has_waist: bool
    has_mobile_base: bool
    fps: int
    chunk_size: int

    def render_prompt(self, instruction: str) -> str:
        """Render the section-2.3 prompt for *instruction* (shared with inference)."""
        return build_embodiment_prompt(
            robot_tag=self.robot_tag,
            arm_config=self.arm_config,
            fps=self.fps,
            chunk_size=self.chunk_size,
            instruction=instruction,
            has_waist=self.has_waist,
            has_mobile_base=self.has_mobile_base,
        )


def _build_registry() -> dict[str, EmbodimentTag]:
    """Derive the embodiment-tag registry from the inference data configs.

    Only canonical (non-alias) configs are included; aliases resolve to the
    same underlying config so they would create duplicate entries.
    """
    seen: dict[int, str] = {}
    registry: dict[str, EmbodimentTag] = {}
    for name, cfg in DATA_CONFIG_MAP.items():
        # Skip aliases: they point at an already-registered config object.
        if id(cfg) in seen:
            continue
        seen[id(cfg)] = name
        registry[name] = EmbodimentTag(
            name=name,
            robot_tag=cfg.robot_tag or name,
            arm_config=cfg.arm_config,
            has_waist=cfg.has_waist,
            has_mobile_base=cfg.has_mobile_base,
            fps=cfg.fps,
            chunk_size=cfg.chunk_size,
        )
    return registry


EMBODIMENT_TAGS: dict[str, EmbodimentTag] = _build_registry()


def get_embodiment_tag(name: str) -> EmbodimentTag:
    """Look up an embodiment tag by config name (or inference alias).

    Args:
        name: Config name (e.g. ``"so100"``) or alias (e.g. ``"aloha"``).

    Returns:
        The :class:`EmbodimentTag`.

    Raises:
        ValueError: If *name* is unknown.
    """
    if name in EMBODIMENT_TAGS:
        return EMBODIMENT_TAGS[name]
    # Resolve through the inference alias map (e.g. "aloha" -> "aloha_bimanual").
    if name in DATA_CONFIG_MAP:
        canonical = DATA_CONFIG_MAP[name].name
        if canonical in EMBODIMENT_TAGS:
            return EMBODIMENT_TAGS[canonical]
    raise ValueError(f"Unknown embodiment '{name}'. Available: {sorted(EMBODIMENT_TAGS)}")


__all__ = ["EmbodimentTag", "EMBODIMENT_TAGS", "get_embodiment_tag"]
