"""Qwen-VLA data configuration - typed per-embodiment key mappings + prompt fields.

Mirrors :class:`~strands_robots.policies.groot.data_config.Gr00tDataConfig`
(same ``_extends`` inheritance, JSON-backed registry) but adds the
**embodiment-prompt** fields that are Qwen-VLA's only platform-specific
interface (section 2.3): ``robot_tag``, ``arm_config``, ``has_waist``,
``has_mobile_base``, ``fps``, ``chunk_size``, plus the per-view image tags
(``image_view_tags``) and the path to per-dataset quantile stats used for
normalization (section 5, eq. 5).

Robot configurations live in ``data_configs.json`` alongside this module.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from strands_robots.policies.qwen_vla.prompt import build_embodiment_prompt

logger = logging.getLogger(__name__)


@dataclass
class QwenVlaDataConfig:
    """Typed representation of a Qwen-VLA embodiment data configuration.

    Attributes:
        name: Config identifier (e.g. ``"so100"``).
        video_keys: Camera observation keys (e.g. ``["video.front", "video.wrist"]``).
        state_keys: Robot proprioceptive state keys.
        action_keys: Action output keys the model produces.
        language_keys: Natural-language instruction keys.
        robot_tag: Embodiment-prompt robot identifier (section 2.3).
        arm_config: ``"single"`` or ``"dual"``.
        has_waist: Robot has a controllable waist.
        has_mobile_base: Robot has a mobile base.
        fps: Control frequency in Hz (rendered into the prompt).
        chunk_size: Action horizon H (manipulation H=16, navigation 8).
        image_view_tags: Per-``video_key`` view tag (e.g.
            ``{"video.front": "ego", "video.wrist": "cam_right_wrist"}``).
            These map directly to the paper's camera view tokens
            ``<|tag_start|> <image> <|tag_end|>`` (section 2.2).
        quantile_stats_path: Optional path to per-dataset quantile stats JSON
            for normalization (eq. 5). ``None`` => identity normalization
            (raw actions); explicit so deployments are never silently
            mis-normalized.
        observation_indices: Temporal indices for observations.
        action_indices: Temporal indices for actions (defines H).
    """

    name: str = ""
    video_keys: list[str] = field(default_factory=list)
    state_keys: list[str] = field(default_factory=list)
    action_keys: list[str] = field(default_factory=list)
    language_keys: list[str] = field(default_factory=list)
    robot_tag: str = ""
    arm_config: str = "single"
    has_waist: bool = False
    has_mobile_base: bool = False
    fps: int = 30
    chunk_size: int = 16
    image_view_tags: dict[str, str] = field(default_factory=dict)
    quantile_stats_path: str | None = None
    observation_indices: list[int] = field(default_factory=lambda: [0])
    action_indices: list[int] = field(default_factory=lambda: list(range(16)))
    # Per-family channel widths (bare action key -> width), e.g.
    # {'single_arm': 6, 'gripper': 1}. Lets a single unified Y[H, K] chunk
    # from an upstream server be split EXACTLY across families (paper §2.4);
    # without it a multi-family unified chunk cannot be split safely and the
    # policy raises rather than guessing widths. Empty = unknown widths.
    action_dims: dict[str, int] = field(default_factory=dict)

    def embodiment_prompt(self, instruction: str) -> str:
        """Render this embodiment's section-2.3 prompt for *instruction*.

        Thin wrapper over :func:`build_embodiment_prompt` that wires in the
        config's morphology fields. Keeps prompt construction in one place so
        inference and training stay byte-identical.
        """
        return build_embodiment_prompt(
            robot_tag=self.robot_tag or self.name,
            arm_config=self.arm_config,
            fps=self.fps,
            chunk_size=self.chunk_size,
            instruction=instruction,
            has_waist=self.has_waist,
            has_mobile_base=self.has_mobile_base,
        )


# Config resolution with _extends inheritance (mirrors GR00T)

_SCALAR_INHERIT = (
    "robot_tag",
    "arm_config",
    "has_waist",
    "has_mobile_base",
    "fps",
    "chunk_size",
    "quantile_stats_path",
)


def _resolve_config(name: str, definitions: dict) -> QwenVlaDataConfig:
    """Resolve a config name to a :class:`QwenVlaDataConfig`, following ``_extends``."""
    definition = definitions[name]

    if "_extends" in definition:
        parent = _resolve_config(definition["_extends"], definitions)
        merged: dict = {
            "video_keys": list(parent.video_keys),
            "state_keys": list(parent.state_keys),
            "action_keys": list(parent.action_keys),
            "language_keys": list(parent.language_keys),
            "observation_indices": list(parent.observation_indices),
            "action_indices": list(parent.action_indices),
            "image_view_tags": dict(parent.image_view_tags),
            "action_dims": dict(parent.action_dims),
        }
        for scalar in _SCALAR_INHERIT:
            merged[scalar] = getattr(parent, scalar)
        for field_name, field_value in definition.items():
            if field_name != "_extends":
                merged[field_name] = field_value
    else:
        merged = {field_name: field_value for field_name, field_value in definition.items()}

    merged["name"] = name
    return QwenVlaDataConfig(**merged)


_CONFIG_FILE = Path(__file__).parent / "data_configs.json"


def _load_config_defs() -> tuple:
    """Load config definitions and aliases from the JSON file."""
    with open(_CONFIG_FILE) as fh:
        raw = json.load(fh)
    return raw["configs"], raw.get("aliases", {})


# Pre-resolve all configs at import time
DATA_CONFIG_MAP: dict[str, QwenVlaDataConfig] = {}
_defs, _aliases = _load_config_defs()
for _config_name in _defs:
    DATA_CONFIG_MAP[_config_name] = _resolve_config(_config_name, _defs)
for _alias_name, _target_name in _aliases.items():
    DATA_CONFIG_MAP[_alias_name] = DATA_CONFIG_MAP[_target_name]
del _defs, _aliases


def load_data_config(data_config: str | QwenVlaDataConfig) -> QwenVlaDataConfig:
    """Load a data configuration by name or pass through an existing instance.

    Args:
        data_config: Config name (e.g. ``"so100"``) or a :class:`QwenVlaDataConfig`.

    Returns:
        Resolved :class:`QwenVlaDataConfig`.

    Raises:
        ValueError: If *data_config* is an unknown name or wrong type.
    """
    if isinstance(data_config, QwenVlaDataConfig):
        return data_config
    if isinstance(data_config, str):
        if data_config in DATA_CONFIG_MAP:
            return DATA_CONFIG_MAP[data_config]
        raise ValueError(f"Unknown data_config '{data_config}'. Available: {sorted(DATA_CONFIG_MAP)}")
    raise ValueError(f"data_config must be str or QwenVlaDataConfig, got {type(data_config)}")


def create_custom_data_config(
    name: str,
    *,
    video_keys: list[str],
    state_keys: list[str],
    action_keys: list[str],
    robot_tag: str,
    arm_config: str = "single",
    fps: int = 30,
    chunk_size: int = 16,
    has_waist: bool = False,
    has_mobile_base: bool = False,
    image_view_tags: dict[str, str] | None = None,
    quantile_stats_path: str | None = None,
    language_keys: list[str] | None = None,
    observation_indices: list[int] | None = None,
    action_indices: list[int] | None = None,
    action_dims: dict[str, int] | None = None,
) -> QwenVlaDataConfig:
    """Create and register a custom Qwen-VLA data config at runtime.

    The config is added to :data:`DATA_CONFIG_MAP` so it can be looked up by
    name via :func:`load_data_config`.
    """
    config = QwenVlaDataConfig(
        name=name,
        video_keys=video_keys,
        state_keys=state_keys,
        action_keys=action_keys,
        language_keys=language_keys or ["annotation.human.task_description"],
        robot_tag=robot_tag,
        arm_config=arm_config,
        fps=fps,
        chunk_size=chunk_size,
        has_waist=has_waist,
        has_mobile_base=has_mobile_base,
        image_view_tags=image_view_tags or {},
        quantile_stats_path=quantile_stats_path,
        observation_indices=observation_indices or [0],
        action_indices=action_indices or list(range(chunk_size)),
        action_dims=action_dims or {},
    )
    DATA_CONFIG_MAP[name] = config
    logger.info("Registered custom Qwen-VLA config '%s' (robot_tag=%s)", name, robot_tag)
    return config


__all__ = [
    "QwenVlaDataConfig",
    "DATA_CONFIG_MAP",
    "load_data_config",
    "create_custom_data_config",
]
