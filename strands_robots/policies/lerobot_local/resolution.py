"""LeRobot policy class resolution.

Resolves the correct LeRobot policy class from:
- HuggingFace Hub config.json (auto-detect)
- Explicit type string (user-specified)

Resolution strategies (in order):
1. PreTrainedConfig draccus resolution (LeRobot 0.5+)
2. Manual config.json reading (fallback for custom/third-party)
3. Direct submodule import: lerobot.policies.{type}.modeling_{type}
4. Package-level import: lerobot.policies.{type}
5. Legacy factory: lerobot.policies.factory.get_policy_class
6. PreTrainedPolicy fallback (only if concrete, not abstract)
"""

import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Tuple, Type

logger = logging.getLogger(__name__)


def resolve_policy_class_from_hub(pretrained_name_or_path: str) -> Tuple[Type, str]:
    """Resolve the LeRobot policy class from a pretrained path or HF repo.

    Uses PreTrainedConfig.from_pretrained() which handles config resolution,
    class lookup, and weight loading via the draccus config registry.

    Falls back to reading config.json manually + class name matching if
    the draccus path fails (e.g. third-party policies not in registry).

    Args:
        pretrained_name_or_path: HF model ID or local directory path.

    Returns:
        Tuple of (PolicyClass, policy_type_string).

    Raises:
        ValueError: If policy type cannot be determined from config.
        ImportError: If the resolved policy class cannot be imported.
    """
    # Strategy 1: PreTrainedConfig draccus resolution → concrete class.
    # We catch broad Exception here because draccus raises its own DecodingError/ParsingError
    # types when a policy type isn't in the choice registry (e.g. older model configs).
    # These are not ImportError/ValueError — they're draccus-specific exceptions.
    try:
        from lerobot.configs.policies import PreTrainedConfig

        config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
        policy_type = getattr(config, "type", type(config).__name__.replace("Config", "").lower())
        logger.info("Auto-resolved via PreTrainedConfig: '%s' -> type='%s'", pretrained_name_or_path, policy_type)

        PolicyClass = resolve_policy_class_by_name(policy_type)
        return PolicyClass, policy_type
    except ImportError:
        raise  # Missing lerobot is a real error, don't swallow
    except Exception as exc:
        logger.debug("PreTrainedConfig resolution failed, trying manual: %s", exc)

    # Strategy 2: Manual config.json reading (fallback for custom/third-party)
    policy_type = _read_policy_type_from_config(pretrained_name_or_path)

    if not policy_type:
        raise ValueError(
            f"Could not determine policy type from '{pretrained_name_or_path}'. "
            f"No 'type' field found in config.json. "
            f"Pass policy_type= explicitly."
        )

    PolicyClass = resolve_policy_class_by_name(policy_type)
    logger.info("Auto-resolved: '%s' -> type='%s' -> %s", pretrained_name_or_path, policy_type, PolicyClass.__name__)
    return PolicyClass, policy_type


def resolve_policy_class_by_name(policy_type: str) -> Type:
    """Resolve policy class from an explicit type string.

    Resolution strategies (in order):
        1. Direct submodule import: lerobot.policies.{type}.modeling_{type}
        2. Package-level import: lerobot.policies.{type}
        3. Legacy factory: lerobot.policies.factory.get_policy_class
        4. PreTrainedPolicy fallback (only if concrete, not abstract)

    LeRobot 0.5+ puts concrete classes in ``modeling_*`` submodules
    (e.g. ``lerobot.policies.act.modeling_act.ACTPolicy``) while the
    package ``__init__`` may re-export only the config.

    Args:
        policy_type: LeRobot policy type string (e.g. "act", "diffusion", "smolvla").

    Returns:
        The resolved policy class.

    Raises:
        ImportError: If no matching class can be found.
    """
    # Strategy 1: modeling_* submodule (LeRobot 0.5+ convention)
    for submodule_name in [f"modeling_{policy_type}", "modeling"]:
        try:
            module = importlib.import_module(f"lerobot.policies.{policy_type}.{submodule_name}")
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (
                    isinstance(obj, type)
                    and attr_name.endswith("Policy")
                    and attr_name != "PreTrainedPolicy"
                    and hasattr(obj, "from_pretrained")
                ):
                    return obj
        except ImportError:
            pass

    # Strategy 2: Direct package-level import
    try:
        module = importlib.import_module(f"lerobot.policies.{policy_type}")
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                isinstance(obj, type)
                and attr_name.endswith("Policy")
                and attr_name != "PreTrainedPolicy"
                and hasattr(obj, "from_pretrained")
            ):
                return obj
    except ImportError:
        pass

    # Strategy 3: Legacy get_policy_class (LeRobot <0.4)
    try:
        from lerobot.policies.factory import get_policy_class

        return get_policy_class(policy_type)
    except (ImportError, AttributeError, RuntimeError):
        pass

    # Strategy 4: PreTrainedPolicy — only if it's NOT abstract
    try:
        from lerobot.policies.pretrained import PreTrainedPolicy

        if not inspect.isabstract(PreTrainedPolicy):
            return PreTrainedPolicy
    except ImportError:
        pass

    raise ImportError(
        f"Could not resolve LeRobot policy class for type '{policy_type}'. "
        f"Tried: lerobot.policies.{policy_type}.modeling_{policy_type}, "
        f"lerobot.policies.{policy_type}, factory, PreTrainedPolicy. "
        f"Ensure lerobot is installed (pip install lerobot)."
    )


def _read_policy_type_from_config(pretrained_name_or_path: str) -> str | None:
    """Read policy type from config.json (local or HF Hub).

    Args:
        pretrained_name_or_path: Local path or HF model ID.

    Returns:
        Policy type string or None if not found.
    """
    local_path = Path(pretrained_name_or_path)
    if local_path.is_dir() and (local_path / "config.json").exists():
        with open(local_path / "config.json") as config_file:
            config = json.load(config_file)
        return config.get("type")

    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(pretrained_name_or_path, "config.json")
        with open(config_path) as config_file:
            config = json.load(config_file)
        return config.get("type")
    except (ImportError, OSError, ValueError, KeyError) as exc:
        logger.warning("Could not download config.json: %s", exc)

    return None


__all__ = ["resolve_policy_class_from_hub", "resolve_policy_class_by_name"]
