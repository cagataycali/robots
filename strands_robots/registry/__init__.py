"""Unified Registry — single source of truth for robots and policies.

Loads robot definitions and policy provider configs from JSON files.

Features:
    - **One file to edit**: Add a robot → edit robots.json, done.
    - **Hot-reload**: JSON is re-read when the file changes (mtime check).
    - **Self-contained entries**: Each robot/policy owns its aliases,
      shorthands, and URL patterns — no separate lookup tables.

Usage::

    from strands_robots.registry import get_robot, resolve_name, list_robots
    from strands_robots.registry import get_policy_provider, resolve_policy_string

    info = get_robot("so100")
    name = resolve_name("franka")       # → "panda"
    providers = list_policy_providers()

Architecture:
    registry/
        __init__.py      ← this file (loader + public API)
        robots.json      ← robot definitions (aliases inside each entry)
        policies.json    ← policy providers (shorthands/urls inside each entry)
"""

import importlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# JSON loading with mtime-based hot-reload
# ─────────────────────────────────────────────────────────────────────

_REGISTRY_DIR = Path(__file__).parent
_cache: Dict[str, dict] = {}
_mtimes: Dict[str, float] = {}


def _load(name: str) -> dict:
    """Load a JSON registry file, re-reading only when mtime changes.

    Args:
        name: Base name without extension (e.g. "robots", "policies").

    Returns:
        Parsed JSON as a dict.
    """
    path = _REGISTRY_DIR / f"{name}.json"
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        logger.error("Registry file not found: %s", path)
        return {}

    if name not in _cache or _mtimes.get(name) != mtime:
        with open(path, encoding="utf-8") as f:
            _cache[name] = json.load(f)
        _mtimes[name] = mtime
        logger.debug("Loaded registry: %s (%d bytes)", path, path.stat().st_size)

    return _cache[name]


def reload():
    """Force-reload all registry files (clears mtime cache)."""
    _cache.clear()
    _mtimes.clear()


# ─────────────────────────────────────────────────────────────────────
# Robot Registry
# ─────────────────────────────────────────────────────────────────────

def _build_robot_alias_map() -> Dict[str, str]:
    """Build alias → canonical name mapping from robot entries.

    Each robot entry may have an ``aliases`` list. This function
    inverts those into a flat lookup dict.
    """
    reg = _load("robots")
    alias_map: Dict[str, str] = {}
    for name, info in reg.get("robots", {}).items():
        for alias in info.get("aliases", []):
            alias_map[alias] = name
    return alias_map


def resolve_name(name: str) -> str:
    """Resolve a robot name or alias to the canonical name.

    Args:
        name: Any robot name, alias, or data_config string.

    Returns:
        Canonical robot name (e.g. "so100", "panda", "unitree_g1").

    Examples::

        resolve_name("franka")        # → "panda"
        resolve_name("SO100_follower") # → "so100"
        resolve_name("g1")            # → "unitree_g1"
    """
    normalized = name.lower().strip().replace("-", "_")
    alias_map = _build_robot_alias_map()
    return alias_map.get(normalized, normalized)


def get_robot(name: str) -> Optional[Dict[str, Any]]:
    """Get full robot definition by name or alias.

    Args:
        name: Robot name, alias, or data_config.

    Returns:
        Robot dict with keys like description, category, joints, asset,
        hardware — or None if not found.
    """
    reg = _load("robots")
    canonical = resolve_name(name)
    return reg.get("robots", {}).get(canonical)


def has_sim(name: str) -> bool:
    """Check if a robot has simulation assets (MJCF/URDF)."""
    info = get_robot(name)
    return info is not None and "asset" in info


def has_hardware(name: str) -> bool:
    """Check if a robot has real hardware support (LeRobot type)."""
    info = get_robot(name)
    return info is not None and "hardware" in info


def get_hardware_type(name: str) -> Optional[str]:
    """Get the LeRobot hardware type for a robot.

    Returns:
        LeRobot type string (e.g. "so100_follower"), or None.
    """
    info = get_robot(name)
    if info and "hardware" in info:
        return info["hardware"].get("lerobot_type")
    return None


def list_robots(mode: str = "all") -> List[Dict[str, Any]]:
    """List available robots, optionally filtered.

    Args:
        mode: Filter — "all", "sim", "real", or "both" (has sim AND real).

    Returns:
        List of dicts with name, description, has_sim, has_real.
    """
    reg = _load("robots")
    results = []
    for name, info in sorted(reg.get("robots", {}).items()):
        _has_sim = "asset" in info
        _has_real = "hardware" in info

        if mode == "sim" and not _has_sim:
            continue
        if mode == "real" and not _has_real:
            continue
        if mode == "both" and not (_has_sim and _has_real):
            continue

        results.append({
            "name": name,
            "description": info.get("description", ""),
            "category": info.get("category", ""),
            "joints": info.get("joints"),
            "has_sim": _has_sim,
            "has_real": _has_real,
        })
    return results


def list_robots_by_category() -> Dict[str, List[Dict[str, Any]]]:
    """List robots grouped by category (arm, humanoid, mobile, ...)."""
    categories: Dict[str, list] = {}
    for robot in list_robots():
        cat = robot.get("category", "other")
        categories.setdefault(cat, []).append(robot)
    return categories


def list_aliases() -> Dict[str, str]:
    """Return the full alias → canonical mapping."""
    return _build_robot_alias_map()


def format_robot_table() -> str:
    """Human-readable table of all robots for CLI/tool output."""
    lines = [
        f"{'Name':<20} {'Category':<15} {'Joints':<8} {'Sim':<5} {'Real':<5} Description",
        "─" * 100,
    ]
    for cat in ["arm", "bimanual", "hand", "humanoid", "expressive", "mobile", "mobile_manip"]:
        by_cat = list_robots_by_category()
        for r in by_cat.get(cat, []):
            sim = "✅" if r["has_sim"] else "  "
            real = "✅" if r["has_real"] else "  "
            joints = str(r["joints"]) if r["joints"] else "?"
            lines.append(
                f"{r['name']:<20} {r['category']:<15} {joints:<8} {sim:<5} {real:<5} {r['description']}"
            )

    robots = list_robots()
    lines.append("")
    lines.append(f"Total: {len(robots)} robots | Aliases: {len(list_aliases())}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Policy Registry
# ─────────────────────────────────────────────────────────────────────

def _build_policy_alias_map() -> Dict[str, str]:
    """Build alias/shorthand → canonical provider mapping from provider entries."""
    reg = _load("policies")
    alias_map: Dict[str, str] = {}
    for name, info in reg.get("providers", {}).items():
        for alias in info.get("aliases", []):
            alias_map[alias] = name
        for shorthand in info.get("shorthands", []):
            alias_map[shorthand] = name
    return alias_map


def get_policy_provider(name: str) -> Optional[Dict[str, Any]]:
    """Get policy provider config by name or alias.

    Args:
        name: Provider name or alias (e.g. "groot", "lerobot", "cosmos").

    Returns:
        Provider dict with module, class, config_keys, defaults, etc.
        None if not found.
    """
    reg = _load("policies")
    alias_map = _build_policy_alias_map()
    canonical = alias_map.get(name, name)
    return reg.get("providers", {}).get(canonical)


def list_policy_providers() -> List[str]:
    """List all registered policy provider names (canonical only)."""
    reg = _load("policies")
    return sorted(reg.get("providers", {}).keys())


def resolve_policy_string(
    policy: str, **extra_kwargs
) -> Tuple[str, Dict[str, Any]]:
    """Resolve a smart policy string to (provider_name, kwargs).

    Accepts HuggingFace model IDs, server URLs, or shorthand names
    and returns the canonical provider + ready-to-use kwargs.

    Resolution order:
        1. URL patterns (ws://, zmq://, grpc://, host:port)
        2. Shorthand names (mock, groot, dreamgen, ...)
        3. HuggingFace model IDs (org/model)
        4. Registered provider name
        5. Fallback to lerobot_local

    Args:
        policy: Smart string — HF model ID, URL, or provider name.
        **extra_kwargs: Additional kwargs merged into result.

    Returns:
        (provider_name, kwargs_dict) tuple.

    Examples::

        resolve_policy_string("lerobot/act_aloha_sim")
        # → ("lerobot_local", {"pretrained_name_or_path": "lerobot/act_aloha_sim"})

        resolve_policy_string("localhost:8080")
        # → ("lerobot_async", {"server_address": "localhost:8080"})

        resolve_policy_string("mock")
        # → ("mock", {})
    """
    reg = _load("policies")
    providers = reg.get("providers", {})
    policy = policy.strip()
    kwargs: Dict[str, Any] = {}

    # 1. URL pattern matching — check each provider's url_patterns
    for prov_name, prov_info in providers.items():
        for pattern in prov_info.get("url_patterns", []):
            if re.match(pattern, policy):
                if pattern.startswith("^wss?://"):
                    match = re.match(r"wss?://([^:]+):?(\d+)?", policy)
                    if match:
                        kwargs["host"] = match.group(1)
                        kwargs["port"] = int(match.group(2) or 8000)
                elif pattern.startswith("^zmq://"):
                    match = re.match(r"zmq://([^:]+):(\d+)", policy)
                    if match:
                        kwargs["host"] = match.group(1)
                        kwargs["port"] = int(match.group(2))
                elif pattern.startswith("^grpc://"):
                    kwargs["server_address"] = policy.replace("grpc://", "")
                elif ":" in policy and "/" not in policy:
                    kwargs["server_address"] = policy
                kwargs.update(extra_kwargs)
                return prov_name, kwargs

    # 2. Shorthand names — built from each provider's shorthands list
    alias_map = _build_policy_alias_map()
    if policy.lower() in alias_map:
        kwargs.update(extra_kwargs)
        return alias_map[policy.lower()], kwargs

    # 3. HuggingFace model IDs (org/model)
    if "/" in policy:
        # Check model_id_overrides across all providers
        for prov_name, prov_info in providers.items():
            for prefix in prov_info.get("model_id_overrides", []):
                if policy.lower().startswith(prefix):
                    kwargs["pretrained_name_or_path"] = policy
                    kwargs.update(extra_kwargs)
                    return prov_name, kwargs

        # Check hf_orgs
        org = policy.split("/")[0].lower()
        for prov_name, prov_info in providers.items():
            if org in prov_info.get("hf_orgs", []):
                kwargs["pretrained_name_or_path"] = policy
                kwargs.update(extra_kwargs)
                return prov_name, kwargs

        # Unknown org → find default HF provider
        for prov_name, prov_info in providers.items():
            if prov_info.get("is_hf_default"):
                kwargs["pretrained_name_or_path"] = policy
                kwargs.update(extra_kwargs)
                return prov_name, kwargs

        # Absolute fallback
        kwargs["pretrained_name_or_path"] = policy
        kwargs.update(extra_kwargs)
        return "lerobot_local", kwargs

    # 4. Check if it's a registered provider name
    if get_policy_provider(policy.lower()):
        kwargs.update(extra_kwargs)
        return policy.lower(), kwargs

    # 5. Fallback
    logger.warning("Unrecognised policy '%s', falling back to lerobot_local", policy)
    kwargs["pretrained_name_or_path"] = policy
    kwargs.update(extra_kwargs)
    return "lerobot_local", kwargs


def import_policy_class(provider: str) -> Type:
    """Dynamically import and return the Policy class for a provider.

    Uses the module + class paths from policies.json. Falls back to
    auto-discovery (strands_robots.policies.<name>) if not in JSON.

    Args:
        provider: Canonical provider name.

    Returns:
        The Policy subclass.

    Raises:
        ValueError: If provider not found.
        ImportError: If the module can't be imported.
    """
    config = get_policy_provider(provider)
    if config:
        # Resolve alias to canonical for module lookup
        reg = _load("policies")
        alias_map = _build_policy_alias_map()
        canonical = alias_map.get(provider, provider)
        config = reg.get("providers", {}).get(canonical, config)

        mod = importlib.import_module(config["module"])
        return getattr(mod, config["class"])

    # Auto-discovery fallback
    try:
        mod = importlib.import_module(f"strands_robots.policies.{provider}")
        class_name = f"{provider.capitalize()}Policy"
        if hasattr(mod, class_name):
            return getattr(mod, class_name)
        from strands_robots.policies import Policy
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and issubclass(attr, Policy) and attr is not Policy:
                return attr
    except ImportError:
        pass

    raise ValueError(
        f"Unknown policy provider: '{provider}'. "
        f"Available: {list_policy_providers()}"
    )


def build_policy_kwargs(
    provider: str,
    policy_port: Optional[int] = None,
    policy_host: str = "localhost",
    model_path: Optional[str] = None,
    server_address: Optional[str] = None,
    policy_type: Optional[str] = None,
    data_config: Any = None,
    **extra,
) -> Dict[str, Any]:
    """Build provider-specific kwargs from generic parameters.

    Maps generic parameter names (policy_port, model_path, ...) to
    the provider-specific keys declared in policies.json.

    Args:
        provider: Policy provider name.
        policy_port: Port number (groot, lerobot_async).
        policy_host: Hostname (default: "localhost").
        model_path: Local model path or HF ID.
        server_address: Full gRPC address (lerobot_async).
        policy_type: Sub-type (pi0, act, smolvla, ...).
        data_config: Data configuration for groot.
        **extra: Any additional provider-specific kwargs.

    Returns:
        Dict of kwargs ready for ``create_policy(provider, **kwargs)``.
    """
    config = get_policy_provider(provider) or {}
    allowed_keys = set(config.get("config_keys", []))
    defaults = dict(config.get("defaults", {}))
    kwargs: Dict[str, Any] = {}

    param_map = {
        "port": policy_port,
        "host": policy_host,
        "data_config": data_config,
        "server_address": server_address or (
            f"{policy_host}:{policy_port}" if policy_port and "server_address" in allowed_keys else None
        ),
        "model_path": model_path,
        "pretrained_name_or_path": model_path if model_path and "pretrained_name_or_path" in allowed_keys else extra.get("pretrained_name_or_path"),
        "policy_type": policy_type,
    }

    for key, value in param_map.items():
        if value is not None and key in allowed_keys:
            kwargs[key] = value

    for key, default_val in defaults.items():
        if key not in kwargs:
            kwargs[key] = default_val

    for key, value in extra.items():
        if key in allowed_keys and key not in kwargs:
            kwargs[key] = value

    return kwargs


# ─────────────────────────────────────────────────────────────────────
# Public API exports
# ─────────────────────────────────────────────────────────────────────

__all__ = [
    # Robot registry
    "resolve_name",
    "get_robot",
    "has_sim",
    "has_hardware",
    "get_hardware_type",
    "list_robots",
    "list_robots_by_category",
    "list_aliases",
    "format_robot_table",
    # Policy registry
    "get_policy_provider",
    "list_policy_providers",
    "resolve_policy_string",
    "import_policy_class",
    "build_policy_kwargs",
    # Utilities
    "reload",
]
