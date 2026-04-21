"""Shared utilities for strands-robots."""

import importlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache of lazy-loaded modules
_lazy_modules: dict[str, object] = {}


def require_optional(
    module_name: str,
    *,
    pip_install: str | None = None,
    extra: str | None = None,
    purpose: str = "",
) -> object:
    """Import an optional dependency, raising a clear error if missing.

    Once imported, the module is cached so subsequent calls are free.

    Args:
        module_name: Dotted module name to import (e.g. ``"zmq"``).
        pip_install: Explicit pip package name if it differs from *module_name*.
        extra: ``pyproject.toml`` extras group (e.g. ``"groot-service"``).
        purpose: Human-readable description shown in the error message.

    Returns:
        The imported module object.

    Raises:
        ImportError: With a helpful install instruction.
    """
    if module_name in _lazy_modules:
        return _lazy_modules[module_name]

    try:
        module = importlib.import_module(module_name)
        _lazy_modules[module_name] = module
        return module
    except ImportError:
        install_hint = pip_install or module_name
        parts = [f"'{module_name}' is required"]
        if purpose:
            parts[0] += f" for {purpose}"
        parts.append("Install with:")
        if extra:
            parts.append(f"  pip install 'strands-robots[{extra}]'")
        parts.append(f"  pip install {install_hint}")
        raise ImportError("\n".join(parts)) from None


# ─────────────────────────────────────────────────────────────────────
# Path resolution — single source of truth for all strands-robots paths
# ─────────────────────────────────────────────────────────────────────

#: Default base directory for all user data.
DEFAULT_BASE_DIR = Path.home() / ".strands_robots"


def get_base_dir() -> Path:
    """Get the base directory for strands-robots user data.

    If ``STRANDS_ASSETS_DIR`` is set, returns its parent
    (the assets dir is a subdirectory of the base).
    Otherwise returns ``~/.strands_robots/``.

    Returns:
        Path to the base directory (created if needed).
    """
    custom = os.getenv("STRANDS_ASSETS_DIR")
    if custom:
        d = Path(custom).parent
    else:
        d = DEFAULT_BASE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_assets_dir() -> Path:
    """Get the assets directory (robot model files, meshes, URDFs).

    Resolution:
        1. ``STRANDS_ASSETS_DIR`` env var — used as-is
        2. ``~/.strands_robots/assets/`` — default

    Returns:
        Path to the assets directory (created if needed).
    """
    custom = os.getenv("STRANDS_ASSETS_DIR")
    if custom:
        d = Path(custom)
    else:
        d = DEFAULT_BASE_DIR / "assets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_asset_path(relative_or_absolute: str | Path | None, default_name: str = "") -> Path:
    """Resolve an asset path against the assets directory.

    Args:
        relative_or_absolute: Path to resolve.
            - ``None`` → ``<assets_dir>/<default_name>/``
            - Absolute (or ``~/...``) → expanded as-is
            - Relative → ``<assets_dir>/<relative>/``
        default_name: Fallback subdirectory name when path is None.

    Returns:
        Resolved absolute Path.
    """
    assets = get_assets_dir()
    if relative_or_absolute is None:
        return assets / default_name
    expanded = Path(relative_or_absolute).expanduser()
    if expanded.is_absolute():
        return expanded
    return assets / expanded


# ─────────────────────────────────────────────────────────────────────
# Path safety — prevent traversal via untrusted components
# ─────────────────────────────────────────────────────────────────────


def safe_join(base: Path, untrusted: str) -> Path:
    """Join *base* with an untrusted relative path, rejecting traversal.

    Used to protect against ``../`` escapes in registry-sourced or
    user-supplied path components before they reach the filesystem.

    Args:
        base: Trusted base directory.
        untrusted: Relative path component (may contain ``/`` but must not
            escape *base*).

    Returns:
        Normalised absolute Path under *base*.

    Raises:
        ValueError: If the resulting path would escape *base*.

    Example::

        safe_join(Path("/assets"), "robot/model.xml")   # OK
        safe_join(Path("/assets"), "../etc/passwd")     # ValueError
    """
    joined = Path(os.path.normpath(base / untrusted))
    base_norm = Path(os.path.normpath(base))
    if not (joined == base_norm or str(joined).startswith(str(base_norm) + os.sep)):
        raise ValueError(f"Path traversal blocked: {untrusted!r} escapes {base}")
    return joined


def get_search_paths() -> list[Path]:
    """Get ordered list of asset search paths.

    Used by both :mod:`strands_robots.assets.manager` and
    :mod:`strands_robots.assets.download` — centralised here to avoid
    a circular dependency between those two modules.

    Order (local assets take priority over defaults):
        1. User asset dir (``STRANDS_ASSETS_DIR`` or ``~/.strands_robots/assets/``)
        2. ``CWD/assets`` (project-local)
    """
    paths: list[Path] = []
    user_cache = get_assets_dir()
    if user_cache not in paths:
        paths.append(user_cache)
    cwd_assets = Path.cwd() / "assets"
    if cwd_assets not in paths:
        paths.append(cwd_assets)
    return paths
