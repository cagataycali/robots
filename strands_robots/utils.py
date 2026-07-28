"""Shared utilities for strands-robots."""

import importlib
import logging
import math
import numbers
import os
from pathlib import Path
from typing import Any

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


def require_optionals(
    module_names: list[str] | tuple[str, ...],
    *,
    extra: str | None = None,
    purpose: str = "",
) -> None:
    """Require several optional dependencies, reporting ALL missing ones at once.

    Unlike calling :func:`require_optional` in a loop -- which raises on the
    FIRST missing module and hides the rest -- this probes every name and, if
    any are absent, raises a single ``ImportError`` naming every missing module.
    That lets a caller in a partially-provisioned environment fix all of them in
    one install instead of discovering them one reinstall at a time (each retry
    of a heavy load path is expensive).

    Present modules are imported and cached (same as :func:`require_optional`),
    so a follow-up ``require_optional`` for any of them is free.

    Args:
        module_names: Dotted module names to require (e.g. ``("transformers",
            "peft", "scipy")``).
        extra: ``pyproject.toml`` extras group naming where the deps ship
            (e.g. ``"molmoact2"``); shown in the install hint.
        purpose: Human-readable description shown in the error message.

    Raises:
        ImportError: If one or more modules are missing, listing every missing
            module and an actionable install instruction.
    """
    missing: list[str] = []
    for name in module_names:
        if name in _lazy_modules:
            continue
        try:
            _lazy_modules[name] = importlib.import_module(name)
        except ImportError:
            missing.append(name)

    if not missing:
        return

    joined = ", ".join(f"'{m}'" for m in missing)
    label = "is required" if len(missing) == 1 else "are required"
    parts = [f"{joined} {label}"]
    if purpose:
        parts[0] += f" for {purpose}"
    parts.append("Install with:")
    if extra:
        parts.append(f"  pip install 'strands-robots[{extra}]'")
    parts.append(f"  pip install {' '.join(missing)}")
    raise ImportError("\n".join(parts)) from None


#
# Path resolution - single source of truth for all strands-robots paths
#

#: Default base directory for all user data.
DEFAULT_BASE_DIR = Path.home() / ".strands_robots"


def get_base_dir() -> Path:
    """Get the base directory for strands-robots user data.

    Resolution (in priority order):

    1. ``STRANDS_BASE_DIR`` env var - explicit override. Use this when
       you want to relocate *all* strands-robots user data (assets,
       user registry, caches) to a non-default location.
    2. ``~/.strands_robots/`` - default.

    Note:
        ``STRANDS_ASSETS_DIR`` **only** controls the assets subdirectory
        (see :func:`get_assets_dir`). It does *not* move the base dir,
        so user-level metadata like ``user_robots.json`` always lands in
        a predictable location rather than wherever the assets happen
        to be pointed.

    Returns:
        Path to the base directory (created if needed).
    """
    custom = os.getenv("STRANDS_BASE_DIR")
    d = Path(custom) if custom else DEFAULT_BASE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_assets_dir() -> Path:
    """Get the assets directory (robot model files, meshes, URDFs).

    Resolution:
        1. ``STRANDS_ASSETS_DIR`` env var - used as-is
        2. ``~/.strands_robots/assets/`` - default

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


#
# Path safety - prevent traversal via untrusted components
#


def safe_join(base: Path, untrusted: str, *, resolve_symlinks: bool = False) -> Path:
    """Join *base* with an untrusted relative path, rejecting traversal.

    Used to protect against ``../`` escapes in registry-sourced or
    user-supplied path components before they reach the filesystem. Containment
    is always verified lexically; set *resolve_symlinks* to additionally reject
    symlinked components that escape *base* after resolution.

    Args:
        base: Trusted base directory.
        untrusted: Relative path component (may contain ``/`` but must not
            escape *base*).
        resolve_symlinks: When ``True``, containment is re-verified after full
            symlink resolution so a symlinked component that points outside
            *base* (e.g. ``base/link -> /etc`` followed by ``link/passwd``) is
            rejected. Enable this when *base* is an untrusted or externally
            sourced tree - e.g. a freshly cloned repository - whose symlinks may
            escape. Leave ``False`` (the default) for the managed asset cache,
            whose robot directories are intentionally symlinked to installed
            ``robot_descriptions`` packages that legitimately live outside the
            cache; resolving those would wrongly reject them.

    Returns:
        Normalised absolute Path under *base*.

    Raises:
        ValueError: If the resulting path would escape *base* (lexically, or via
            a symlink when *resolve_symlinks* is set).

    Example::

        safe_join(Path("/assets"), "robot/model.xml")   # OK
        safe_join(Path("/assets"), "../etc/passwd")     # ValueError
    """
    joined = Path(os.path.normpath(base / untrusted))
    base_norm = Path(os.path.normpath(base))
    if not (joined == base_norm or str(joined).startswith(str(base_norm) + os.sep)):
        raise ValueError(f"Path traversal blocked: {untrusted!r} escapes {base}")
    if resolve_symlinks:
        # Lexical normalisation cannot see through symlinks: a component such as
        # ``link/passwd`` where ``base/link`` targets ``/etc`` stays lexically
        # under *base* yet resolves outside it. ``resolve(strict=False)``
        # resolves the existing prefix and appends the remainder lexically for
        # not-yet-created files; resolving *base* too keeps a symlinked base
        # prefix (e.g. /tmp on macOS) consistent on both sides.
        base_resolved = base_norm.resolve()
        joined_resolved = joined.resolve()
        if not (joined_resolved == base_resolved or str(joined_resolved).startswith(str(base_resolved) + os.sep)):
            raise ValueError(f"Path traversal blocked: {untrusted!r} escapes {base} via symlink")
    return joined


def get_search_paths() -> list[Path]:
    """Get ordered list of asset search paths.

    Used by both :mod:`strands_robots.assets.manager` and
    :mod:`strands_robots.assets.download` - centralised here to avoid
    a circular dependency between those two modules.

    Order (local assets take priority over defaults):
        1. User asset dir (``STRANDS_ASSETS_DIR`` or ``~/.strands_robots/assets/``)
        2. ``CWD/assets`` (project-local, deduplicated if it resolves to the same dir)
    """
    paths: list[Path] = []
    user_cache = get_assets_dir()
    paths.append(user_cache)
    cwd_assets = Path.cwd() / "assets"
    if cwd_assets not in paths:
        paths.append(cwd_assets)
    return paths


def process_rss_mb() -> float | None:
    """Current resident set size (RSS) of this process, in megabytes.

    Used to surface ``policy_resident_rss_mb`` telemetry so a caller can see
    whether a heavy model (e.g. a multi-GB VLA checkpoint) is actually resident
    after a load - and, across a multi-episode loop, that it stays resident
    rather than oscillating as it would if the policy were rebuilt per episode.

    Prefers :mod:`psutil` (true *current* RSS, the meaningful "is it resident
    now" number). Falls back to :func:`resource.getrusage`, which reports peak
    RSS for the process (``ru_maxrss``) - an over-estimate of current usage, but
    still a useful floor when psutil is absent. ``ru_maxrss`` is in kilobytes on
    Linux and bytes on macOS; both are normalised to MB.

    Returns:
        Resident memory in MB as a float, or ``None`` when neither source is
        available (e.g. a platform without ``resource``), so callers can omit
        the field rather than report a misleading zero.
    """
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except (ImportError, OSError):
        # psutil missing or the /proc read failed; fall back to stdlib resource.
        pass
    try:
        import resource
        import sys

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss units differ by platform: bytes on macOS, kilobytes on Linux.
        divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
        return float(maxrss) / divisor
    except (ImportError, ValueError, OSError):
        return None


def positive_finite_number_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable positive finite number.

    Shared domain for every CONTINUOUS knob that names a rate or a span of
    time - a control-loop frequency in Hz, a rollout or teleop ``duration`` in
    seconds. Unlike :func:`positive_whole_number_error` a fractional value is
    perfectly usable here (``2.5`` seconds, ``62.5`` Hz), so only the sign and
    the finiteness are constrained. It lives here rather than beside one of its
    callers because those callers sit in different layers
    (:mod:`strands_robots.teleop_mixin` must not depend on
    :mod:`strands_robots.simulation`), and the accepted domain must not diverge
    between them.

    Only a positive finite value can be honored. Such a knob is always a
    divisor (the loop period is ``1 / hz``) or a horizon (``duration *
    frequency`` steps), so ``0`` makes the period undefined or the horizon
    empty, a negative value inverts it, ``nan`` poisons every comparison it
    reaches (``nan > 0`` and ``nan <= 0`` are both ``False``), and ``inf``
    collapses the period to ``0`` - an unthrottled loop, not a fast one.
    Accepts any real scalar (so a NumPy ``np.float32`` rate read from a config
    array passes) and rejects ``bool`` explicitly - an ``int`` subclass whose
    ``True`` would act as a silent ``1``.

    Args:
        value: The caller-supplied value.
        param: The parameter it came from, used in the message.
        context: Message prefix identifying the surface that received it -
            normally the public method name.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        # ``isfinite`` before the sign test: ``nan`` is never ``<= 0``, so
        # ordering these the other way lets it through.
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        return f"{context}: {param} must be > 0, got {value!r}."
    return None


def positive_whole_number_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable positive whole number.

    Shared domain for every media knob that counts frames or pixels - the
    recorders' ``fps``, ``width``, ``height`` and in-memory frame cap, the
    ``run_policy(video=...)`` dict fields, and the
    :func:`~strands_robots.rendering.encode_clip` playback rate. It lives here
    rather than beside one of its callers because those callers sit in different
    layers (:mod:`strands_robots.rendering` must not depend on
    :mod:`strands_robots.simulation`), and the accepted domain must not diverge
    between them. Only a positive whole number can be honored: ``0`` makes the capture loop's ``1 / fps``
    period undefined, a negative rate is rejected by the ffmpeg writer, and a
    zero/negative frame cap drops every frame. Accepts any real scalar with an
    integral value (so a NumPy ``np.int64`` height or a ``30.0`` computed from a
    config float passes) and rejects ``bool`` explicitly - an ``int`` subclass
    whose ``True`` would act as a silent 1.

    Args:
        value: The caller-supplied value.
        param: The parameter (or dict key) it came from, used in the message.
        context: Message prefix identifying the surface that received it -
            ``"video"`` for the :class:`VideoConfig` dict, the method name for a
            keyword parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    message = f"{context}: {param} must be a positive whole number, got {value!r}."
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return message
    numeric = float(value)
    # ``isfinite`` first: ``int(nan)`` raises, and short-circuiting keeps it
    # out of the integrality check below.
    if not math.isfinite(numeric) or numeric != int(numeric) or numeric < 1:
        return message
    return None
