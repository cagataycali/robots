"""Shared utilities for strands-robots."""

import functools
import importlib
import logging
import math
import numbers
import os
import pkgutil
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

# Cache of lazy-loaded modules
_lazy_modules: dict[str, object] = {}


def require_optional(
    module_name: str,
    *,
    pip_install: str | None = None,
    extra: str | None = None,
    purpose: str = "",
    system_install: str | None = None,
) -> object:
    """Import an optional dependency, raising a clear error if missing.

    Args:
        module_name: Dotted module name to import (e.g. ``"zmq"``).
        pip_install: Explicit pip package name if it differs from *module_name*.
        extra: ``pyproject.toml`` extras group (e.g. ``"groot-service"``).
        purpose: Human-readable description shown in the error message.
        system_install: Remedy for a module that arrives with a system package
            rather than from an index - the ROS 2 client libraries are the case
            in this package. Replaces the ``pip install`` block entirely, and
            *pip_install* / *extra* are then not consulted, because a pip
            command for such a module is a remedy the caller can follow to no
            effect: it either installs something that leaves the module exactly
            as missing, or fails outright.

    Returns:
        The imported module object.

    Raises:
        ImportError: With a helpful install instruction, and ``name`` set to
            *module_name* so a caller can tell an absent optional dependency
            from a broken package path without parsing the message. The
            interpreter's own ``ModuleNotFoundError`` stays reachable on
            ``__context__``, which names the module actually not found - that
            differs from *module_name* when the requested module is present but
            something it imports is not.
    """
    if module_name in _lazy_modules:
        return _lazy_modules[module_name]

    try:
        module = importlib.import_module(module_name)
        _lazy_modules[module_name] = module
        return module
    except ImportError:
        parts = [f"'{module_name}' is required"]
        if purpose:
            parts[0] += f" for {purpose}"
        if system_install is not None:
            # No pip line at all: naming one here would hand the caller an
            # instruction that reports success without supplying the module.
            parts.append(system_install)
        else:
            install_hint = pip_install or module_name
            parts.append("Install with:")
            if extra:
                parts.append(f"  pip install 'strands-robots[{extra}]'")
            parts.append(f"  pip install {install_hint}")
        # ``name`` is the module that was attempted, which is what the message
        # already claims is required; the chained ModuleNotFoundError keeps the
        # module the interpreter actually could not find.
        raise ImportError("\n".join(parts), name=module_name) from None


def require_optionals(
    module_names: list[str] | tuple[str, ...],
    *,
    extra: str | None = None,
    purpose: str = "",
    pip_install: Mapping[str, str] | None = None,
) -> None:
    """Require several optional dependencies, reporting ALL missing ones at once.

    Args:
        module_names: Dotted module names to require (e.g. ``("transformers",
            "peft", "scipy")``).
        extra: ``pyproject.toml`` extras group naming where the deps ship
            (e.g. ``"molmoact2"``); shown in the install hint.
        purpose: Human-readable description shown in the error message.
        pip_install: Distribution name per module, for the modules whose import
            name is not what pip installs. Only differing names need an entry;
            anything absent from the mapping is named as-is. Without this the
            per-module hint is built from import names, and for a module like
            ``jwt`` that spelled a remedy -- ``pip install jwt`` -- which
            resolves to a DIFFERENT project on PyPI than the ``PyJWT`` that
            supplies it, so following it leaves the module exactly as missing.
            :func:`require_optional` takes the same argument as a plain string.

    Raises:
        ImportError: If one or more modules are missing, listing every missing
            module and an actionable install instruction, with ``name`` set to
            the first missing module in the order given. The interpreter names
            only the first module an import fails on too, and the full set stays
            in the message. Nothing else carries it here: the raise is outside
            the ``except`` block that probed the modules, so ``__context__`` is
            empty and ``name`` is the only machine-readable report.
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
    distributions = [(pip_install or {}).get(name, name) for name in missing]
    parts.append(f"  pip install {' '.join(distributions)}")
    raise ImportError("\n".join(parts), name=missing[0]) from None


def lerobot_version() -> str:
    """Return the installed lerobot version, or ``"unknown"`` if undeterminable."""
    try:
        from importlib.metadata import version

        return version("lerobot")
    except ImportError:
        return "unknown"


# The lerobot that first accepts ``repo_type: Literal["dataset", "bucket"]``;
# every lerobot-bearing extra in pyproject floors at or above it.
BUCKET_STREAMING_MIN_LEROBOT = "0.6.1"
LEROBOT_UPGRADE = "pip install -U 'strands-robots[lerobot]'"


def lerobot_floor_error() -> str | None:
    """Return why the installed lerobot is below ``BUCKET_STREAMING_MIN_LEROBOT``, else None.

    pip leaves an already-installed older lerobot in place, and its
    ``StreamingLeRobotDataset`` refuses the ``return_uint8`` / ``repo_type``
    keywords :meth:`strands_robots.streaming_dataset.StreamingDatasetReader.open`
    forwards - a ``TypeError`` naming a keyword the caller never passed.
    :func:`strands_robots.doctor.check_lerobot` and
    :mod:`strands_robots.streaming_dataset` both ask here, so one state is
    reported one way.

    Returns:
        A message naming the installed version and the floor (callers add
        ``LEROBOT_UPGRADE`` in their own frame), or ``None`` when the version
        meets the floor or cannot be determined.
    """
    installed = lerobot_version()
    release = re.match(r"\d+(?:\.\d+)*", installed)
    if release is None:
        return None
    floor = tuple(int(part) for part in BUCKET_STREAMING_MIN_LEROBOT.split("."))
    if tuple(int(part) for part in release.group().split(".")) >= floor:
        return None
    return (
        f"lerobot {installed} is below the {BUCKET_STREAMING_MIN_LEROBOT} strands-robots needs: its "
        "StreamingLeRobotDataset refuses the return_uint8 / repo_type keywords stream_dataset() passes"
    )


def lerobot_install_error() -> str | None:
    """Return why ``import lerobot`` cannot reach an install, or None when it can.

    Returns:
        A message naming the cause and its remedy, or ``None`` when lerobot
        imports as an installed package. Callers frame the message for their own
        surface (:func:`strands_robots.doctor.check_lerobot` as a check line, the
        ``use_lerobot`` tool as a tool error) rather than deciding the cause
        themselves, so one state is reported one way everywhere.
    """
    try:
        import lerobot
    except ImportError:
        return "lerobot not installed"

    if getattr(lerobot, "__file__", None) is not None:
        return None

    directories = [str(entry) for entry in getattr(lerobot, "__path__", [])]
    where = ", ".join(directories) or "an empty namespace path"
    remedy = f"uv pip install -e {directories[0]}" if directories else 'uv pip install "strands-robots[lerobot]"'
    return (
        f"lerobot not installed: the name resolves to {where}, a directory with no "
        f"package to import (Python reads it as a namespace package). Install lerobot "
        f"- '{remedy}' if that directory is a lerobot checkout, otherwise "
        f"'uv pip install \"strands-robots[lerobot]\"' and run from a directory with "
        f"no 'lerobot' entry"
    )


@functools.cache
def ensure_lerobot_family_registered(family: str) -> None:
    """Import every subpackage of ``lerobot.<family>`` so its registry is populated.

    Args:
        family: The subpackage of ``lerobot`` to walk, e.g. ``"robots"``.
            Each family is cached separately, so the walk runs once per family.
    """
    package = f"lerobot.{family}"
    try:
        root = importlib.import_module(package)
    except ImportError as exc:
        # Two failure modes, and the log level is what separates them: no
        # reachable install is expected on a sim-only host (debug - the caller
        # still gets a clean "Unsupported <kind> type" at the lookup), while
        # lerobot installed with this family unimportable is a partial install
        # worth a warning without --log-level=DEBUG. A directory named lerobot
        # is the first case, not the second, so the reachability question goes
        # to lerobot_install_error rather than to a bare import.
        problem = lerobot_install_error()
        if problem is not None:
            logger.debug("%s: %s", problem, exc)
        else:
            logger.warning("lerobot is installed but %s is not importable (partial install?): %s", package, exc)
        return

    for _, sub_name, is_pkg in pkgutil.iter_modules(root.__path__):
        if not is_pkg:
            continue
        full_name = f"{package}.{sub_name}"
        try:
            importlib.import_module(full_name)
        except (ImportError, OSError) as exc:
            # A device whose vendor SDK is absent (``pyrealsense2``, ``hidapi``,
            # ``unitree_sdk2py``) or whose ``__init__`` probes the OS. It simply
            # does not appear in the registry, which is the correct outcome: the
            # lookup then refuses the name and lists what did register.
            # ``(ImportError, OSError)`` is the canonical narrow pair for a
            # hardware-probing import per AGENTS.md > Review Learnings (#86).
            logger.debug("[lerobot registry] skip %s: %s", full_name, exc)

    ensure_lerobot_plugins_registered()


@functools.cache
def ensure_lerobot_plugins_registered() -> None:
    """Import every installed third-party lerobot plugin distribution."""
    try:
        from lerobot.utils.import_utils import register_third_party_plugins
    except ImportError:
        # ``register_third_party_plugins`` lives in modern lerobot only; older
        # versions skip this opt-in step (built-ins still work).
        logger.debug("[lerobot registry] register_third_party_plugins unavailable")
        return
    try:
        register_third_party_plugins()
    except (ImportError, AttributeError, OSError) as exc:
        # #291: narrowed from bare ``except Exception`` per AGENTS.md Review
        # Learnings (#86). Three benign, recoverable reasons: a plugin
        # distribution whose import chain is broken (ImportError), a lerobot
        # whose loader entry-point shape differs (AttributeError), or an
        # OS-level probe inside a plugin's registration (OSError). Each
        # degrades to "that plugin is absent from the registry" rather than
        # crashing hardware init; anything else propagates unmasked.
        logger.warning("[lerobot registry] third-party plugin registration failed: %s", exc)


#
# Path resolution - single source of truth for all strands-robots paths
#

#: Default base directory for all user data.
DEFAULT_BASE_DIR = Path.home() / ".strands_robots"


def base_dir_path() -> Path:
    """Where the base directory for strands-robots user data resolves to.

    Returns:
        Path to the base directory, whether or not it exists.
    """
    custom = os.getenv("STRANDS_BASE_DIR")
    return Path(custom) if custom else DEFAULT_BASE_DIR


def get_base_dir() -> Path:
    """Get the base directory for strands-robots user data, creating it if needed.

    Returns:
        Path to the base directory (created if needed).
    """
    d = base_dir_path()
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_assets_dir() -> Path:
    """Get the assets directory (robot model files, meshes, URDFs).

    Returns:
        Path to the assets directory (created if needed).
    """
    custom = os.getenv("STRANDS_ASSETS_DIR")
    if custom:
        d = Path(custom)
    else:
        d = base_dir_path() / "assets"
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
    """Get ordered list of asset search paths."""
    paths: list[Path] = []
    user_cache = get_assets_dir()
    paths.append(user_cache)
    cwd_assets = Path.cwd() / "assets"
    if cwd_assets not in paths:
        paths.append(cwd_assets)
    return paths


def process_rss_mb() -> float | None:
    """Current resident set size (RSS) of this process, in megabytes.

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


def is_boolean(value: Any) -> bool:
    """Return True when ``value`` is a python or a numpy boolean."""
    if isinstance(value, bool):
        return True
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return isinstance(item(), bool)
        except (TypeError, ValueError):  # a multi-element array has no single item
            return False
    return False


def sequence_length(value: Any) -> int | None:
    """Return the length of ``value``, or ``None`` when it does not carry one.

    Args:
        value: Any caller-supplied value a validator needs the length of.

    Returns:
        The component count, or ``None`` when the value has no readable length.
    """
    try:
        return len(value)
    except Exception:
        return None


def refusal_repr(value: Any) -> str:
    """``repr(value)`` for a refusal message, or a description when it cannot be built.

    Args:
        value: The rejected value.

    Returns:
        Its ``repr``, or a bracketed description when that cannot be produced.
    """
    try:
        return repr(value)
    except Exception:
        return _describe_unrenderable(value)


def refusal_str(value: Any) -> str:
    """``str(value)`` for a refusal message, or a description when it cannot be built.

    Args:
        value: The rejected value.

    Returns:
        Its ``str``, or a bracketed description when that cannot be produced.
    """
    try:
        return str(value)
    except Exception:
        return _describe_unrenderable(value)


_LOG_CONTROL_RE: Final = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def log_safe(value: Any, limit: int = 200) -> str:
    """A caller-supplied value, safe to place in one log line.

    A log file is parsed by line, so a carriage return or a line feed inside a
    value that arrived from outside the process (a dataset id, a robot name, a
    download URL, a peer id) forges a second entry that never happened. Both
    become their two-character escapes, every other control character becomes
    ``\\xNN``, and a value longer than ``limit`` is cut with ``...`` so one
    value cannot fill the file either. Use it as the ``%s`` argument of a log
    call rather than formatting the value into the message string.

    Args:
        value: The value to render; ``str(value)`` is used, or a description
            when that itself raises.
        limit: Longest text returned, in characters.

    Returns:
        A single-line string.
    """
    text = refusal_str(value).replace("\r", "\\r").replace("\n", "\\n")
    text = _LOG_CONTROL_RE.sub(lambda m: f"\\x{ord(m.group()):02x}", text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _describe_unrenderable(value: Any) -> str:
    """Describe a value whose own rendering raised.

    Args:
        value: The value whose rendering raised.

    Returns:
        A bracketed description, which is built from the type and (for an
        integer) a bit count, so it cannot itself raise.
    """
    if isinstance(value, int):
        return f"<int of {value.bit_length()} bits>"
    return f"<unrepresentable {type(value).__name__}>"


def _describe_failed_read(exc: Exception) -> str:
    """``RuntimeError: no keys for you`` for a read a refusal could not complete.

    Args:
        exc: The exception the read raised.

    Returns:
        Its type name and text, which cannot itself raise.
    """
    return f"{type(exc).__name__}: {refusal_str(exc)}"


def _read_to_quote(value: Any) -> tuple[list[Any] | None, str | None]:
    """The elements of ``value`` for a refusal to quote, or why they could not be read.

    Args:
        value: The caller-supplied value a refusal wants to quote.

    Returns:
        ``(elements, None)`` when the read finished, or ``(None, description)``
        when it did not. Exactly one side is ever populated, so a caller holding
        a ``None`` description holds the elements.
    """
    try:
        return list(value), None
    except Exception as exc:
        return None, _describe_failed_read(exc)


def refusal_container_repr(value: Any) -> str:
    """``repr(value)`` for a refusal that reports a whole container, elementwise if it must.

    Args:
        value: The rejected container.

    Returns:
        Its ``repr``; an elementwise rendering when that raises; or a bracketed
        description when ``value`` cannot be iterated either, which is
        :func:`refusal_repr`'s answer for a value that is not a container at
        all - every one of these guards accepts ``Any``, so a scalar reaches
        them too.
    """
    try:
        return repr(value)
    except Exception:
        pass
    if isinstance(value, Mapping):
        try:
            items = list(value.items())
        except Exception:
            return _describe_unrenderable(value)
        return "{" + ", ".join(f"{refusal_repr(key)}: {refusal_repr(val)}" for key, val in items) + "}"
    try:
        elements = list(value)
    except Exception:
        return _describe_unrenderable(value)
    return "[" + ", ".join(refusal_repr(element) for element in elements) + "]"


def _beyond_float_range(value: Any) -> bool:
    """Whether ``float(value)`` overflows, i.e. no float64 stands for ``value``.

    Returns:
        ``True`` when the conversion overflows. ``False`` when it succeeds *or*
        fails any other way - a :class:`numbers.Real` registration guarantees no
        working ``__float__``, and a value no number can be read from at all is
        not a magnitude complaint and must not be reported as one.
    """
    try:
        float(value)
    except OverflowError:
        return True
    except Exception:
        return False
    return False


def positive_finite_number_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable positive finite number.

    Args:
        value: The caller-supplied value.
        param: The parameter it came from, used in the message.
        context: Message prefix identifying the surface that received it -
            normally the public method name.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return f"{context}: {param} must be > 0, got {refusal_repr(value)}."
    if _beyond_float_range(value):
        # A real past the float64 range is positive-or-negative and finite, so
        # neither of this guard's own reasons is true of it - hence its own text.
        # Refusing stays right: the value is a divisor (``1 / hz``) or a
        # multiplier evaluated in float64, and no float64 stands for it.
        return f"{context}: {param} must be within the range of a 64-bit float, got {refusal_repr(value)}."
    try:
        # ``isfinite`` before the sign test: ``nan`` is never ``<= 0``, so
        # ordering these the other way lets it through.
        unusable = not math.isfinite(float(value)) or float(value) <= 0
    except Exception:
        # A ``numbers.Real`` registration owes this guard no working
        # ``__float__``, and a value no number can be read from is refused for
        # the same reason a non-real one is - the message it already had.
        unusable = True
    if unusable:
        return f"{context}: {param} must be > 0, got {refusal_repr(value)}."
    return None


def finite_number_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable finite number of either sign.

    Args:
        value: The caller-supplied value.
        param: The parameter it came from, used in the message.
        context: Message prefix identifying the surface that received it -
            normally the public method name.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return f"{context}: {param} must be a finite number, got {refusal_repr(value)}."
    if _beyond_float_range(value):
        # ``10**400`` *is* a finite number, so this guard's own reason would be
        # a false statement about it. Refusing stays right: the docstring above
        # is explicit that an accepted value is serialized onto the wire as an
        # IEEE-754 float64, and this one has no float64 form to serialize.
        return f"{context}: {param} must be within the range of a 64-bit float, got {refusal_repr(value)}."
    try:
        unusable = not math.isfinite(float(value))
    except Exception:
        unusable = True
    if unusable:
        return f"{context}: {param} must be a finite number, got {refusal_repr(value)}."
    return None


def positive_whole_number_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable positive whole number.

    Args:
        value: The caller-supplied value.
        param: The parameter (or dict key) it came from, used in the message.
        context: Message prefix identifying the surface that received it -
            ``"video"`` for the :class:`VideoConfig` dict, the method name for a
            keyword parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """

    def message() -> str:
        # Rendered on demand, not up front. The text used to be built on this
        # function's first line, before the value had been classified at all, so
        # ``repr`` raised on an outsized ``int`` ahead of every verdict - the
        # guard failing while preparing a refusal it had not decided to return,
        # and doing it on the accept path too.
        return f"{context}: {param} must be a positive whole number, got {refusal_repr(value)}."

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return message()
    if _beyond_float_range(value):
        # ``10**400`` is a positive whole number, so ``message()`` would state
        # something false about it. It is refused rather than accepted, and
        # deliberately unlike its ``non_negative`` sibling - see the docstring.
        return f"{context}: {param} must be within the range of a 64-bit float, got {refusal_repr(value)}."
    try:
        numeric = float(value)
    except Exception:
        # No number can be read from it at all; refused, never raised.
        return message()
    # ``isfinite`` first: ``int(nan)`` raises, and short-circuiting keeps it
    # out of the integrality check below.
    if not math.isfinite(numeric) or numeric != int(numeric) or numeric < 1:
        return message()
    return None


def non_negative_whole_number_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable non-negative whole number.

    Args:
        value: The caller-supplied value.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it - the
            public method name, or the class name for a constructor parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """

    def message() -> str:
        # Rendered on demand, not up front: an accepted count must neither pay
        # for nor be refused by a message it never receives. A count wider than
        # ``sys.get_int_max_str_digits()`` is accepted here, and building the
        # text eagerly made ``repr`` raise on it - the guard failing on the
        # accept path, doing work only the refuse path needs.
        return f"{context}: {param} must be a non-negative whole number, got {refusal_repr(value)}."

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return message()
    try:
        integral = int(value)
    except (OverflowError, TypeError, ValueError):
        # The one conversion answers every value no count can be read from:
        # ``int(nan)`` raises ``ValueError`` and ``int(+-inf)``
        # ``OverflowError``, so no separate ``isfinite`` check is needed - and
        # none is possible, because ``isfinite`` needs a ``float()`` whose own
        # ``OverflowError`` on an ``int`` wider than a float is the escape this
        # form exists to remove. ``TypeError`` covers a ``numbers.Real``
        # registered without an ``__int__``: refused, never raised.
        return message()

    # ``int`` rather than ``math.trunc``, which looks like the conversion the
    # ABC guarantees and is not usable here: NumPy scalars implement
    # ``__int__`` but not ``__trunc__``, so ``math.trunc(np.int64(3))`` raises
    # ``TypeError`` and the NumPy rows this domain must honor would be refused.
    #
    # The sign is read off the coerced ``int`` rather than the original: an
    # ``int``-to-``int`` comparison is total, where ``value < 0`` would defer to
    # a ``__lt__`` a ``numbers.Real`` registration does not actually guarantee.
    # ``integral != value`` is what establishes integrality - exactly, and with
    # no float round-trip, so a large-but-usable count is neither rounded nor
    # refused - and it falls back to Python's default inequality rather than
    # raising when a type supplies no ``__eq__``.
    if integral < 0 or integral != value:
        return message()
    return None


def step_aborted_msg(completed: int, requested: int, *, context: str = "step") -> str:
    """Refusal text when a batched step loop loses its world mid-run.

    Args:
        completed: Steps actually advanced before the world went away.
        requested: The count the caller asked for.
        context: Calling method name, for the message prefix.

    Returns:
        Human-readable refusal text.
    """
    return (
        f"{context}: world was destroyed mid-run after {completed} of {requested} steps; aborting. "
        "The steps already advanced are not rolled back."
    )


def positive_count_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable positive integer count.

    Args:
        value: The caller-supplied value.
        param: The parameter name it came from, used in the message. Callers that
            accept a ``{robot_name: count}`` mapping pass a subscripted label
            (``"action_horizon['alice']"``) so the message names the entry the
            caller got wrong rather than the whole mapping.
        context: Message prefix identifying the surface that received it - the
            public method name, or the class name for a constructor parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return f"{context}: {param} must be a positive integer, got {refusal_repr(value)}."
    return None


def tcp_port_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` cannot address a TCP port.

    Args:
        value: The caller-supplied value.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it - the
            requested action for an agent tool, or the class name for a
            constructor parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        return f"{context}: invalid {param}: {refusal_repr(value)} (expected 1-65535)"
    return None


# Characters that end the host inside ``<scheme>://<host>:<port>``. Each one
# starts a later URI component, so a host carrying one does not name a bad host -
# it names a different URI. ``:`` is in the set because the port follows it, and a
# bracketed IPv6 literal (``[::1]``) is the one place it belongs to the host.
_URI_COMPONENT_DELIMITERS = frozenset("/?#@:[]\\")


def _read_uri_host(value: str) -> tuple[tuple[str, str, list[str]] | None, str | None]:
    """The host as a plain string, its body and its delimiters - or why it did not read.

    Args:
        value: The caller-supplied host, already known to be a ``str``.

    Returns:
        ``((spelling, body, delimiters), None)`` when the read finished - ``spelling``
        a plain ``str`` copy the refusal can interpolate, ``body`` unbracketed and
        ``delimiters`` sorted - or ``(None, description)`` when it did not. Exactly
        one side is ever populated.
    """
    try:
        spelling = str(value)
        bracketed = value.startswith("[") and value.endswith("]")
        body = str(value[1:-1] if bracketed else value)
        own = frozenset(":") if bracketed else frozenset()
        bad = sorted(
            {c for c in body if (c in _URI_COMPONENT_DELIMITERS and c not in own) or not c.isprintable() or c.isspace()}
        )
        return (spelling, body, bad), None
    except Exception as exc:
        return None, _describe_failed_read(exc)


def dial_host_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` cannot address the host half of a websocket URI.

    Args:
        value: The caller-supplied host.
        param: The field or parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it.

    Returns:
        An error message, or ``None`` when the value can address a host.
    """
    shown = refusal_repr(value)
    if not isinstance(value, str):
        return (
            f"{context}: {param} must be a string hostname or IP literal, got {shown} "
            f"({type(value).__name__}). It is interpolated into the websocket URI the client "
            "dials (ws://<host>:<port>), which carries it verbatim, so the client dials a name "
            "nobody wrote rather than reporting the value."
        )
    read, unreadable = _read_uri_host(value)
    if read is None:
        return (
            f"{context}: {param} could not be read as a host ({unreadable}), got {shown}; "
            "a value whose own string operations do not answer cannot be checked against the "
            "host half of the websocket URI it would be interpolated into (ws://<host>:<port>). "
            "Pass a plain hostname or IP literal, e.g. '127.0.0.1'."
        )
    spelling, body, bad = read
    would_be = refusal_repr(f"ws://{spelling}:<port>")
    if not body:
        return (
            f"{context}: {param} must name a host to dial, got {shown}; "
            f'{would_be} is not a URI (the parse reports "hostname isn\'t provided"). '
            "Use '0.0.0.0' to reach a server bound on every interface, or '127.0.0.1' for a local one."
        )
    if bad:
        hint = " Pass a bracketed literal for IPv6 (e.g. '[::1]')." if ":" in bad else ""
        return (
            f"{context}: {param} must be a bare hostname or IP literal, got {shown}; "
            f"{refusal_container_repr(bad)} cannot appear in the host half of the websocket URI it "
            f"is interpolated into (ws://<host>:<port>), so {would_be} names a different URI rather "
            "than a host - a '/' puts the validated port in the path and the client dials :80 "
            f"instead.{hint}"
        )
    return None


def non_negative_count_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable non-negative integer count.

    Args:
        value: The caller-supplied value.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it - the
            public method name, or the class name for a constructor parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return f"{context}: {param} must be a non-negative integer, got {refusal_repr(value)}."
    return None


def declared_count(value: object) -> int | None:
    """The count a dataset's ``meta/info.json`` declares, or ``None`` for none.

    Args:
        value: The value the metadata file carried under the count's key, or
            ``None`` when the key is absent.

    Returns:
        The declared count, or ``None`` when the file declares no usable one.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def step_cadence_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable cadence in training steps.

    Args:
        value: The caller-supplied cadence.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it - the
            public method name, the tool name, or a backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`.

    Returns:
        An error message, or ``None`` when the value is a usable cadence.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return (
            f"{context}: {param} must be an integer number of steps, got {refusal_repr(value)}. "
            "A fractional, non-finite, boolean or non-numeric cadence cannot be honored - it is "
            "used as the modulus of a step % cadence test; pass a whole number of steps, or a "
            "non-positive one to disable periodic saving."
        )
    return None


def torch_device_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a device string torch can parse.

    Args:
        value: The caller-supplied device.
        param: Field name, quoted in the message so the refusal names the knob.
        context: Public surface or provider name, prefixed to the message.

    Returns:
        An error message naming *context* and *param*, or ``None`` when torch can
        parse the value.
    """
    if not isinstance(value, str):
        return (
            f"{context}: {param} must be a torch device string, got {type(value).__name__}. "
            "Pass a device type, optionally with an index (e.g. 'cuda', 'cuda:0', 'cpu', 'mps')."
        )
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch missing -> domain unknown, pass through
        return None
    try:
        torch.device(value)
    except (RuntimeError, ValueError) as e:
        return (
            f"{context}: {param}={refusal_repr(value)} is not a torch device string ({e}). "
            "Pass a device type, optionally with an index (e.g. 'cuda', 'cuda:0', 'cpu', 'mps')."
        )
    return None


#: Highest DDS domain id whose RTPS discovery ports fit the 16-bit port space.
#:
#: RTPS derives every discovery port from the domain id (RTPS 2.2 sec. 9.6.1.1):
#: ``PB + DG * domain_id + d0`` for the SPDP multicast port and
#: ``PB + DG * domain_id + d1 + PG * participant_id`` for the unicast one. With
#: the standard values (``PB=7400``, ``DG=250``, ``d0=0``, ``d1=10``,
#: ``PG=2``) domain 232 lands on ports 65400/65410 and domain 233 lands on
#: 65650 - past the end of the port space, so there is nothing to bind. The
#: bound is the protocol's, not a policy choice.
MAX_DDS_DOMAIN_ID = 232


def dds_domain_id_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` cannot name a DDS domain.

    Args:
        value: The caller-supplied value.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it,
            usually the class name for a constructor parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_DDS_DOMAIN_ID:
        return f"{context}: invalid {param}: {refusal_repr(value)} (expected 0-{MAX_DDS_DOMAIN_ID})"
    return None


#: Isaac-GR00T releases :class:`~strands_robots.policies.groot.Gr00tPolicy` loads.
#:
#: The domain of its ``groot_version=``, which selects a loader rather than
#: naming a package version: each spelling has a branch in
#: ``Gr00tPolicy._load_local_policy`` that imports that release's own entry
#: point. The tuple is the loaders the policy has, not the releases NVIDIA
#: ships, which is why it is stated once here and graded against the dispatch.
SUPPORTED_GROOT_VERSIONS = ("n1.5", "n1.6", "n1.7")


def groot_version_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` names no Isaac-GR00T release with a loader.

    Args:
        value: The caller-supplied release selector.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it,
            usually the class name for a constructor parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if value is None or value in SUPPORTED_GROOT_VERSIONS:
        return None
    return (
        f"{context}: invalid {param}: {refusal_repr(value)} names no Isaac-GR00T release "
        f"this policy has a loader for (expected one of {list(SUPPORTED_GROOT_VERSIONS)}, "
        "or None to auto-detect the installed release)"
    )


MAX_ZMQ_TIMEOUT_MS = 2**31 - 1


def coerce_zmq_timeout_ms(method: str, param_name: str, value: Any) -> tuple[int | None, str | None]:
    """Read ``value`` as a ZMQ send/receive timeout in milliseconds.

    Args:
        method: Message prefix identifying the surface that received the value,
            usually the class name for a constructor parameter.
        param_name: The parameter name it came from, used in the message.
        value: The caller-supplied value.

    Returns:
        ``(timeout_ms, None)`` with ``timeout_ms`` an ``int`` when the value is
        usable, or ``(None, reason)`` when it is not.
    """
    if (reason := positive_whole_number_error(value, param_name, method)) is not None:
        return None, reason
    # Read once, and compare the number this function produced rather than the
    # caller's value a second time. The first spelling of this validated with
    # the guard and then read ``value`` twice more - ``float(value)`` for the
    # range and ``int(value)`` for the result - which is the escape #1906 closed
    # for the vector guards: independent reads are not obliged to agree, so the
    # magnitude a refusal quoted need not have been the magnitude the ceiling
    # examined, and a ``numbers.Real`` whose second ``__float__`` refuses raised
    # straight out of a function whose contract is to answer with text. This
    # module carries that as a scanned invariant - no function in it may reach a
    # ``float()`` that no ``try`` protects - and the second read was exactly such
    # a conversion, so this is the invariant's verdict and not a style
    # preference.
    #
    # ``int`` is exact for every value that reaches here and needs no ``try``:
    # the guard established a ``numbers.Real`` whose ``float()`` succeeded, is
    # finite and is integral, so no rounding is possible, and the arbitrary
    # precision of ``int`` means an integral ``1e300`` converts rather than
    # overflowing and is then refused by the ceiling below.
    timeout_ms = int(value)
    if timeout_ms > MAX_ZMQ_TIMEOUT_MS:
        return None, (
            f"{method}: {param_name} must be at most {MAX_ZMQ_TIMEOUT_MS} ms "
            f"(the largest send/receive timeout ZMQ can store), got {refusal_repr(value)}."
        )
    return timeout_ms, None


def _read_name_list(value: object, param: str, context: str) -> tuple[list[Any], str | None]:
    """Read ``value`` into a list once, or return why the read could not finish.

    Args:
        value: The caller-supplied value, already known to be a
            :class:`~collections.abc.Sequence` by the check above the call.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it.

    Returns:
        The entries read and ``None``, or the entries read so far and the
        message naming why the read stopped.
    """
    try:
        elements = iter(value)  # type: ignore[call-overload]
    except Exception as exc:
        return [], (
            f"{context}: {param} could not be iterated: "
            f"{_describe_failed_read(exc)} (got {refusal_container_repr(value)}). "
            f"Pass a list or tuple of names."
        )
    entries: list[Any] = []
    while True:
        # ``next()`` is called explicitly because a ``for`` cannot guard the call
        # it makes, so an exception raised while *producing* an entry would escape
        # this guard exactly as one from ``__iter__`` would. ``StopIteration`` is
        # the read finishing normally; an empty value is accepted as before,
        # emptiness meaning "not supplied" to every caller of this function.
        try:
            entry = next(elements)
        except StopIteration:
            return entries, None
        except Exception as exc:
            return entries, (
                f"{context}: {param}[{len(entries)}] could not be read: "
                f"{_describe_failed_read(exc)} (got {refusal_container_repr(value)}). "
                f"Pass a list or tuple of names."
            )
        entries.append(entry)


def name_list_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable list of distinct key names.

    Args:
        value: The caller-supplied value.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it - the
            public method name, or the class name for a constructor parameter.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if isinstance(value, str | bytes):
        # One read builds the whole consequence clause. ``bytes.decode`` is
        # overridable and a ``str`` subclass owns its own ``__iter__`` and
        # ``__len__``, so producing the characters and counting them are both
        # reads of the caller's value - and taking the count from the same read
        # that produced the characters is #1897's property applied to a message,
        # rather than a second read the first is not obliged to agree with.
        shown: Any = value
        characters: list[Any] | None = None
        unreadable: str | None = None
        try:
            shown = value.decode(errors="replace") if isinstance(value, bytes) else value
        except Exception as exc:
            unreadable = _describe_failed_read(exc)
        else:
            characters, unreadable = _read_to_quote(shown)
        if characters is None:
            consequence = (
                f"A string is iterable per character, so this would be read as one name per "
                f"character; its own characters could not be read to quote them here ({unreadable})."
            )
        else:
            consequence = (
                f"A string is iterable per character, so this would be read as "
                f"{refusal_container_repr(characters[:6])}{' ...' if len(characters) > 6 else ''} "
                f"({len(characters)} name(s))."
            )
        return (
            f"{context}: {param} must be a list of names, not a single string, "
            f"got {refusal_container_repr(value)}. {consequence} "
            f"Wrap it in a list: [{refusal_repr(shown)}]."
        )
    if isinstance(value, Mapping):
        # The verdict is not in doubt on this branch, so a key read that fails
        # degrades the remedy and leaves the verdict standing (#1903).
        names, unquotable = _read_to_quote(value)
        remedy = (
            f"pass the names as a list; its own keys could not be read to quote them here ({unquotable})."
            if names is None
            else f"pass the names as a list: {refusal_container_repr(names)}."
        )
        return (
            f"{context}: {param} must be a list of names, not a mapping, "
            f"got {refusal_container_repr(value)}. "
            f"A mapping is iterable over its keys, so its values would be discarded - {remedy}"
        )
    if not isinstance(value, Sequence):
        return (
            f"{context}: {param} must be a list of names, got {type(value).__name__} "
            f"({refusal_container_repr(value)}). Pass a list or tuple; a one-shot iterator cannot be used "
            f"because the value is read more than once."
        )
    # One read, and every verdict below is about the list it produced. A
    # ``Sequence`` is not obliged to answer two reads the same way, so the
    # per-entry checks and the duplicate check have to see the same entries for
    # their verdicts to be about the same value - see :func:`_read_name_list`.
    entries, unread = _read_name_list(value, param, context)
    if unread is not None:
        return unread
    for i, entry in enumerate(entries):
        if not isinstance(entry, str):
            return f"{context}: {param}[{i}] must be a name (str), got {type(entry).__name__} ({refusal_repr(entry)})."
        if not entry.strip():
            return f"{context}: {param}[{i}] must be a non-blank name, got {refusal_repr(entry)}."
    seen: set[str] = set()
    repeated: set[str] = set()
    for entry in entries:
        if entry in seen:
            repeated.add(entry)
        seen.add(entry)
    if repeated:
        return (
            f"{context}: {param} must not repeat a name, got {refusal_container_repr(entries)} "
            f"({refusal_container_repr(sorted(repeated))} appears more than once)."
        )
    return None


def camera_schema_key(name: str) -> str:
    """The LeRobot dataset feature name a scene camera is recorded under.

    Args:
        name: A scene camera name, raw (``arm0/wrist``) or already schema-safe.

    Returns:
        The dataset feature name, without the ``observation.images.`` prefix.
    """
    return name.replace("/", "__")


#: Why a boolean is refused as a vector component. Worded for what a vector
#: actually carries - a coordinate, a geom extent, a friction coefficient, a
#: colour channel - rather than the radians / rad/s / newtons of the joint
#: writers, whose reason is stated separately beside them. Lives here because
#: every surface that coerces a vector component shares this one answer.
BOOLEAN_VECTOR_REASON = (
    "float(True) is 1.0, so a boolean would silently write 1.0 where a "
    "coordinate, extent or colour channel belongs, and the call would report "
    "success. Pass the component as a number."
)


def _read_finite_vector(method: str, param_name: str, vec: Any) -> tuple[list[float], str | None]:
    """Read ``vec`` into floats once, or return why it is unusable.

    Args:
        method: Calling method name, used in error text.
        param_name: Parameter name, used in error text.
        vec: The caller-supplied value.

    Returns:
        The floats read and ``None``, or the floats read so far and the message
        naming why the value is unusable - the same text
        :func:`finite_vector_error` has always returned.
    """
    # Collected as the read proceeds, so every refusal below can report the
    # components read before the one that stopped it, and an accepted value can
    # be returned without a second read of it.
    floats: list[float] = []
    # ``TypeError`` is what "not iterable" means in Python and the only exception
    # a well-behaved ``__iter__`` raises, but a guard whose whole purpose is to
    # answer an unusable input with a message cannot assume good behaviour of the
    # value it was handed: any other exception escaping here would be the same
    # defect as the rendering (#1873), scalar-conversion (#1874) and
    # container-conversion (#1875) escapes, on the one path that must not raise.
    # It gets its own text because the two verdicts are not the same measurement -
    # a value whose iteration raised may well have been a list/tuple of numbers,
    # and this guard never found out, so reporting it as "must be a list/tuple of
    # numbers" would state something unmeasured (#1878).
    #
    # The iterator is bound and then iterated, rather than ``iter(vec)`` being
    # called for its exception and ``vec`` iterated again: one ``__iter__`` call
    # is what the probe is asking about, and a second one is free to answer
    # differently than the one that was checked.
    #
    # This clause answers for ``__iter__`` only, and a success here says nothing
    # about the read that follows (#1889): CPython synthesises the iterator for a
    # legacy ``__getitem__`` sequence *without* calling ``__getitem__``, and
    # ``iter()`` of a generator cannot fail at all, so a value that fails part-way
    # through its own iteration arrives past this line intact. The element loop
    # below guards the call that produces each element for that reason. A
    # materialising ``list(vec)`` would cover both in one clause and is declined
    # for a stronger reason than the memory it holds: it raises before any element
    # has been examined, so its verdict could not say how far the read got - the
    # one thing that distinguishes a part-way failure from an outright refusal.
    # ``exc`` is rendered through ``refusal_str`` rather than interpolated: a
    # value hostile enough to raise a non-``TypeError`` from ``__iter__`` is not
    # a value whose exception is assumed to have a working ``__str__``, and that
    # is the #1873 escape reintroduced inside the fix for this one.
    try:
        elements = iter(vec)
    except TypeError:
        return floats, f"{method}: '{param_name}' must be a list/tuple of numbers, got {refusal_container_repr(vec)}"
    except Exception as exc:
        return floats, (
            f"{method}: '{param_name}' could not be iterated: "
            f"{type(exc).__name__}: {refusal_str(exc)} (got {refusal_container_repr(vec)}). "
            f"Pass a list or tuple of numbers."
        )
    while True:
        # ``next()`` is called explicitly because a ``for`` cannot guard the call
        # it makes: an exception raised while *producing* an element is the
        # ``__iter__`` escape one level in, on the same path that must not raise.
        # It is reported against the element's own index, the convention every
        # other guard here that names one already uses (``{param}[{i}]``), so the
        # message states both what failed and what was read before it.
        # ``StopIteration`` is the read finishing normally; an empty ``vec`` is
        # accepted exactly as before, a component count not being this guard's
        # question.
        try:
            _elem = next(elements)
        except StopIteration:
            break
        except Exception as exc:
            return floats, (
                f"{method}: '{param_name}[{len(floats)}]' could not be read: "
                f"{type(exc).__name__}: {refusal_str(exc)} (got {refusal_container_repr(vec)}). "
                f"Pass a list or tuple of numbers."
            )
        # ``numbers.Real`` accepts a numpy scalar (``np.float32`` / ``np.int64``
        # are registered) and rejects a string, ``None`` or a nested list.
        # ``bool`` is an ``int`` subclass, so it would otherwise pass as a
        # silent ``1.0`` - a ``True`` coordinate placing a body 1 m out. The
        # agent-tool router already refuses a bool component, so refusing it
        # here keeps the direct API and the tool surface in step.
        if is_boolean(_elem):
            return floats, (
                f"{method}: '{param_name}' elements must be numbers, not a bool "
                f"(got {refusal_container_repr(vec)}). {BOOLEAN_VECTOR_REASON}"
            )
        if not isinstance(_elem, numbers.Real):
            return floats, f"{method}: '{param_name}' elements must be numbers, got {refusal_container_repr(vec)}"
        # An element past the float64 range is a *magnitude* complaint and gets
        # its own reason, exactly as the scalar guards give one (#1874). The
        # order matters: ``_beyond_float_range`` answers only ``OverflowError``,
        # so a registered ``numbers.Real`` with no working ``__float__`` falls
        # through to the not-a-number text below rather than being mis-reported
        # as out of range.
        if _beyond_float_range(_elem):
            return floats, (
                f"{method}: '{param_name}' must contain numbers within the range of a 64-bit float, "
                f"got {refusal_container_repr(vec)}"
            )
        try:
            numeric = float(_elem)
        except Exception:
            return floats, f"{method}: '{param_name}' elements must be numbers, got {refusal_container_repr(vec)}"
        if not math.isfinite(numeric):
            return floats, (
                f"{method}: '{param_name}' must contain finite numbers (no nan/inf), got {refusal_container_repr(vec)}"
            )
        floats.append(numeric)
    return floats, None


def finite_vector_error(method: str, param_name: str, vec: Any) -> str | None:
    """Return an error message if any element of ``vec`` is not a finite number."""
    err = _read_finite_vector(method, param_name, vec)[1]
    if err is not None:
        return err
    if sequence_length(vec) is None:
        # The components were read and were finite; what the value cannot supply
        # is a length for the caller to count. Same words as the sibling
        # coercions, because it is the same verdict about the same value.
        return f"{method}: '{param_name}' must be a list/tuple of numbers, got {refusal_container_repr(vec)}"
    return None


def _read_pose_vector(method: str, param_name: str, vec: Any, expected_len: int) -> tuple[list[float], str | None]:
    """Read ``vec`` into ``expected_len`` floats once, or say why it is unusable.

    Args:
        method: Calling method name, used in error text.
        param_name: Parameter name, used in error text.
        vec: The caller-supplied value.
        expected_len: Component count the target buffer defines.

    Returns:
        The floats read and ``None``, or the floats read so far (none, when the
        length was refused) and the message :func:`pose_vector_error` returns.
    """
    # Read through the shared probe rather than a second ``len()`` here. The
    # rule "how many components is this?" has one owner (:func:`sequence_length`)
    # precisely so it cannot be answered two ways, and this call site answered it
    # a second way: an ``except TypeError`` clause, which is the gap that owner
    # closed - a ``__len__`` refusing any other way, or returning a negative or
    # an oversized ``int`` that CPython itself refuses to convert, escaped this
    # guard and the structured contract its callers document. Both verdicts below
    # are unchanged, including the text for a value carrying no readable length.
    # A str/bytes is refused on TYPE, before its length is asked for, because it
    # carries one and the answer is meaningless: it counts characters, not
    # components. ``add_camera(target="cube")`` - a plausible call, since a camera
    # aimed at a named body is what the parameter looks like it takes - was refused
    # as ``'target' must be a 3-element vector, got 4 ('cube')``, which reports the
    # string's character count as though it were a component count and points the
    # caller at fixing the length rather than the type.
    #
    # Every string is refused either way - ``"123"`` reaches the element read and is
    # refused there, on its characters - so this changes no verdict, only which
    # question the caller is sent to fix. That is the whole point: the two refusals
    # a string currently draws are picked by its LENGTH, so the same mistake reads
    # as three unrelated problems. At ``expected_len`` 3, ``target="box"`` reports
    # ``elements must be numbers``, ``target="cube"`` reports a wrong element
    # *count*, and ``target="0.1,0.2,0.3"`` reports a count of 11. One of those
    # names the actual error and the other two describe the string's characters as
    # though they were components.
    #
    # :func:`image_keys_error` already refuses str/bytes this way on the name-list
    # path next door, for the same reason (a string is iterable per character), so
    # this is one library rule applied to the surface that was missing it.
    if isinstance(vec, str | bytes):
        return [], (
            f"{method}: '{param_name}' must be a list/tuple of {expected_len} numbers, "
            f"got {type(vec).__name__} {refusal_container_repr(vec)}. A string carries a "
            f"length, but it counts characters rather than components, so it cannot be read "
            f"as a pose - pass the {expected_len} numbers themselves."
        )
    length = sequence_length(vec)
    if length is None:
        return [], (
            f"{method}: '{param_name}' must be a list/tuple of {expected_len} numbers, "
            f"got {refusal_container_repr(vec)}"
        )
    if length != expected_len:
        return [], (
            f"{method}: '{param_name}' must be a {expected_len}-element vector, "
            f"got {length} ({refusal_container_repr(vec)})"
        )
    floats, err = _read_finite_vector(method, param_name, vec)
    if err is not None:
        return floats, err
    # The gate above accepted a length the value *reported*; these are the
    # components it *produced*. A ``__len__`` of 4 over a read yielding 3 passed
    # that gate and returned a 3-component vector where a wxyz quaternion was
    # promised - reaching the bare ``ValueError`` inside the ``data.qpos``
    # assignment this guard exists to prevent, through the guard rather than
    # around it (#1909). The refusal quotes the components, since they are what
    # disagrees with the count; ``length`` is named too, because "got 3" alone
    # would not say that the value's own length claimed otherwise.
    if len(floats) != expected_len:
        return floats, (
            f"{method}: '{param_name}' must be a {expected_len}-element vector, "
            f"got {len(floats)}: {floats}. Its length reported {length}, so the "
            f"components it produced are not the vector its length promised."
        )
    return floats, None


def pose_vector_error(method: str, param_name: str, vec: Any, expected_len: int) -> str | None:
    """Return an error message if ``vec`` is not ``expected_len`` finite numbers."""
    return _read_pose_vector(method, param_name, vec, expected_len)[1]


def coerce_pose_vector(
    method: str, param_name: str, vec: Any, expected_len: int
) -> tuple[list[float] | None, str | None]:
    """Validate an optional pose vector and normalize it to plain floats.

    Args:
        method: Calling method name, used in error text.
        param_name: Parameter name, used in error text.
        vec: The caller's value, or ``None`` when the parameter was omitted.
        expected_len: Component count the target buffer defines (3 for a
            position). An orientation is not validated through here directly:
            four components are necessary but not sufficient for a rotation,
            so wxyz callers use
            :func:`coerce_orientation_quaternion`, which adds the norm rule.

    Returns:
        ``(None, None)`` when ``vec`` is ``None`` (omitted - the caller applies
        its own default), ``(floats, None)`` for an acceptable vector - always
        ``expected_len`` components, whether counted from the value's length or
        from the read - or ``(None, error_message)`` for a wrong length, a
        non-numeric element or a ``nan``/``inf`` component.
    """
    if vec is None:
        return None, None
    # One read, through the guard's own: the floats returned here are the ones the
    # domain checks examined, rather than the product of a second read nothing
    # required to agree with the first (#1906).
    floats, err = _read_pose_vector(method, param_name, vec, expected_len)
    if err is not None:
        return None, err
    return floats, None


#: The smallest quaternion norm this library reads as a rotation.
#:
#: A wxyz value below it cannot be turned into one. MuJoCo refuses the all-zero
#: quaternion through its XML door outright ("XML Error: zero quaternion is not
#: allowed"), but the spec-attribute and ``qpos`` doors this package writes
#: through accept it, and a norm it does consider too small to normalize it
#: stores verbatim - measured on mujoco 3.12.0, ``quat="1e-9 0 0 0"`` compiles
#: to ``body_quat = [1e-9, 0, 0, 0]`` while ``quat="2 0 0 0"`` is normalized to
#: identity. Either way the rotation the caller asked for is not the rotation
#: the model carries, so the value is refused at the door instead.
#:
#: The bound is the one ``move_to`` has always applied; it moved here unchanged
#: so every orientation entry point shares it.
MIN_QUATERNION_NORM = 1e-8


def _quaternion_direction_error(method: str, param_name: str, floats: list[float]) -> str | None:
    """The one rule a quaternion adds over a position, shared by both wrappers.

    Args:
        method: Calling method or op name, used in error text.
        param_name: Parameter or field name, used in error text.
        floats: The four wxyz components.

    Returns:
        A message when the components have no direction to recover, else ``None``.
    """
    norm = math.sqrt(sum(component * component for component in floats))
    if norm < MIN_QUATERNION_NORM:
        return f"{method}: '{param_name}' quaternion has ~zero norm; pass a valid [w, x, y, z]."
    return None


def coerce_orientation_quaternion(method: str, param_name: str, quat: Any) -> tuple[list[float] | None, str | None]:
    """Validate an optional wxyz orientation and normalize it to plain floats.

    Args:
        method: Calling method name, used in error text.
        param_name: Parameter name, used in error text.
        quat: The caller's wxyz value, or ``None`` when the parameter was
            omitted.

    Returns:
        ``(None, None)`` when ``quat`` is ``None`` (omitted - the caller applies
        its own default), ``(floats, None)`` for four finite components whose
        norm is a usable direction, or ``(None, error_message)`` otherwise.
    """
    if quat is None:
        return None, None
    # One read, through the guard's own, for the reason :func:`pose_vector_error`
    # defers to it: the floats the direction check examines are the ones the
    # component checks read, and reading them here cannot run the caller's code.
    floats, err = _read_pose_vector(method, param_name, quat, 4)
    if err is not None:
        return None, err
    if (direction_err := _quaternion_direction_error(method, param_name, floats)) is not None:
        return None, direction_err
    return floats, None


def orientation_quaternion_error(method: str, param_name: str, quat: Any) -> str | None:
    """Return an error message if ``quat`` is not a usable wxyz orientation.

    Args:
        method: Calling method or op name, used in error text.
        param_name: Parameter or field name, used in error text.
        quat: The caller's wxyz value, or ``None`` when omitted.

    Returns:
        ``None`` when ``quat`` is acceptable (including omitted), else the
        message naming the method, the parameter and what is wrong.
    """
    floats, err = _read_pose_vector(method, param_name, quat, 4)
    if err is not None:
        return err
    return _quaternion_direction_error(method, param_name, floats)


#: The component counts a 4-component RGBA row can be built from. Alpha is the
#: only component with a meaningful default (opaque), so an RGB triple can be
#: completed without inventing a colour, while any other count cannot.
RGBA_ACCEPTED_LENGTHS: tuple[int, ...] = (3, 4)

#: What those components mean, quoted in the component-count refusal.
RGBA_LAYOUT = "RGB, or RGBA with alpha"


def coerce_rgba(method: str, param_name: str, color: Any) -> tuple[list[float] | None, str | None]:
    """Validate an optional colour and normalize it to 4 RGBA components.

    Args:
        method: Calling method name, used in error text.
        param_name: Parameter name, used in error text.
        color: The caller's colour, or ``None`` when the parameter was omitted.

    Returns:
        ``(None, None)`` when ``color`` is ``None`` (omitted - the caller
        applies its own documented default), ``(rgba, None)`` with exactly 4
        finite floats, or ``(None, error_message)`` for an unusable component
        count, a non-numeric or ``bool`` component, or a ``nan``/``inf`` one.
    """
    if color is None:
        return None, None
    # Asked only as "is this a sized sequence at all", the question this probe
    # owns and cannot raise answering (#1888). It is what refuses a generator,
    # whose components a read would consume before anything could count them. The
    # component count is not taken from it - see below.
    if sequence_length(color) is None:
        return None, f"{method}: '{param_name}' must be a sequence of numbers, got {refusal_container_repr(color)}"
    # One read: the floats quoted by the component-count refusal below are the ones
    # the domain checks examined. They used to come from a second, unguarded read,
    # so a colour that answered the checked read and refused this one raised out of
    # the branch whose purpose is to answer an unusable colour with text (#1906).
    floats, err = _read_finite_vector(method, param_name, color)
    if err is not None:
        return None, err
    # The count is the read's, not the value's ``__len__``. Those were two
    # independent reads and nothing obliged them to agree: a ``__len__`` of 4 over
    # a read yielding 3 skipped the alpha completion below and returned a
    # 3-component rgba under a success result - breaking this function's stated
    # promise of exactly 4 finite floats, and with it the ``color[:3]`` reads the
    # shape builders do - while the refusal named a count from one read beside
    # components from the other (#1909). Counting the list that is returned makes
    # the promise true by construction rather than by the two reads agreeing.
    count = len(floats)
    if count not in RGBA_ACCEPTED_LENGTHS:
        expected = " or ".join(str(n) for n in RGBA_ACCEPTED_LENGTHS)
        return None, (
            f"{method}: '{param_name}' must have exactly {expected} "
            f"component(s) ({RGBA_LAYOUT}), got {count}: {floats}. Pass every "
            f"component - a partial '{param_name}' cannot be applied "
            "without inventing the missing values."
        )
    return (floats if count == 4 else [*floats, 1.0]), None


def coerce_size_vector(method: str, param_name: str, size: Any) -> tuple[list[float] | None, str | None]:
    """Validate an optional object ``size`` and normalize it to plain floats.

    Args:
        method: Calling method name, used in error text.
        param_name: Parameter name, used in error text.
        size: The caller's extent vector, or ``None`` when it was omitted.

    Returns:
        ``(None, None)`` when ``size`` is ``None`` (omitted - the caller applies
        its own documented default), ``(floats, None)`` for an acceptable vector,
        or ``(None, error_message)`` for a non-numeric, ``bool`` or
        ``nan``/``inf`` component, a value that is not a vector at all, or an
        empty vector.
    """
    if size is None:
        return None, None
    # Component classes first, so a value that is not a vector at all is refused
    # in the SAME words the MuJoCo backend already uses for it - one verdict
    # should not have two spellings across backends. The empty-vector refusal is
    # this helper's own, because MuJoCo reaches that case through its per-shape
    # count instead and so states a count the shape needs rather than the
    # omission the caller probably meant.
    floats, err = _read_finite_vector(method, param_name, size)
    if err is not None:
        return None, err
    if sequence_length(size) is None:
        # Reachable only for something iterable but unsized - a generator, which
        # the check above has now consumed, so there is nothing left to store.
        return None, f"{method}: '{param_name}' must be a list/tuple of numbers, got {refusal_container_repr(size)}"
    # Empty means the read produced no component, not that ``__len__`` reported
    # zero: the two are independent reads (#1909), and it is the absence of an
    # extent to write that makes the value unusable. A value whose length reports
    # three and whose read yields nothing has no extent, and one reporting zero
    # whose read yields components has one.
    if not floats:
        return None, (
            f"{method}: '{param_name}' must have at least one component, got an empty "
            f"vector ({refusal_container_repr(size)}). An empty '{param_name}' is a component count, not an "
            f"omission - omit '{param_name}' to take the default extent."
        )
    # The floats the component checks above examined, not a second read of the
    # caller's value (#1906).
    return floats, None


#: Camera names that a backend's render entry points resolve to the FREE camera
#: instead of looking up, by an explicit token check rather than a registry miss.
#: ``None`` and ``""`` mean "no camera was named"; ``"default"`` and ``"free"``
#: are spellings of the free view that :meth:`describe` advertises as always
#: available, so ``render(camera_name="default")`` is a documented call.
#:
#: It lives here, beside :func:`entity_name_error` and :func:`camera_fov_error`,
#: because it is read from two sides that must agree: the render entry points
#: that route it, and the ``add_camera`` guard that refuses it as a *name*. Those
#: two lived as eleven separate copies of the same tuple literal across
#: ``simulation.mujoco.rendering``, ``simulation.mujoco.simulation``,
#: ``simulation.newton.simulation`` and ``simulation.base`` - one of them
#: written in a different order - and the MuJoCo
#: ``add_camera`` had the set in a comment but not in code, which is exactly the
#: drift that made a reserved name accepted there while Newton refused it.
FREE_CAMERA_TOKENS: Final[tuple[str | None, ...]] = (None, "", "default", "free")


def reserved_camera_name_error(method: str, param_name: str, name: Any) -> str | None:
    """Return an error message if ``name`` is a free-camera routing token."""
    if not isinstance(name, str):
        return None
    if name not in FREE_CAMERA_TOKENS:
        return None
    # Through the shared renderer like every other guard here, even though this
    # one has narrowed to ``str`` and could interpolate safely: a ``str``
    # subclass owes its ``__repr__`` nothing, and the rule that no guard renders
    # a caller value directly is worth more than the exception would save.
    rendered = refusal_repr(name)
    return (
        f"{method}: {rendered} is reserved; pick a distinct camera name. "
        f"render/get_frame resolve {param_name}={rendered} to the free camera by an "
        f"explicit token check, so a camera created under it could never be rendered from."
    )


def free_camera_routing_rank(name: Any) -> int:
    """Sort rank that orders the free view behind every real camera.

    Args:
        name: An observation camera key. Membership is the whole rule, so the
            answer agrees with :data:`FREE_CAMERA_TOKENS` for every member
            (including the ``None`` / ``""`` spellings a render call site uses,
            which no observation key can carry). A name that is not a token at
            all is ranked with the real cameras; judging whether it is a usable
            name belongs to the caller's own guard, not to an ordering.

    Returns:
        ``1`` for a free-camera routing token, ``0`` for every other name.
    """
    return 1 if name in FREE_CAMERA_TOKENS else 0


def mounted_camera_pose_error(
    method: str,
    name: str,
    parent_body: str | None,
    position: Any,
    target: Any,
    *,
    start: str | None = None,
) -> str | None:
    """Return an error message when a camera is mounted on a body with no pose of its own.

    Args:
        method: The calling method, opening the message (``"add_camera"``).
        name: The camera's name, so the caller running several knows which.
        parent_body: The mount, or ``None``/``""`` for a free camera - a free
            camera never trips this rule.
        position: The ``position`` argument as passed (``None`` = omitted).
        target: The ``target`` argument as passed (``None`` = omitted).
        start: Optional backend-computed starting pose for THIS body, appended
            verbatim; when absent a generic hint closes the message.

    Returns:
        ``None`` when the camera is free or carries a pose of its own (either
        of ``position`` / ``target`` supplied), else the refusal.
    """
    if not parent_body:
        return None
    if position is not None or target is not None:
        return None
    tail = start or (
        "A few centimetres off the body's origin, looking along its approach axis, is the "
        "usual start; render the camera and adjust."
    )
    return (
        f"{method}: camera {name!r} is mounted on {parent_body!r}, so position and target are "
        f"read in that body's LOCAL frame - and both were omitted. The free-camera defaults "
        f"(position [1, 1, 1] looking at [0, 0, 0]) would place it 1.73 m from the body looking "
        f"back at it: a third-person view that rides along with the arm, not a wrist view. Pass "
        f"both in the body frame. {tail}"
    )


def camera_fov_error(method: str, param_name: str, value: Any) -> str | None:
    """Return an error message if ``value`` is not a usable camera field of view."""

    def not_a_number() -> str:
        return f"{method}: '{param_name}' must be a finite number in degrees, got {refusal_repr(value)}."

    def outside_interval() -> str:
        return f"{method}: '{param_name}' must be in the open interval (0, 180) degrees, got {refusal_str(value)}."

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return not_a_number()
    if _beyond_float_range(value):
        # This guard is the one member of the family that needs no new text. A
        # magnitude past the float64 range exceeds 180 by three hundred orders
        # of magnitude, so "outside the open interval" is already a true
        # statement about it - and no comparison is needed to establish that,
        # which is why the overflow itself is the whole test.
        return outside_interval()
    try:
        numeric = float(value)
    except Exception:
        return not_a_number()
    if not math.isfinite(numeric):
        return not_a_number()
    if not (0.0 < numeric < 180.0):
        return outside_interval()
    return None


def entity_name_error(method: str, param_name: str, name: Any) -> str | None:
    """Return an error message if ``name`` cannot address the entity it names.

    Args:
        method: Calling method name, used in error text.
        param_name: Parameter name, used in error text.
        name: The caller's value.

    Returns:
        ``None`` when ``name`` can address the entity it creates, otherwise the
        error message to report through the structured tool-result dict.
    """
    if not isinstance(name, str):
        return (
            f"{method}: '{param_name}' must be a non-empty string, got {refusal_repr(name)} "
            f"({type(name).__name__}); an entity is addressed by name and every "
            "agent-tool call carries that name as a string."
        )
    if not name:
        return (
            f"{method}: '{param_name}' must be a non-empty string, got ''; an empty "
            "name is the backend's own sentinel for an unnamed entity, so the entity "
            "created under it could not be addressed afterwards."
        )
    if "\x00" in name:
        return (
            f"{method}: '{param_name}' must not contain a NUL character, got {refusal_repr(name)}; "
            "the compiled model reads a name only up to the first NUL, so the registry "
            "and the model would disagree about the entity's name."
        )
    return None


def camera_name_error(method: str, param_name: str, name: Any, *, routes_free_camera_tokens: bool) -> str | None:
    """Return an error message if ``name`` cannot address the camera it claims.

    Args:
        method: The calling method, for the message prefix (e.g. ``"add_camera"``).
        param_name: The parameter being validated, for the message.
        name: The claimed camera name. Anything at all; a value that is not a
            ``str`` is refused by the first guard.
        routes_free_camera_tokens: Whether this backend's render entry points
            resolve :data:`FREE_CAMERA_TOKENS` to the free camera. Only a backend
            that routes them may refuse them as names - the Isaac backend's
            ``get_frame`` looks a name up directly, so ``"default"`` there is an
            ordinary camera name and is that backend's documented signature
            default. Passing the flag makes that divergence a stated property of
            the call rather than a guard one site happens to omit.

    Returns:
        The first refusal in the documented order, or ``None`` when *name* can
        address a camera on this backend *and* key that camera's frames at the
        consumers which read the name as structure.
    """
    if (err := entity_name_error(method, param_name, name)) is not None:
        return err
    if routes_free_camera_tokens and (err := reserved_camera_name_error(method, param_name, name)) is not None:
        return err
    return scoped_camera_name_error(method, param_name, name)


#: The alphabet a camera token is written in: letters, digits, ``_`` and ``-``,
#: opening on a letter or a digit. Both camera-name shapes below are built from
#: this one string, so the door that takes a bare token and the door that takes a
#: scoped one cannot come to accept different alphabets.
_CAMERA_TOKEN_ALPHABET: Final = r"[A-Za-z0-9][A-Za-z0-9_-]*"

#: What a camera's name in a ``cameras`` mapping may be: one bare token. Every
#: consumer keys the camera's frames by this name, and each of them reserves
#: punctuation of its own - see :func:`camera_token_error`.
_CAMERA_TOKEN: Final = re.compile(rf"\A{_CAMERA_TOKEN_ALPHABET}\Z")

#: What a camera's name in a sim scene may be: a bare token, optionally scoped to
#: one robot as ``<robot>/<camera>``. One optional level, because that is the one
#: namespace ``add_robot`` gives what it spawns and the only one the mesh strips -
#: see :func:`scoped_camera_name_error`.
_SCOPED_CAMERA_NAME: Final = re.compile(rf"\A{_CAMERA_TOKEN_ALPHABET}(?:/{_CAMERA_TOKEN_ALPHABET})?\Z")


def camera_token_error(method: str, param_name: str, name: Any) -> str | None:
    """Return an error message if ``name`` cannot key the camera it names.

    Args:
        method: The surface being called, for the message prefix (e.g.
            ``"Robot(cameras=...)"``).
        param_name: What is being named, for the message (e.g.
            ``"camera name"``).
        name: The claimed camera name. Anything at all; a value that cannot be a
            key of any kind is refused first by :func:`entity_name_error`.

    Returns:
        An error message naming the value and the alphabet, or ``None`` when
        *name* is a bare token.
    """
    if (err := entity_name_error(method, param_name, name)) is not None:
        return err
    if _CAMERA_TOKEN.match(name) is not None:
        return None
    rendered = refusal_repr(name)
    return (
        f"{method}: {param_name}={rendered} is not a bare token. A camera's name is the "
        "identity every consumer keys its frames by, and each reserves punctuation of its "
        "own: the mesh publishes them on 'strands/<peer_id>/camera/<name>', where '/' adds a "
        "topic level and '*' is a wildcard a put is routed by; the S3 offload joins it into "
        "the object key, where '..' walks out of the peer's prefix; a recording writes it as "
        "the 'observation.images.<name>' dataset feature key; and lerobot parses "
        "--robot.cameras as a nested dict, where ',', ':', '{', '}', '=' and whitespace are "
        "structure. Use letters, digits, '_' or '-'."
    )


def scoped_camera_name_error(method: str, param_name: str, name: Any) -> str | None:
    """Return an error message if a sim camera's *name* cannot key its own frames.

    Args:
        method: The surface being called, for the message prefix (e.g.
            ``"add_camera"``).
        param_name: The parameter being validated, for the message.
        name: The claimed camera name. Anything at all; a value that cannot be a
            registry key of any kind is refused first by
            :func:`entity_name_error`.

    Returns:
        An error message naming the value and the alphabet, or ``None`` when
        *name* is a camera token optionally scoped to one robot.
    """
    if (err := entity_name_error(method, param_name, name)) is not None:
        return err
    if _SCOPED_CAMERA_NAME.match(name) is not None:
        return None
    rendered = refusal_repr(name)
    return (
        f"{method}: {param_name}={rendered} cannot key a camera's frames. A sim camera's name "
        "is a bare token of letters, digits, '_' or '-', opening on a letter or a digit, "
        "optionally scoped to one robot as '<robot>/<camera>' (the namespace add_robot gives "
        "what it spawns, and the one the mesh strips). Every other character is structure to "
        "a consumer that keys frames by this name: the mesh publishes them on "
        "'strands/<peer_id>/camera/<name>', where a further '/' adds a topic level no "
        "'strands/*/camera/*' subscription matches and '*' / '**' are wildcards a put is "
        "routed by intersection; the S3 offload joins it into the object key, where '..' "
        "walks out of the peer's prefix; and a recording writes it as the "
        "'observation.images.<name>' dataset feature key."
    )


def published_string_error(value: Any, param: str, context: str) -> str | None:
    """Return an error message if a field published as a string did not arrive as one.

    Args:
        value: The caller's value.
        param: Parameter name, used in error text.
        context: Calling context, used in error text (e.g. ``"Action 'add_robot'"``).

    Returns:
        ``None`` when *value* is a string, otherwise the error message to report
        through the structured tool-result dict.
    """
    if isinstance(value, str):
        return None
    return (
        f"{context}: '{param}' must be a string, got {refusal_repr(value)} "
        f"({type(value).__name__}); this tool publishes '{param}' as a string and "
        "every agent-tool call carries it as one."
    )


def stale_output_dir_is_clearable(output_dir: str) -> bool:
    """True when ``output_dir`` exists and holds nothing, so clearing it is free.

    Args:
        output_dir: Path a run is about to write checkpoints into.

    Returns:
        ``True`` only when the path is an existing directory with no entries.
        ``False`` for a path that does not exist (there is nothing to clear), is
        not a directory, or holds anything at all.
    """
    path = Path(output_dir)
    if not path.is_dir():
        return False
    return not any(path.iterdir())


def _episode_indices(value: Any) -> list[int] | None:
    """Episode indices a passthrough value carries, or ``None`` for no restriction."""
    if isinstance(value, str):
        try:
            from draccus import cfgparsing

            value = cfgparsing.parse_string(value)
        except Exception:  # noqa: BLE001
            # Both stages are third-party and raise from disjoint hierarchies
            # (ImportError with no lerobot installed, yaml.YAMLError for the
            # scalar parse). Nothing is swallowed that a run could proceed on:
            # the same text is refused again by the config field it targets.
            return None
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return None
    try:
        elements = iter(value)
    except Exception:  # noqa: BLE001
        return None
    indices: list[int] = []
    while True:
        try:
            entry = next(elements)
        except StopIteration:
            return indices
        except Exception:  # noqa: BLE001
            return None
        if type(entry) is not int:
            return None
        indices.append(entry)


def effective_episode_count(total_episodes: int, episodes: Any, exclude_episodes: Any = None) -> int:
    """Episodes a run will actually train and validate over.

    Args:
        total_episodes: What the dataset's ``meta/info.json`` declares.
        episodes: The allowlist as the caller wrote it in their passthrough - a
            sequence of indices, the text form lerobot's CLI decoder accepts, or
            ``None`` for every episode.
        exclude_episodes: The exclusion list, same accepted spellings.

    Returns:
        The size of the subset the run will carry, or ``total_episodes`` when no
        usable restriction was asked for.
    """
    chosen = _episode_indices(episodes)
    excluded = _episode_indices(exclude_episodes)
    if chosen is None and not excluded:
        return total_episodes
    try:
        from lerobot.datasets.utils import resolve_episode_indices
    except ImportError:
        return total_episodes if chosen is None else len(chosen)
    resolved = resolve_episode_indices(chosen, total_episodes, excluded)
    return total_episodes if resolved is None else len(resolved)


def episode_subset_budget_error(
    val_episodes: int,
    total_episodes: int,
    effective_episodes: int,
    context: str,
    *,
    passthrough_param: str,
) -> str | None:
    """Error text when a holdout does not fit the SUBSET a passthrough left.

    Args:
        val_episodes: The requested held-out episode count.
        total_episodes: What the dataset's ``meta/info.json`` declares.
        effective_episodes: What :func:`effective_episode_count` measured.
        context: Caller label the message is prefixed with.
        passthrough_param: Name of the caller's own raw passthrough parameter,
            interpolated into the remedy. Required rather than defaulted for the
            reason :func:`validation_split_error` carries: the surfaces disagree
            (``extra_flags`` on the tool, ``extra`` on :class:`TrainSpec`), so a
            default would name a keyword one of them does not accept.

    Returns:
        The error text, or ``None`` when the holdout fits - and when no subset
        narrowed the dataset, which is the caller's own whole-dataset refusal to
        report because it already names the header count.
    """
    if effective_episodes >= total_episodes or val_episodes < effective_episodes:
        return None
    return (
        f"{context}: val_episodes={val_episodes} cannot be reserved from the "
        f"{effective_episodes} episode(s) {passthrough_param}['dataset.episodes'/"
        f"'dataset.exclude_episodes'] selects, out of {total_episodes} in the dataset. "
        "lerobot sizes the validation split against the episodes the dataset was built "
        "from, not against the header count, so the subset is the budget. Either reserve "
        f"fewer than {effective_episodes}, widen the subset, or pass the fraction directly, "
        f"e.g. {passthrough_param}={{'dataset.eval_split': 0.1, 'eval_steps': 1000}}."
    )


def validation_split_fraction(val_episodes: int, total_episodes: int) -> float:
    """``dataset.eval_split`` that holds out exactly ``val_episodes`` episodes.

    Args:
        val_episodes: Number of episodes to hold out. Must be positive and
            smaller than ``total_episodes``; callers validate that themselves so
            they can name their own parameter in the error.
        total_episodes: Episode count of the dataset being split.

    Returns:
        The fraction to pass as lerobot's ``--dataset.eval_split``.
    """
    return (val_episodes - 0.5) / total_episodes


def validation_split_error(val_episodes: int, total_tasks: Any, context: str, *, passthrough_param: str) -> str | None:
    """Error text when a global episode COUNT cannot be honored as a split.

    Args:
        val_episodes: The requested held-out episode count, for the message.
        total_tasks: The value the dataset's ``meta/info.json`` carried under
            ``total_tasks``, verbatim, or ``None`` when there is no header.
        context: Caller label the message is prefixed with.
        passthrough_param: Name of the caller's own raw-flag passthrough
            parameter, interpolated into the remedy. Required rather than
            defaulted because the surfaces disagree: the ``lerobot_train`` tool
            spells it ``extra_flags`` while :class:`TrainSpec` (and the
            ``train_policy`` tool) spell it ``extra``, so a default would name a
            keyword one of them does not accept - the reader would apply the
            remedy verbatim and get a ``TypeError``.

    Returns:
        The error text, or None when the count can be honored exactly.
    """
    if total_tasks is None:
        return None
    declared = declared_count(total_tasks)
    if declared is None:
        return (
            f"{context}: val_episodes={val_episodes} cannot be checked against a dataset whose "
            f"meta/info.json declares total_tasks={refusal_repr(total_tasks)}, which is not a "
            "task count. Whether one global count is expressible depends on how many tasks the "
            "dataset holds - lerobot holds out ceil(episodes_in_task * eval_split) from every "
            "task - so a header declaring no usable count is neither single-task nor multi-task, "
            "and reading it as single-task is what let a three-task dataset spelling its count "
            "3.0 past this guard. Repair meta/info.json, or pass the fraction directly, e.g. "
            f"{passthrough_param}={{'dataset.eval_split': 0.1, 'eval_steps': 1000}}."
        )
    if declared <= 1:
        return None
    return (
        f"{context}: val_episodes={val_episodes} cannot be reserved exactly on a "
        f"dataset with {refusal_str(declared)} tasks. A validation split is a per-task "
        "fraction in lerobot (it holds out ceil(episodes_in_task * eval_split) "
        "from every task), so a single global count is not expressible: the "
        "ceiling would be applied once per task. Pass the fraction directly, "
        f"e.g. {passthrough_param}={{'dataset.eval_split': 0.1, 'eval_steps': 1000}}, "
        "and the split will hold out a tenth of each task."
    )


def optional_callable_error(value: Any, param: str, context: str) -> str | None:
    """Return an error message unless ``value`` is callable or ``None``.

    Args:
        value: The optional callback supplied by the caller.
        param: Parameter name, used in the refusal text.
        context: Calling surface, used as the message prefix.

    Returns:
        ``None`` for ``None`` or a callable, otherwise the refusal text.
    """
    if value is None or callable(value):
        return None
    return f"{context}: {param} must be callable or None, got {refusal_repr(value)}."


def teleoperator_contract_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` cannot serve as a teleoperator.

    Args:
        value: The caller-supplied teleoperator.
        param: The parameter it came from, used in the message.
        context: Message prefix identifying the surface that received it -
            normally the public method name.

    Returns:
        An error message, or ``None`` when the device can be polled.
    """
    if callable(getattr(value, "get_action", None)):
        return None
    return (
        f"{context}: {param} must expose a callable get_action(), got "
        f"{refusal_repr(value)} - it does not satisfy the teleoperator contract. "
        "The loop that drives it polls get_action() once per tick on a background "
        "thread, so a device without one yields a stream that reports running "
        "while putting no frame on the wire."
    )


def boolean_flag_error(value: Any, param: str, context: str) -> str | None:
    """Return an error message unless *value* is a python or numpy boolean.

    Args:
        value: The flag as supplied.
        param: Parameter name, for the message.
        context: Caller label the message is prefixed with.

    Returns:
        The error text, or None when *value* is a boolean and can be honoured.
    """
    if is_boolean(value):
        return None
    return (
        f"{context}: {param} must be a boolean, got {refusal_repr(value)}. "
        "It selects a posture rather than scaling a quantity, so it is checked "
        "rather than parsed - a truthy spelling of off, such as 'false', would "
        "otherwise select the opposite posture from the one it reads as."
    )


def partial_construction_repr(obj: object) -> str:
    """Describe an object whose ``__init__`` did not finish, naming no attribute.

    Args:
        obj: The partially constructed object being rendered.

    Returns:
        ``"<ClassName>(partially constructed, id=0x...)"``.
    """
    return f"{type(obj).__name__}(partially constructed, id=0x{id(obj):x})"
