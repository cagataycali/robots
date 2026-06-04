"""Shared path validation utilities for tools that write to the filesystem.

Provides a consistent ``validate_save_path`` helper that all tool modules
can import to reject dangerous path values before any I/O occurs.

Cross-platform: blocks sensitive directories on Linux, macOS, and Windows.
"""

import os
import re
import sys

# Characters that have no business appearing in file paths supplied by tool callers.
_DANGEROUS_CHARS = re.compile(r"[\x00]")

# Well-known sensitive system directories that tool callers should never write to.
# Each entry ends with '/' (or '\' on Windows) so ``str.startswith`` only matches
# paths *inside* the directory, not unrelated paths that share a common prefix
# (e.g. "/var/spool/crondata" should NOT match "/var/spool/cron/").
_LINUX_BLOCKED_PREFIXES = (
    "/etc/",
    "/usr/",
    "/bin/",
    "/sbin/",
    "/boot/",
    "/dev/",
    "/proc/",
    "/sys/",
    "/var/spool/cron/",
    "/var/spool/at/",
)

_MACOS_BLOCKED_PREFIXES = (
    "/System/",
    "/Library/LaunchDaemons/",
    "/Library/LaunchAgents/",
)

_WINDOWS_BLOCKED_PREFIXES = (
    "C:\\Windows\\",
    "C:\\Program Files\\",
    "C:\\Program Files (x86)\\",
)


def _get_blocked_prefixes() -> tuple[str, ...]:
    """Return blocked prefixes for the current platform.

    On macOS, many system directories (``/etc``, ``/var``, ``/tmp``) are
    symlinks into ``/private/``. Since :func:`validate_save_path` compares
    against ``os.path.realpath`` output, we must include the ``/private/``-
    prefixed variants so that ``/etc/passwd`` (which resolves to
    ``/private/etc/passwd``) is still rejected.
    """
    if sys.platform == "win32":
        return _WINDOWS_BLOCKED_PREFIXES
    elif sys.platform == "darwin":
        private_variants = tuple("/private" + p for p in _LINUX_BLOCKED_PREFIXES)
        return _LINUX_BLOCKED_PREFIXES + private_variants + _MACOS_BLOCKED_PREFIXES
    else:
        return _LINUX_BLOCKED_PREFIXES


BLOCKED_PREFIXES = _get_blocked_prefixes()


def validate_save_path(path: str, *, label: str = "path") -> str:
    """Validate and resolve a user-supplied file-system path.

    Rejects paths that contain:
    - Null bytes (``\\x00``)
    - ``..`` traversal components

    Then resolves the path to an absolute form via ``os.path.realpath``
    and ensures it does **not** escape into well-known sensitive directories.

    Cross-platform: validates against OS-specific blocked directories on
    Linux, macOS, and Windows.

    Args:
        path: The raw path string from the tool caller.
        label: A human-readable name for error messages (e.g. ``"save_path"``).

    Returns:
        The validated, resolved absolute path.

    Raises:
        ValueError: If the path fails any validation check.
    """
    if not path:
        raise ValueError(f"{label} must not be empty")

    if _DANGEROUS_CHARS.search(path):
        raise ValueError(f"{label} contains invalid characters")

    # Reject explicit '..' components (before resolution to catch intent)
    parts = path.replace("\\", "/").split("/")
    if ".." in parts:
        raise ValueError(f"{label} must not contain '..' path traversal components")

    # Resolve to absolute path (follows symlinks)
    resolved = os.path.realpath(os.path.expanduser(path))

    # Ensure resolved path ends with separator for directory-prefix matching
    sep = "\\" if sys.platform == "win32" else "/"
    check_path = resolved if resolved.endswith(sep) else resolved + sep

    for prefix in BLOCKED_PREFIXES:
        if check_path.startswith(prefix):
            raise ValueError(f"{label} resolves to a protected system directory ({prefix}): {resolved}")

    return resolved


# Hostname / IP allowlist: letters, digits, dots, hyphens, colons (IPv6),
# underscores. No shell metacharacters, spaces, slashes, or pipes. Shared by
# the qwen_vla_inference / qwen_vla_train tools so connect-target validation is
# consistent (PR #92 LLM-input-safety baseline).
_HOST_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


def validate_host(host: str, *, label: str = "host") -> str:
    """Validate a hostname / IP supplied by a tool caller.

    Args:
        host: The host string to validate.
        label: Field name for error messages.

    Returns:
        The validated host unchanged.

    Raises:
        ValueError: If the host is empty or contains characters outside the
            allowlist (letters, digits, '.', '-', ':', '_').
    """
    if not host:
        raise ValueError(f"{label} must not be empty")
    if not _HOST_RE.match(host):
        raise ValueError(f"{label} {host!r} contains invalid characters (allowed: letters, digits, '.', '-', ':', '_')")
    return host


# Subprocess-executable allowlist (PR #92 LLM-input-safety baseline). Tool
# parameters that flow into ``subprocess.Popen``'s argv[0] must be matched
# against this regex before launch; a malicious or LLM-coerced caller could
# otherwise pick an arbitrary binary as the entrypoint even though
# ``shell=False`` neutralises shell injection.
#
# The allowlist matches the documented Qwen-VLA / GR00T entrypoints only:
#   * ``python``, ``python3``, ``python3.<minor>``
#   * ``uv``
#   * absolute paths whose basename satisfies the same rule
#     (e.g. ``/usr/bin/python3.12``, ``/opt/venv/bin/uv``)
_EXECUTABLE_BASENAME_RE = re.compile(r"^(?:python(?:3(?:\.\d+)?)?|uv)$")
_EXECUTABLE_PATH_RE = re.compile(r"^[A-Za-z0-9_./-]+$")


def validate_executable(executable: str, *, label: str = "executable") -> str:
    """Validate a subprocess-entrypoint executable supplied by a tool caller.

    Accepts only Python-launcher style entrypoints: bare ``python`` /
    ``python3`` / ``python3.<minor>`` / ``uv``, or absolute paths whose
    basename matches the same rule (e.g. ``/usr/bin/python3.12``).

    Args:
        executable: The argv[0] value to validate.
        label: Field name for error messages.

    Returns:
        The validated executable unchanged.

    Raises:
        ValueError: If the executable is empty, contains characters outside
            the path allowlist, or its basename is not on the allowlist.
    """
    if not executable:
        raise ValueError(f"{label} must not be empty")
    if not _EXECUTABLE_PATH_RE.match(executable):
        raise ValueError(
            f"{label} {executable!r} contains invalid characters (allowed: letters, digits, '.', '_', '-', '/')"
        )
    basename = executable.rsplit("/", 1)[-1]
    if not _EXECUTABLE_BASENAME_RE.match(basename):
        raise ValueError(
            f"{label} {executable!r} is not on the entrypoint allowlist "
            "(must be python / python3 / python3.<minor> / uv, optionally "
            "as an absolute path)"
        )
    return executable
