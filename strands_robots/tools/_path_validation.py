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
    """Return blocked prefixes for the current platform."""
    if sys.platform == "win32":
        return _WINDOWS_BLOCKED_PREFIXES
    elif sys.platform == "darwin":
        return _LINUX_BLOCKED_PREFIXES + _MACOS_BLOCKED_PREFIXES
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
            raise ValueError(
                f"{label} resolves to a protected system directory ({prefix}): {resolved}"
            )

    return resolved
