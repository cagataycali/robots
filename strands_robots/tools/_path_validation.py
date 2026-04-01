"""Shared path validation utilities for tools that write to the filesystem.

Provides a consistent ``validate_save_path`` helper that all tool modules
can import to reject dangerous path values before any I/O occurs.
"""

import os
import re

# Characters that have no business appearing in file paths supplied by tool callers.
_DANGEROUS_CHARS = re.compile(r"[\x00]")


def validate_save_path(path: str, *, label: str = "path") -> str:
    """Validate and resolve a user-supplied file-system path.

    Rejects paths that contain:
    - Null bytes (``\\x00``)
    - ``..`` traversal components

    Then resolves the path to an absolute form via ``os.path.realpath``
    and ensures it does **not** escape into well-known sensitive directories.

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

    # Block writes into known sensitive system directories
    _BLOCKED_PREFIXES = (
        "/etc/",
        "/usr/",
        "/bin/",
        "/sbin/",
        "/boot/",
        "/dev/",
        "/proc/",
        "/sys/",
        "/var/spool/cron",
        "/var/spool/at",
    )
    for prefix in _BLOCKED_PREFIXES:
        if resolved.startswith(prefix):
            raise ValueError(f"{label} resolves to a protected system directory ({prefix}): {resolved}")

    return resolved
