"""Serialize the ``robot_descriptions`` clone that importing a description triggers.

Importing ``robot_descriptions.<name>_mj_description`` clones an upstream asset
repository into a shared cache directory on first use, and 40-odd descriptions
name one ``mujoco_menagerie``. Upstream's ``clone_to_directory`` tests that
directory for a usable clone and then creates it, holding no lock, so two
callers inside that window both clone into one directory and the loser's ``git``
lands in a tree the winner is already building. Four shapes were measured, all
from the one unprotected window: ``cannot copy '.../hooks/sendemail-validate.sample'
... File exists`` from ``git init``, ``remote origin already exists``, ``could not
lock config file .git/config``, ``reference is not a tree`` from a checkout racing
a sibling's fetch, and ``Unable to create '.../.git/shallow.lock': File exists``
from two fetches of one revision. The raiser is then reported as the robot's own download
failure, when it is the other caller's clone.

:func:`import_description` owns that window wherever the package triggers a
clone - :mod:`strands_robots.assets.download` and
:mod:`strands_robots.registry.discovery`: the first caller clones while the rest
wait on an exclusive lock, and each of them then finds the finished clone. The
lock lives in the cache directory it guards, so a process that redirects
``ROBOT_DESCRIPTIONS_CACHE`` serializes against the others using that cache and
not against unrelated ones, and the kernel releases it if the holder dies.

The guard is best-effort by design, because a clone is worth more than the
serialization of it: where ``fcntl`` is absent (Windows) or the cache directory
cannot hold a lock file, the import proceeds unguarded.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - POSIX-only; the supported surface is POSIX
    _HAS_FCNTL = False

logger = logging.getLogger(__name__)

#: Environment variable upstream reads to relocate the cache, and its default.
CACHE_ENV = "ROBOT_DESCRIPTIONS_CACHE"
DEFAULT_CACHE = "~/.cache/robot_descriptions"

#: Lock file, kept inside the cache directory whose clones it serializes.
LOCK_NAME = ".strands-clone.lock"


def cache_dir() -> Path:
    """Return the directory ``robot_descriptions`` clones into.

    Read on every call, exactly as upstream's cache reads it, so a process that
    redirects the cache is guarded by the lock file of the cache it uses.
    """
    return Path(os.path.expanduser(os.environ.get(CACHE_ENV, DEFAULT_CACHE)))


@contextmanager
def clone_lock() -> Iterator[Path | None]:
    """Hold an exclusive lock on the description cache directory for the block.

    Yields:
        The lock file being held, or ``None`` when there is no lock to take -
        no ``fcntl``, or a cache directory that cannot hold the file. The block
        runs either way.
    """
    if not _HAS_FCNTL:
        yield None
        return
    lock_file = cache_dir() / LOCK_NAME
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_file.open("wb")
    except OSError as exc:
        logger.debug("description cache lock unavailable at %s: %s", lock_file, exc)
        yield None
        return
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError as exc:  # a filesystem without flock, e.g. some network mounts
            logger.debug("description cache lock refused at %s: %s", lock_file, exc)
            yield None
            return
        try:
            yield lock_file
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def import_description(module_name: str) -> ModuleType:
    """Import ``robot_descriptions.<module_name>`` with the cache lock held.

    The one seam through which the package triggers a description clone: no
    other module in the package calls ``importlib.import_module`` on a
    ``robot_descriptions`` module, and a scan in the test tree keeps it that way.

    Args:
        module_name: A description module name, e.g. ``go2_mj_description``.
            Callers validate the name before it reaches the import machinery.

    Returns:
        The imported description module.

    Raises:
        ImportError: As :func:`importlib.import_module` does - an unknown
            description, or a clone that failed.
    """
    with clone_lock():
        return importlib.import_module(f"robot_descriptions.{module_name}")
