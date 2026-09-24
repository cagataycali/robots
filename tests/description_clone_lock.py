"""Serialize the ``robot_descriptions`` clone across concurrent test workers.

Every ``*_mj_description`` module clones ONE repository into ONE shared cache
directory at import time -- 40-odd of them name ``mujoco_menagerie`` -- and
``robot_descriptions._cache.clone_to_directory`` takes no lock: it tests the
target directory for a usable clone and then creates it. A distributed run
imports those modules in every worker, because each worker collects the whole
tree, so two workers reach that window together on a cold cache. The loser's
``git`` lands in a tree the winner is already building and raises, which during
collection ERRORs the whole session out with a message about the cache rather
than about the code under test.

Three shapes were measured for the loser, all from one unprotected window:
``fatal: cannot copy '.../hooks/sendemail-validate.sample' ... File exists`` from
``git init``, ``error: remote origin already exists``, and ``error: could not
lock config file .git/config: File exists``.

:func:`serialize_description_clones` wraps ``clone_to_cache`` in a lock on the
cache directory, so the first caller clones while the rest wait and each of them
then finds the finished clone. The lock covers the clone only, is taken per cache
directory (a run that redirects ``ROBOT_DESCRIPTIONS_CACHE`` locks its own), and
is released by the kernel if the holder dies.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

#: Marker set on an installed wrapper, so a second install is a no-op rather
#: than another layer of the same lock.
INSTALLED = "__clone_lock_installed__"

#: Lock file, kept inside the cache directory it guards.
LOCK_NAME = ".clone.lock"


def cache_dir() -> Path:
    """Return the directory ``robot_descriptions`` clones into.

    Reads ``ROBOT_DESCRIPTIONS_CACHE`` on every call, exactly as the upstream
    cache does, so a test that redirects it is guarded by its own lock file.
    """
    return Path(os.path.expanduser(os.environ.get("ROBOT_DESCRIPTIONS_CACHE", "~/.cache/robot_descriptions")))


@contextmanager
def clone_lock() -> Iterator[Path]:
    """Hold an exclusive lock on the description cache directory.

    Yields:
        The lock file being held.
    """
    import fcntl

    directory = cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    lock_file = directory / LOCK_NAME
    with lock_file.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield lock_file
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def serialize_description_clones() -> bool:
    """Route ``robot_descriptions`` clones through :func:`clone_lock`.

    Returns:
        Whether the lock is in place. ``False`` where there is nothing to guard:
        ``robot_descriptions`` is not installed, or the platform has no
        ``flock`` (Windows), where a distributed run is not what CI grades.
    """
    try:
        import fcntl  # noqa: F401  # absent on Windows

        from robot_descriptions import _cache  # type: ignore[import-not-found]
    except ImportError:
        return False

    if getattr(_cache.clone_to_cache, INSTALLED, False):
        return True

    unguarded = _cache.clone_to_cache

    @functools.wraps(unguarded)
    def clone_to_cache(description_name: str, commit: str | None = None) -> str:
        with clone_lock():
            return str(unguarded(description_name, commit))

    setattr(clone_to_cache, INSTALLED, True)
    _cache.clone_to_cache = clone_to_cache
    return True
