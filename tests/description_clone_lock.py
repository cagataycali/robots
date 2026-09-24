"""Serialize the ``robot_descriptions`` clone across concurrent test workers.

Every ``*_mj_description`` module clones ONE repository into ONE shared cache
directory at import time -- 40-odd of them name ``mujoco_menagerie`` -- and
``robot_descriptions._cache.clone_to_directory`` takes no lock: it tests the
target directory for a usable clone and then creates it. A distributed run
imports those modules in every worker, because each worker collects the whole
tree, so two workers reach that window together on a cold cache. The loser's
``git`` lands in a tree the winner is already building and raises, which during
collection ERRORs the whole session out with a message about the cache rather
than about the code under test. Measured on a cold cache at ``-n 2``: ``fatal:
cannot copy '.../hooks/sendemail-validate.sample' ... File exists`` from ``git
init``, ``error: remote origin already exists``, and ``error: could not lock
config file .git/config: File exists``.

:mod:`strands_robots._description_cache` owns that window wherever the package
triggers a clone, but a test module reaches a description through its own
``pytest.importorskip("robot_descriptions.<name>")``, which no package seam sees.
:func:`serialize_description_clones` closes the remaining half by wrapping
``clone_to_cache`` itself in *that* module's lock -- the same lock file, so a
collecting worker and a production caller in the same cache directory wait for
each other rather than each holding a lock of its own.
"""

from __future__ import annotations

import functools

#: Marker set on an installed wrapper, so a second install is a no-op rather
#: than another layer of the same lock.
INSTALLED = "__clone_lock_installed__"


def serialize_description_clones() -> bool:
    """Route ``robot_descriptions`` clones through the package's cache lock.

    Returns:
        Whether the lock is in place. ``False`` where there is nothing to guard:
        ``robot_descriptions`` is not installed, or the platform has no
        ``flock`` (Windows), where a distributed run is not what CI grades.
    """
    try:
        from robot_descriptions import _cache  # type: ignore[import-not-found]

        from strands_robots._description_cache import _HAS_FCNTL, clone_lock
    except ImportError:
        return False

    if not _HAS_FCNTL:
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
