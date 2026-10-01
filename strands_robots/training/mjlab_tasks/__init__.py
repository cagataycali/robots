"""mjlab RL tasks defined on strands-robots' own assets.

Importing this package registers the tasks with mjlab's task registry so that
``python -m mjlab.scripts.train Strands-Reach-SO101`` (or
``create_trainer("rsl_rl")``) can find them. mjlab is optional: the import is a
no-op without it.
"""

from __future__ import annotations

import importlib.util

__all__ = ["register_all", "TASK_IDS"]

TASK_IDS = ("Strands-Reach-SO101",)


def register_all() -> list[str]:
    """Register every strands-robots mjlab task once; returns the ids."""
    from strands_robots.training.mjlab_tasks.so101_reach import register

    return [register()]


# Register on import only when the optional extra is present. An explicit
# presence check rather than a swallowed ImportError: without mjlab there is
# no registry to register into (the rsl_rl trainer reports the missing extra
# with its install hint when it is actually used), while an mjlab that IS
# installed but fails to import is a real error that must surface here.
if importlib.util.find_spec("mjlab") is not None:  # pragma: no cover - exercised by the mjlab tests
    register_all()
