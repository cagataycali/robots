"""mjlab RL tasks defined on strands-robots' own assets.

Importing this package registers the tasks with mjlab's task registry so that
``python -m mjlab.scripts.train Strands-Reach-SO101`` (or
``create_trainer("rsl_rl")``) can find them. mjlab is optional: the import is a
no-op without it.
"""

from __future__ import annotations

__all__ = ["register_all", "TASK_IDS"]

TASK_IDS = ("Strands-Reach-SO101",)


def register_all() -> list[str]:
    """Register every strands-robots mjlab task once; returns the ids."""
    from strands_robots.training.mjlab_tasks.so101_reach import register

    return [register()]


try:  # pragma: no cover - exercised by the mjlab tests
    register_all()
except ImportError:
    pass
