"""Isolation fixtures shared by every ``tests/registry`` module.

The registry read API (``list_robots``, ``format_robot_table``,
``resolve_name``, ``get_robot``) merges the user-local overlay
(``$STRANDS_BASE_DIR/user_robots.json``, defaulting to
``~/.strands_robots/user_robots.json``) on top of the package
``robots.json`` - see ``loader._merge_user_robots``. Registering a custom
robot is an explicitly supported, documented workflow, so a developer's
machine may legitimately hold user robots whose descriptions contain
non-ASCII text or extra entries.

Without isolation those host robots leak into assertions like
``format_robot_table().isascii()`` or exact registry counts, making the
suite pass on clean CI yet fail locally. Worse, any test that calls
``register_robot`` / ``unregister_robot`` would mutate the real user home.

This autouse fixture repoints both ``STRANDS_BASE_DIR`` (user registry +
metadata) and ``STRANDS_ASSETS_DIR`` (asset cache) at per-test temp dirs and
invalidates the loader cache on entry and exit, so every registry test sees
the package registry plus only the robots it registers itself.
"""

from __future__ import annotations

import pytest

from strands_robots.registry.user_registry import _invalidate_cache


@pytest.fixture(autouse=True)
def _isolate_user_registry(tmp_path, monkeypatch):
    """Point user-registry + asset paths at temp dirs for every registry test."""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(assets_dir))
    _invalidate_cache()
    yield
    _invalidate_cache()
