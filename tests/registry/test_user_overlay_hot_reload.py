"""The mtime hot-reload must honor the user overlay, not just the package JSON.

The ``robots`` registry the read API serves is the package ``robots.json``
merged with the user-local overlay ``$STRANDS_BASE_DIR/user_robots.json``
(:func:`loader._merge_user_robots`).  ``register_robot`` / ``unregister_robot``
invalidate the cache explicitly, but any *other* writer - a second process, a
manual edit, or a tool that writes the file directly - relies on the loader's
documented "re-read when the source changes" behavior.  These tests pin that
the cache signature tracks the overlay file's mtime, so create / modify / delete
of ``user_robots.json`` are all observed without a manual ``invalidate_cache``.

Each test warms the cache first, then mutates the overlay file *directly*
(never via ``register_robot``, which would invalidate the cache and mask the
bug).  The autouse fixture in ``conftest`` isolates ``STRANDS_BASE_DIR`` per
test and clears the cache on entry/exit.
"""

from __future__ import annotations

import json
import os

from strands_robots.registry import get_robot, list_robots
from strands_robots.utils import get_base_dir


def _overlay_path():
    return get_base_dir() / "user_robots.json"


def _write_overlay(robots: dict) -> None:
    """Write user_robots.json directly, bypassing register_robot()."""
    path = _overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"robots": robots}))


def _entry(description: str) -> dict:
    return {
        "description": description,
        "category": "arm",
        "joints": 6,
        "asset": {"dir": "x", "model_xml": "x.xml", "scene_xml": "x.xml"},
    }


def test_external_overlay_creation_is_picked_up():
    """A user robot written to the overlay by an external writer becomes visible
    on the next read, without a manual cache invalidation."""
    # Warm the cache while the overlay file does not exist yet.
    assert get_robot("ext_created") is None

    _write_overlay({"ext_created": _entry("added out of process")})

    got = get_robot("ext_created")
    assert got is not None, "external overlay creation was not picked up (stale cache)"
    assert got["description"] == "added out of process"


def test_external_overlay_modification_is_picked_up():
    """Rewriting an existing overlay entry is observed on the next read."""
    _write_overlay({"ext_mod": _entry("first")})
    assert get_robot("ext_mod")["description"] == "first"  # warm cache

    _write_overlay({"ext_mod": _entry("second")})
    # Force a strictly-newer mtime so the change is unambiguous regardless of
    # filesystem timestamp granularity.
    path = _overlay_path()
    now = path.stat().st_mtime
    os.utime(path, (now + 10, now + 10))

    assert get_robot("ext_mod")["description"] == "second", (
        "external overlay modification was not picked up (stale cache)"
    )


def test_external_overlay_deletion_is_picked_up():
    """Deleting the overlay file drops its robots on the next read."""
    _write_overlay({"ext_deleted": _entry("temporary")})
    assert get_robot("ext_deleted") is not None  # warm cache

    _overlay_path().unlink()

    assert get_robot("ext_deleted") is None, "external overlay deletion was not picked up (stale cache)"
    # Package robots are unaffected by the overlay churn.
    assert any(r["name"] == "so101" for r in list_robots())
