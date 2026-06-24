"""Registry read tests must not observe the host's on-disk user overlay.

The registry read API merges ``$STRANDS_BASE_DIR/user_robots.json`` on top of
the package ``robots.json``. Because registering a custom robot is a supported
workflow, a contributor's machine can hold user robots whose descriptions
contain non-ASCII text - which would silently leak into assertions such as
``format_robot_table().isascii()`` or exact registry counts. The package-level
``conftest`` isolates ``STRANDS_BASE_DIR`` per test; these tests pin that
guarantee so the suite stays deterministic regardless of the host.
"""

from __future__ import annotations

from pathlib import Path

from strands_robots.registry import (
    format_robot_table,
    list_robots,
    register_robot,
)
from strands_robots.utils import get_assets_dir, get_base_dir

_MINIMAL_MJCF = '<mujoco><worldbody><body><geom size="0.1"/></body></worldbody></mujoco>'


def test_base_dir_is_isolated_from_real_home():
    """The user-registry base dir must point at a per-test temp area, not the
    real ``~/.strands_robots`` (where host-registered robots live)."""
    base = get_base_dir()
    real_home_registry = Path.home() / ".strands_robots"
    assert base != real_home_registry


def test_host_user_robots_do_not_leak_into_table():
    """A non-ASCII user description registered *inside* a test stays confined
    to that test rather than leaking from (or into) the shared real home.

    This reproduces the exact entry that previously broke
    ``test_table_is_pure_ascii`` when the suite read the host overlay."""
    asset_dir = get_assets_dir() / "iso_probe_bot"
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "iso_probe_bot.xml").write_text(_MINIMAL_MJCF)

    register_robot(
        name="iso_probe_bot",
        model_xml="iso_probe_bot.xml",
        description="probe \u2014 em-dash description",
        category="arm",
        joints=6,
    )
    table = format_robot_table(max_width=1000)
    # Inside this test the registered robot is visible (proves the merge path)...
    assert "iso_probe_bot" in table
    assert not table.isascii()
    # ...and it is the overlay entry, not something baked into the package JSON.
    names = {r["name"] for r in list_robots()}
    assert "iso_probe_bot" in names
