"""``convert_mjcf_to_usd(import_scene=True)`` keeps the worldbody it was asked for.

The floor fix-up (``_deactivate_worldbody_geoms``) exists for the robot-only
conversion ``add_robot`` makes. Run unconditionally it also switched off the
scene a caller explicitly imported, reported success, and published that USD
into the shared cache under the ``import_scene=True`` key. The converter now
hands the flag to one fix-up helper, which deactivates only when the scene was
not asked for; the pxr-backed cells live next to the floor test.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.isaac import mjcf_assets  # noqa: E402


def test_the_converter_threads_import_scene_into_the_fixups() -> None:
    source = inspect.getsource(mjcf_assets.convert_mjcf_to_usd)
    assert "_post_import_fixups(resolved, mjcf_path, import_scene=import_scene)" in source
    assert "_deactivate_worldbody_geoms(" not in source, "the floor fix-up must go through the flag"


def test_the_fixup_helper_reads_the_flag() -> None:
    source = inspect.getsource(mjcf_assets._post_import_fixups)
    assert "if not import_scene:" in source
    assert source.index("if not import_scene:") < source.index("_deactivate_worldbody_geoms(")
