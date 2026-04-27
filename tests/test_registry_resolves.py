"""Integration test: every robot in robots.json resolves to an existing file.

Walks the full registry and asserts that every ``asset.dir / asset.model_xml``
path is a valid relative path that would resolve under the search directories
**if** the assets were downloaded. This catches:
    - Typos in ``robots.json`` (e.g. ``asimov_v0.xml`` → ``xmls/asimov.xml``)
    - Upstream layout regressions in robot_descriptions / MuJoCo Menagerie
    - Missing ``dir`` or ``model_xml`` keys in sim-capable robots
    - Path traversal sequences in registry entries

The test does NOT require downloaded assets or GPU — it only validates the
registry metadata itself (directory/file names, path safety). Run it in the
unit or integ hatch env.

Added as follow-up to PR #84 review (issue #105, task 2).
"""

import json
from pathlib import Path

import pytest

# ─────────────────────────────────────────────────────────────────────
# Load registry directly to avoid import side effects
# ─────────────────────────────────────────────────────────────────────

_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "strands_robots" / "registry" / "robots.json"


def _load_registry() -> dict:
    """Load robots.json and return the robots dict."""
    with open(_REGISTRY_PATH) as f:
        data = json.load(f)
    return data.get("robots", data)


_ROBOTS = _load_registry()

# Robots that have simulation assets (asset.dir + asset.model_xml).
# Hardware-only robots (e.g. lekiwi, reachy2) have no 'asset' key.
_SIM_ROBOTS = {name: info for name, info in _ROBOTS.items() if "asset" in info}
_SIM_ROBOT_NAMES = list(_SIM_ROBOTS.keys())


# ─────────────────────────────────────────────────────────────────────
# Tests for ALL robots (sim + hardware-only)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(_ROBOTS.keys()), ids=list(_ROBOTS.keys()))
def test_registry_entry_is_well_formed(name: str) -> None:
    """Every robot must have a description and category."""
    info = _ROBOTS[name]
    assert "description" in info, f"Robot '{name}' missing 'description'"
    assert "category" in info, f"Robot '{name}' missing 'category'"


@pytest.mark.parametrize("name", list(_ROBOTS.keys()), ids=list(_ROBOTS.keys()))
def test_registry_resolve_via_api(name: str) -> None:
    """Verify the registry API can look up each robot without errors."""
    from strands_robots.registry import get_robot, resolve_name

    canonical = resolve_name(name)
    assert canonical is not None, f"resolve_name({name!r}) returned None"

    info = get_robot(name)
    assert info is not None, f"get_robot({name!r}) returned None"


# ─────────────────────────────────────────────────────────────────────
# Tests for sim-capable robots only (have 'asset' key)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", _SIM_ROBOT_NAMES, ids=_SIM_ROBOT_NAMES)
def test_sim_robot_has_required_asset_fields(name: str) -> None:
    """Sim robots must have asset.dir and asset.model_xml."""
    asset = _SIM_ROBOTS[name]["asset"]
    assert "dir" in asset, f"Robot '{name}' missing 'asset.dir'"
    assert "model_xml" in asset, f"Robot '{name}' missing 'asset.model_xml'"
    assert isinstance(asset["dir"], str) and asset["dir"], f"Robot '{name}' has empty 'asset.dir'"
    assert isinstance(asset["model_xml"], str) and asset["model_xml"], f"Robot '{name}' has empty 'asset.model_xml'"


@pytest.mark.parametrize("name", _SIM_ROBOT_NAMES, ids=_SIM_ROBOT_NAMES)
def test_sim_robot_paths_are_safe(name: str) -> None:
    """No registry path should contain traversal sequences."""
    asset = _SIM_ROBOTS[name]["asset"]
    dir_name = asset.get("dir", "")
    model_xml = asset.get("model_xml", "")
    scene_xml = asset.get("scene_xml", "")

    for field, value in [("dir", dir_name), ("model_xml", model_xml), ("scene_xml", scene_xml)]:
        if not value:
            continue
        assert ".." not in value, f"Robot '{name}' asset.{field} contains '..': {value!r}"
        assert not value.startswith("/"), f"Robot '{name}' asset.{field} is absolute: {value!r}"


@pytest.mark.parametrize("name", _SIM_ROBOT_NAMES, ids=_SIM_ROBOT_NAMES)
def test_sim_robot_xml_has_xml_extension(name: str) -> None:
    """model_xml and scene_xml should end with .xml."""
    asset = _SIM_ROBOTS[name]["asset"]
    model_xml = asset.get("model_xml", "")
    scene_xml = asset.get("scene_xml", "")

    if model_xml:
        assert model_xml.endswith(".xml"), f"Robot '{name}' model_xml doesn't end with .xml: {model_xml!r}"
    if scene_xml:
        assert scene_xml.endswith(".xml"), f"Robot '{name}' scene_xml doesn't end with .xml: {scene_xml!r}"
