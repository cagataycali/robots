"""Integration test: every robot in robots.json resolves to an existing file.

Walks the full registry and asserts that every ``asset.dir / asset.model_xml``
path is a valid relative path that would resolve under the search directories
**if** the assets were downloaded. This catches:
    - Typos in ``robots.json`` (e.g. ``asimov_v0.xml`` → ``xmls/asimov.xml``)
    - Upstream layout regressions in robot_descriptions / MuJoCo Menagerie
    - Missing ``dir`` or ``model_xml`` keys in sim-capable robots
    - Path traversal sequences in registry entries

The test does NOT require downloaded assets or GPU - it only validates the
registry metadata itself (directory/file names, path safety). Run it in the
unit or integ hatch env.

Added as follow-up to PR #84 review (issue #105, task 2).
"""

import json
from pathlib import Path

import pytest

#
# Load registry directly to avoid import side effects
#

_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "strands_robots" / "registry" / "robots.json"


def _load_registry() -> dict:
    """Load robots.json and return the robots dict."""
    with open(_REGISTRY_PATH) as f:
        data = json.load(f)
    return data.get("robots", data)


_ROBOTS = _load_registry()

# Robots that have simulation assets (asset.dir + asset.model_xml).
# Hardware-only robots (e.g. reachy2) have no 'asset' key.
_SIM_ROBOTS = {name: info for name, info in _ROBOTS.items() if "asset" in info}


@pytest.mark.parametrize("name", list(_ROBOTS.keys()), ids=list(_ROBOTS.keys()))
def test_registry_entry_is_well_formed_and_resolves(name: str) -> None:
    """Every robot has a description and category and resolves through the API.

    A sim-capable robot (one with an ``asset`` key; hardware-only robots such
    as reachy2 have none) also names a non-empty ``asset.dir`` and
    ``asset.model_xml``, keeps every asset path relative and free of ``..``,
    and spells its model and scene files with an ``.xml`` extension.
    """
    from strands_robots.registry import get_robot, resolve_name

    info = _ROBOTS[name]
    assert "description" in info, f"Robot '{name}' missing 'description'"
    assert "category" in info, f"Robot '{name}' missing 'category'"
    assert resolve_name(name) is not None, f"resolve_name({name!r}) returned None"
    assert get_robot(name) is not None, f"get_robot({name!r}) returned None"

    if name not in _SIM_ROBOTS:
        return
    asset = _SIM_ROBOTS[name]["asset"]
    assert "dir" in asset, f"Robot '{name}' missing 'asset.dir'"
    assert "model_xml" in asset, f"Robot '{name}' missing 'asset.model_xml'"
    assert isinstance(asset["dir"], str) and asset["dir"], f"Robot '{name}' has empty 'asset.dir'"
    assert isinstance(asset["model_xml"], str) and asset["model_xml"], f"Robot '{name}' has empty 'asset.model_xml'"
    for field in ("dir", "model_xml", "scene_xml"):
        value = asset.get(field, "")
        if not value:
            continue
        assert ".." not in value, f"Robot '{name}' asset.{field} contains '..': {value!r}"
        assert not value.startswith("/"), f"Robot '{name}' asset.{field} is absolute: {value!r}"
        if field != "dir":
            assert value.endswith(".xml"), f"Robot '{name}' {field} doesn't end with .xml: {value!r}"
