"""The URDF long tail of ``robot_descriptions`` is part of the registry surface.

``urdf_robots.json`` is generated from assets the loader built (see
``scripts/build_urdf_registry.py``) and lists every ``robot_descriptions`` robot
that has a URDF, no MJCF sibling and no curated entry. These tests grade the
promises the file and its readers make: the set is exactly that complement,
curated always wins, ``list_robots()`` / ``get_robot()`` / ``has_sim()`` report
the URDF robots with ``source: "urdf"``, a robot the sweep could not build is
listed without a simulation asset and with the loader's refusal, every built
entry carries the compiled model's actuator count and an upstream pin, and the
file is in step with the installed ``robot_descriptions`` table. No network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_robots.registry import discovery, get_robot, has_sim, list_robots, resolve_name

_JSON = Path(discovery.__file__).parent / "urdf_robots.json"


@pytest.fixture(autouse=True)
def _fresh_tables():
    discovery.invalidate_cache()
    yield
    discovery.invalidate_cache()


def _table() -> dict[str, dict]:
    return json.loads(_JSON.read_text(encoding="utf-8"))["robots"]


def test_the_file_has_the_documented_shape() -> None:
    doc = json.loads(_JSON.read_text(encoding="utf-8"))
    assert set(doc) == {"_comment", "robot_descriptions_version", "robots"}
    assert "do not hand-edit" in doc["_comment"]
    for name, entry in doc["robots"].items():
        assert entry["module"] == f"{name}_description", name
        assert entry["category"] in {"arm", "bimanual", "hand", "humanoid", "mobile", "mobile_manip", "aerial"}, name
        assert isinstance(entry["floating"], bool) and isinstance(entry["tags"], list)
        if entry["has_sim"]:
            assert isinstance(entry["joints"], int) and entry["joints"] > 0, name
            assert entry["nq"] >= entry["joints"], name
            assert entry["repository"] and entry["commit"], f"{name}: upstream pin missing"
            assert isinstance(entry["mesh_formats"], list) and isinstance(entry["streamable"], bool), name
            # Streamable means no mesh was re-encoded; a native STL above MuJoCo's
            # face cap (bambot) is re-encoded too, so the formats alone do not decide.
            if entry["streamable"]:
                assert all(ext in {".stl", ".obj", ".msh"} for ext in entry["mesh_formats"]), name
        else:
            assert entry["refusal"], name


def test_the_file_lists_exactly_the_urdf_only_complement() -> None:
    pytest.importorskip("robot_descriptions")
    assert sorted(_table()) == discovery.list_urdf_only()


def test_urdf_only_is_disjoint_from_mjcf_discovery_and_the_curated_registry() -> None:
    pytest.importorskip("robot_descriptions")
    urdf_only = set(discovery.list_urdf_only())
    assert not urdf_only & set(discovery.list_discoverable())
    curated = {r["name"] for r in list_robots() if r["source"] == "curated"}
    assert not urdf_only & curated
    for name in urdf_only:
        assert resolve_name(name) == name, f"{name} is a curated alias"


def test_a_curated_name_that_also_has_a_urdf_description_is_served_by_the_curated_entry() -> None:
    pytest.importorskip("robot_descriptions")
    # panda has panda_description (URDF) and panda_mj_description (MJCF); the curated entry wins.
    assert discovery.is_urdf_discoverable("panda")
    assert not discovery.is_urdf_only("panda")
    entry = get_robot("panda")
    assert entry is not None and entry.get("source") != "urdf"


def test_list_robots_reports_urdf_robots_with_their_source_and_sim_flag() -> None:
    pytest.importorskip("robot_descriptions")
    table = _table()
    listed = {r["name"]: r for r in list_robots()}
    for name, entry in table.items():
        assert name in listed, name
        assert listed[name]["source"] == "urdf"
        assert listed[name]["has_sim"] is entry["has_sim"]
        assert listed[name]["has_real"] is False
        assert listed[name]["category"] == entry["category"]
        if entry["has_sim"]:
            assert listed[name]["joints"] == entry["joints"]
    curated = [r for r in listed.values() if r["source"] == "curated"]
    assert curated, "the curated registry still lists"
    assert all(r["source"] == "urdf" for r in list_robots(mode="sim") if r["name"] in table)
    assert not [r for r in list_robots(mode="real") if r["name"] in table]


def test_get_robot_synthesizes_an_asset_block_the_downloader_recognizes() -> None:
    pytest.importorskip("robot_descriptions")
    built = next(n for n, e in _table().items() if e["has_sim"])
    entry = get_robot(built)
    assert entry is not None and entry["source"] == "urdf" and entry["discovered"] is True
    asset = entry["asset"]
    assert asset["dir"] == f"{built}_description" == asset["robot_descriptions_module"]
    assert asset["model_xml"] == "robot.xml" and asset["scene_xml"] == "scene.xml"
    assert asset["source"] == {"type": "urdf"}
    assert entry["joints"] == _table()[built]["joints"]
    assert has_sim(built) is True


def test_a_description_that_did_not_build_has_no_asset_and_says_why() -> None:
    pytest.importorskip("robot_descriptions")
    table = _table()
    refused = [n for n, e in table.items() if not e["has_sim"]]
    if not refused:
        pytest.skip("every URDF description built at this commit")
    for name in refused:
        entry = get_robot(name)
        assert entry is not None and "asset" not in entry
        assert entry["refusal"] == table[name]["refusal"]
        assert has_sim(name) is False


def test_an_unknown_name_is_still_unknown() -> None:
    assert get_robot("definitely_not_a_robot_xyz") is None
    assert discovery.urdf_registry_entry("definitely_not_a_robot_xyz") is None
    assert discovery.is_urdf_only("../evil") is False


def test_the_file_matches_the_installed_robot_descriptions_release() -> None:
    """A newer robot_descriptions that adds a URDF-only robot must regenerate the file."""
    rd = pytest.importorskip("robot_descriptions")
    from importlib.metadata import version

    doc = json.loads(_JSON.read_text(encoding="utf-8"))
    installed = version("robot_descriptions")
    assert doc["robot_descriptions_version"] == installed, (
        f"urdf_robots.json was generated against robot_descriptions {doc['robot_descriptions_version']}, "
        f"{installed} is installed: run scripts/build_urdf_registry.py"
    )
    assert rd is not None
