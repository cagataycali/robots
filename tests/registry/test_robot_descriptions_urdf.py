"""URDF auto-discovery of the ``robot_descriptions`` long tail.

The curated ``robots.json`` and the MJCF discovery surface only expose robots
that ship an MJCF model, because the MuJoCo backend needs an ``.xml``. URDF-only
descriptions (humanoids, quadrupeds, hands, dual-arm rigs) were therefore
invisible even though URDF-native backends such as Newton can load them
directly.

These tests exercise observable behavior of the parallel URDF surface in
``strands_robots.registry.discovery``:
    - the cheap name -> module lookup (``urdf_descriptions_module`` /
      ``is_urdf_discoverable`` / ``list_urdf_discoverable``) against the real
      description table,
    - the heavy path resolution (``discover_urdf_path``) with a stubbed import so
      the logic is verified without any network clone,
    - that the URDF surface covers the URDF-only long tail that MJCF discovery
      cannot.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from strands_robots.registry import discovery


@pytest.fixture(autouse=True)
def _clear_discovery_cache():
    """Reset discovery caches around every test so stubbed tables do not leak."""
    discovery.invalidate_cache()
    yield
    discovery.invalidate_cache()


# Cheap lookup surface (real description table)


def test_urdf_module_resolves_urdf_only_robot() -> None:
    pytest.importorskip("robot_descriptions")
    # atlas_v4 ships a URDF but no MJCF Menagerie model -> URDF discovery only.
    assert discovery.urdf_descriptions_module("atlas_v4") == "atlas_v4_description"


def test_urdf_module_resolves_panda() -> None:
    pytest.importorskip("robot_descriptions")
    assert discovery.urdf_descriptions_module("panda") == "panda_description"


def test_urdf_module_normalizes_dashes_and_case() -> None:
    pytest.importorskip("robot_descriptions")
    assert discovery.urdf_descriptions_module("Atlas-V4") == "atlas_v4_description"


def test_urdf_module_rejects_traversal_names() -> None:
    # Names with path/dot/slash chars must never reach importlib.
    assert discovery.urdf_descriptions_module("../evil") is None
    assert discovery.urdf_descriptions_module("robot_descriptions.os") is None


def test_urdf_module_unknown_returns_none() -> None:
    assert discovery.urdf_descriptions_module("definitely_not_a_robot_xyz") is None


def test_is_urdf_discoverable() -> None:
    pytest.importorskip("robot_descriptions")
    assert discovery.is_urdf_discoverable("panda") is True
    assert discovery.is_urdf_discoverable("definitely_not_a_robot_xyz") is False


def test_list_urdf_discoverable_is_sorted_and_nonempty() -> None:
    pytest.importorskip("robot_descriptions")
    names = discovery.list_urdf_discoverable()
    assert names == sorted(names)
    assert "panda" in names
    # MJCF modules (``_mj_description``) must not leak into the URDF table.
    assert all(not n.endswith("_mj") for n in names)


def test_urdf_surface_covers_mjcf_only_long_tail() -> None:
    pytest.importorskip("robot_descriptions")
    urdf = set(discovery.list_urdf_discoverable())
    mjcf = set(discovery.list_discoverable())
    # The URDF long tail must include robots with no MJCF model at all -
    # that is the whole point of URDF-native discovery.
    urdf_only = urdf - mjcf
    assert len(urdf_only) >= 5
    assert "atlas_v4" in urdf_only


# Heavy path resolution (stubbed import - no network)


def test_discover_urdf_path_resolves_existing_file(monkeypatch, tmp_path) -> None:
    urdf_file = tmp_path / "panda.urdf"
    urdf_file.write_text("<robot name='panda'></robot>")
    monkeypatch.setattr(discovery, "_urdf_modules", lambda: {"panda": "panda_description"})
    monkeypatch.setattr(
        discovery.importlib,
        "import_module",
        lambda modpath: SimpleNamespace(URDF_PATH=str(urdf_file)),
    )
    assert discovery.discover_urdf_path("panda") == str(urdf_file)


def test_discover_urdf_path_returns_none_for_non_description(monkeypatch) -> None:
    monkeypatch.setattr(discovery, "_urdf_modules", dict)

    def _explode(modpath: str):
        raise AssertionError(f"import_module must not be called for {modpath!r}")

    monkeypatch.setattr(discovery.importlib, "import_module", _explode)
    assert discovery.discover_urdf_path("definitely_not_a_robot_xyz") is None


def test_discover_urdf_path_none_when_module_lacks_urdf_path(monkeypatch) -> None:
    monkeypatch.setattr(discovery, "_urdf_modules", lambda: {"fakebot": "fakebot_description"})
    monkeypatch.setattr(
        discovery.importlib,
        "import_module",
        lambda modpath: SimpleNamespace(),  # no URDF_PATH attribute
    )
    assert discovery.discover_urdf_path("fakebot") is None


def test_discover_urdf_path_none_when_file_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(discovery, "_urdf_modules", lambda: {"fakebot": "fakebot_description"})
    monkeypatch.setattr(
        discovery.importlib,
        "import_module",
        lambda modpath: SimpleNamespace(URDF_PATH=str(tmp_path / "nope.urdf")),
    )
    assert discovery.discover_urdf_path("fakebot") is None


def test_discover_urdf_path_handles_import_error(monkeypatch) -> None:
    monkeypatch.setattr(discovery, "_urdf_modules", lambda: {"fakebot": "fakebot_description"})

    def _raise(modpath: str):
        raise ImportError(modpath)

    monkeypatch.setattr(discovery.importlib, "import_module", _raise)
    assert discovery.discover_urdf_path("fakebot") is None


def test_invalidate_cache_clears_urdf_modules(monkeypatch) -> None:
    pytest.importorskip("robot_descriptions")
    # Prime the cache, then confirm invalidate_cache resets it.
    first = discovery.list_urdf_discoverable()
    assert first  # non-empty real table
    discovery.invalidate_cache()
    # After invalidation a fresh stub takes effect (proves the lru_cache cleared).
    monkeypatch.setattr(
        discovery,
        "_urdf_modules",
        lambda: {"onlybot": "onlybot_description"},
    )
    assert discovery.list_urdf_discoverable() == ["onlybot"]
