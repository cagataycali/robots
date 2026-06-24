"""Tests for ``strands_robots.simulation.model_registry``.

Covers:
* ``register_urdf`` runtime insertion
* ``resolve_model`` happy path, unknown-name, and scene -> non-scene fallback
* ``resolve_urdf`` happy path, unknown-name, relative search-path resolution,
  and ``legacy_urdf`` (absolute + relative) registry entries
* ``list_registered_urdfs`` resolution-status mapping
* ``list_available_models`` formatted listing + asset-manager-absent fallback
* ``count_sim_robots`` registry count
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.model_registry import (
    list_available_models,
    list_registered_urdfs,
    register_urdf,
    resolve_model,
    resolve_urdf,
)


def test_list_available_models_contains_builtins():
    out = list_available_models()
    assert isinstance(out, str)
    assert "Name" in out and "Category" in out


def test_resolve_model_known_builtin_returns_path():
    """A Menagerie-backed robot is always resolvable (panda ships with mujoco_menagerie)."""
    pytest.importorskip("mujoco")  # panda requires mujoco_menagerie
    path = resolve_model("panda")
    assert path is not None
    assert path.endswith((".xml", ".urdf"))


def test_resolve_model_unknown_returns_none():
    assert resolve_model("this_does_not_exist_xyz") is None


def test_resolve_model_strips_common_suffix():
    """Friction fix: 'panda_default'/'panda_sim' should resolve to 'panda'.

    LLMs and humans frequently append a decorative qualifier to the bare
    registry key. resolve_model() strips a small allow-list of suffixes and
    retries once before returning None.
    """
    pytest.importorskip("mujoco")  # panda requires mujoco_menagerie
    bare = resolve_model("panda")
    assert bare is not None
    for variant in ("panda_default", "panda_sim", "panda_robot", "panda_arm"):
        assert resolve_model(variant) == bare, f"{variant} should resolve to panda"


def test_resolve_model_strip_does_not_overreach():
    """Stripping must not turn a genuinely-unknown name into a false hit."""
    # 'this_does_not_exist_xyz_default' strips to a still-unknown base -> None.
    assert resolve_model("this_does_not_exist_xyz_default") is None


def test_resolve_urdf_unknown_returns_none():
    assert resolve_urdf("this_does_not_exist_xyz") is None


def test_register_urdf_roundtrips(tmp_path):
    """register_urdf + resolve_urdf round-trip works."""
    fake_xml = tmp_path / "fake_robot.xml"
    fake_xml.write_text("<mujoco/>")

    register_urdf("__pytest_fake_robot__", str(fake_xml))

    resolved = resolve_urdf("__pytest_fake_robot__")
    assert resolved == str(fake_xml)


def test_resolve_urdf_relative_registered_path_found_in_search_dir(tmp_path, monkeypatch):
    """A relative registered path is resolved against the asset search paths."""
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path))
    rel = "myrobot/myrobot.xml"
    asset = tmp_path / rel
    asset.parent.mkdir(parents=True)
    asset.write_text("<mujoco/>")

    register_urdf("__pytest_rel_robot__", rel)

    assert resolve_urdf("__pytest_rel_robot__") == str(asset)


def test_resolve_urdf_registered_path_missing_returns_none(tmp_path, monkeypatch):
    """A registered relative path that exists nowhere resolves to None."""
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path))
    register_urdf("__pytest_missing_robot__", "nope/nowhere.xml")

    assert resolve_urdf("__pytest_missing_robot__") is None


def test_resolve_urdf_legacy_urdf_absolute_path(tmp_path, monkeypatch):
    """A registry entry carrying an absolute ``legacy_urdf`` is honored."""
    import strands_robots.simulation.model_registry as mr

    legacy = tmp_path / "legacy.urdf"
    legacy.write_text("<robot/>")

    monkeypatch.setattr(mr, "_HAS_REGISTRY", True)
    monkeypatch.setattr(mr, "resolve_name", lambda n: n)
    monkeypatch.setattr(mr, "get_robot", lambda n: {"legacy_urdf": str(legacy)})

    assert mr.resolve_urdf("__pytest_legacy_abs__") == str(legacy)


def test_resolve_urdf_legacy_urdf_relative_path(tmp_path, monkeypatch):
    """A relative ``legacy_urdf`` is resolved against the asset search paths."""
    import strands_robots.simulation.model_registry as mr

    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path))
    rel = "legacy_arm/legacy_arm.urdf"
    asset = tmp_path / rel
    asset.parent.mkdir(parents=True)
    asset.write_text("<robot/>")

    monkeypatch.setattr(mr, "_HAS_REGISTRY", True)
    monkeypatch.setattr(mr, "resolve_name", lambda n: n)
    monkeypatch.setattr(mr, "get_robot", lambda n: {"legacy_urdf": rel})

    assert mr.resolve_urdf("__pytest_legacy_rel__") == str(asset)


def test_list_registered_urdfs_reports_resolution_status(tmp_path):
    """``list_registered_urdfs`` maps each registered name to its resolved path."""
    good = tmp_path / "good.xml"
    good.write_text("<mujoco/>")
    register_urdf("__pytest_lr_good__", str(good))
    register_urdf("__pytest_lr_bad__", "/does/not/exist.xml")

    mapping = list_registered_urdfs()

    assert mapping["__pytest_lr_good__"] == str(good)
    assert mapping["__pytest_lr_bad__"] is None


def test_list_available_models_fallback_without_asset_manager(tmp_path, monkeypatch):
    """Without the asset manager, listing falls back to the URDF registry table."""
    import strands_robots.simulation.model_registry as mr

    good = tmp_path / "present.xml"
    good.write_text("<mujoco/>")
    register_urdf("__pytest_listing_present__", str(good))
    register_urdf("__pytest_listing_absent__", "/no/such/file.xml")

    monkeypatch.setattr(mr, "_HAS_ASSET_MANAGER", False)
    out = mr.list_available_models()

    assert "Registered URDFs:" in out
    assert "[OK] __pytest_listing_present__" in out
    assert "[MISSING] __pytest_listing_absent__" in out


def test_resolve_model_falls_back_to_non_scene_asset(monkeypatch):
    """When the scene variant is unavailable, ``resolve_model`` retries non-scene."""
    import strands_robots.simulation.model_registry as mr

    monkeypatch.setattr(mr, "_HAS_ASSET_MANAGER", True)

    class _FakePath:
        def __init__(self, exists: bool):
            self._exists = exists

        def exists(self) -> bool:
            return self._exists

        def __str__(self) -> str:
            return "/fake/non_scene_model.xml"

    def fake_resolve(name, prefer_scene=True):
        # Scene variant missing, plain model present.
        return _FakePath(exists=not prefer_scene)

    monkeypatch.setattr(mr, "resolve_model_path", fake_resolve)

    assert mr.resolve_model("__pytest_nonscene__", prefer_scene=True) == "/fake/non_scene_model.xml"


def test_count_sim_robots_matches_registry(monkeypatch):
    """``count_sim_robots`` counts sim-capable robots from the registry."""
    import strands_robots.simulation.model_registry as mr

    count = mr.count_sim_robots()
    assert isinstance(count, int)
    assert count > 0


def test_resolve_model_prefers_registered_local_path(tmp_path):
    """``resolve_model`` returns a user-registered URDF before the asset manager."""
    asset = tmp_path / "local_first.xml"
    asset.write_text("<mujoco/>")
    register_urdf("__pytest_local_first__", str(asset))

    assert resolve_model("__pytest_local_first__") == str(asset)
