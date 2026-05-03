"""Tests for ``strands_robots.simulation.model_registry``.

Covers:
* ``register_urdf`` runtime insertion
* ``resolve_model`` happy path + unknown-name
* ``resolve_urdf`` happy path + unknown-name
* ``list_available_models`` formatted listing
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.model_registry import (
    list_available_models,
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


def test_resolve_urdf_unknown_returns_none():
    assert resolve_urdf("this_does_not_exist_xyz") is None


def test_register_urdf_roundtrips(tmp_path):
    """register_urdf + resolve_urdf round-trip works."""
    fake_xml = tmp_path / "fake_robot.xml"
    fake_xml.write_text("<mujoco/>")

    register_urdf("__pytest_fake_robot__", str(fake_xml))

    resolved = resolve_urdf("__pytest_fake_robot__")
    assert resolved == str(fake_xml)
