"""Registry integrity tests — catch silent regressions in robots.json.

These tests enforce invariants on the robot registry that prevent classes
of bugs like the one flagged by @awsarron on PR #84 review (2026-04-21):
entries where ``robot_descriptions_module`` was accidentally dropped during
the 38→68 robot expansion, silently breaking auto-download.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REGISTRY_PATH = Path(__file__).parent.parent / "strands_robots" / "registry" / "robots.json"


@pytest.fixture(scope="module")
def registry() -> dict:
    """Load the robot registry once per module."""
    with open(REGISTRY_PATH) as f:
        data = json.load(f)
    return data.get("robots", data)


def test_registry_loads(registry: dict) -> None:
    """Registry file parses as valid JSON with robot entries."""
    assert len(registry) > 0


def test_every_robot_declares_auto_download_strategy(registry: dict) -> None:
    """Every robot with an ``asset`` block must declare HOW it gets auto-downloaded.

    Valid options (exactly one required):
        1. ``asset.robot_descriptions_module`` — the robot_descriptions pip module name.
        2. ``asset.source`` with ``type: "github"`` — custom GitHub source block.
        3. ``asset.auto_download: false`` — explicit opt-out (user must supply assets).

    Without one of these, auto-download silently falls through to the
    naming-convention heuristic, which fails for most robots and only
    logs a warning. This was the trossen_wxai + google_robot regression.
    """
    offenders = []
    for name, info in registry.items():
        asset = info.get("asset")
        if not asset:
            continue  # No asset block — nothing to auto-download.

        has_rd = "robot_descriptions_module" in asset
        has_source = isinstance(asset.get("source"), dict) and asset["source"].get("type") == "github"
        opts_out = asset.get("auto_download") is False

        if not (has_rd or has_source or opts_out):
            offenders.append(name)

    assert not offenders, (
        "Robots missing auto-download strategy (add `robot_descriptions_module`, "
        "`source: {type: github, ...}`, or `auto_download: false`): " + ", ".join(offenders)
    )


def test_asset_dirs_are_unique(registry: dict) -> None:
    """No two robots should share the same asset directory name."""
    dir_counts: dict[str, list[str]] = {}
    for name, info in registry.items():
        asset_dir = info.get("asset", {}).get("dir")
        if asset_dir:
            dir_counts.setdefault(asset_dir, []).append(name)

    duplicates = {d: names for d, names in dir_counts.items() if len(names) > 1}
    assert not duplicates, f"Duplicate asset dirs: {duplicates}"


def test_no_path_traversal_in_asset_paths(registry: dict) -> None:
    """Registry-sourced paths must not contain ``..`` (path-traversal defense in depth)."""
    for name, info in registry.items():
        asset = info.get("asset", {})
        for key in ("dir", "model_xml", "scene_xml"):
            value = asset.get(key, "")
            assert ".." not in str(value).split("/"), f"{name}.asset.{key} contains '..': {value!r}"


def test_auto_download_false_is_bool_not_string(registry: dict) -> None:
    """``auto_download`` must be a proper JSON boolean, not the string ``"false"``."""
    for name, info in registry.items():
        ad = info.get("asset", {}).get("auto_download")
        if ad is not None:
            assert isinstance(ad, bool), f"{name}.asset.auto_download must be bool, got {type(ad).__name__}: {ad!r}"
