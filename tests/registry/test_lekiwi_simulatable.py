"""Regression: LeKiwi must be simulatable, not hardware-only.

``Robot("lekiwi", mode="sim")`` previously failed with ``No model found for
'lekiwi'`` because the registry entry carried only a ``hardware`` block and no
``asset`` block, while ``robot_descriptions`` ships no lekiwi module. The fix
points the entry at the Apache-2.0 Ekumen-OS/lekiwi MuJoCo description (SO-ARM
arm on a 3-omniwheel base, 9 actuators) via a GitHub asset source.

These checks are network-free: they validate the registry metadata that makes
LeKiwi resolvable. End-to-end download + load + render is covered in
``tests_integ/simulation/test_lekiwi_sim.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "strands_robots" / "registry" / "robots.json"


def _lekiwi() -> dict:
    with open(_REGISTRY_PATH) as f:
        robots = json.load(f)["robots"]
    return robots["lekiwi"]


def test_lekiwi_has_sim_asset_block() -> None:
    """LeKiwi exposes a sim asset (regression: was hardware-only)."""
    asset = _lekiwi().get("asset")
    assert asset is not None, "lekiwi must have an 'asset' block to be simulatable"
    assert asset["dir"] == "lekiwi"
    assert asset["model_xml"].endswith(".xml")
    assert asset["scene_xml"].endswith(".xml")


def test_lekiwi_declares_github_download_source() -> None:
    """The asset is auto-downloadable from a GitHub source (no robot_descriptions module exists)."""
    asset = _lekiwi()["asset"]
    source = asset.get("source")
    assert isinstance(source, dict) and source.get("type") == "github"
    assert source["repo"] == "Ekumen-OS/lekiwi"
    assert source.get("subdir"), "github source needs a subdir to locate the MJCF tree"


def test_lekiwi_retains_hardware_block() -> None:
    """Adding sim support must not drop the existing hardware mapping."""
    assert _lekiwi().get("hardware", {}).get("lerobot_type") == "lekiwi"
