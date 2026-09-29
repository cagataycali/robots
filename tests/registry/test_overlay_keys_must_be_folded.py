"""A robot key the lookup fold cannot produce is refused, not silently unreachable.

Every reader looks a robot up by :func:`~strands_robots.registry.loader.normalize_robot_name`
of the query - lowercase, trimmed, dashes as underscores. ``register_robot``
folds the name before it writes, so its entries are always reachable. A
``user_robots.json`` written by hand (or by any tool other than
``register_robot``) was merged verbatim, so a key like ``rover-001`` or ``My_Arm``
loaded without complaint and then answered no query at all - not even the
spelling it was declared in, because that query is folded before it reaches the
registry.

The loader does not fold such a key on the user's behalf: two overlay keys, or
an overlay key and a package key, could collapse onto one entry and the merge
would keep whichever came last. It refuses the load instead, the same way it
refuses an alias collision or an unknown driver, and the error names the
spelling to rename to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_robots.registry import get_robot, list_robots, register_robot, unregister_robot


def _write_overlay(base: Path, robots: dict) -> Path:
    """Write user_robots.json directly under *base*, bypassing register_robot()."""
    path = base / "user_robots.json"
    path.write_text(json.dumps({"robots": robots}))
    return path


def _entry(description: str = "hand-written") -> dict:
    return {
        "description": description,
        "category": "arm",
        "joints": 6,
        "asset": {"dir": "x", "model_xml": "x.xml", "scene_xml": "x.xml"},
    }


@pytest.mark.parametrize(
    ("declared", "folded"),
    [("rover-001", "rover_001"), ("My_Arm", "my_arm"), (" padded ", "padded")],
    ids=["dashed", "mixed-case", "padded"],
)
def test_an_overlay_key_that_is_not_folded_is_refused_with_its_file_and_folded_spelling(
    tmp_path: Path, declared: str, folded: str
) -> None:
    """The load fails, names the overlay file the key is in and the spelling to rename it to."""
    overlay = _write_overlay(tmp_path, {declared: _entry()})

    with pytest.raises(ValueError) as refused:
        get_robot(declared)

    message = str(refused.value)
    assert f"Robot key '{declared}' in {overlay} is not a lookup key" in message
    assert f"rename it to '{folded}'" in message


def test_an_overlay_key_that_folds_onto_a_shipped_robot_warns_that_renaming_replaces_it(tmp_path: Path) -> None:
    """Renaming ``Panda`` to ``panda`` would override the shipped entry, so the error says so."""
    _write_overlay(tmp_path, {"Panda": _entry()})

    with pytest.raises(ValueError) as refused:
        list_robots()

    assert "(a robot named 'panda' already exists; renaming replaces it, so choose another name to keep both)" in str(
        refused.value
    )


def test_unregister_robot_removes_an_unfolded_key_by_its_raw_spelling(tmp_path: Path) -> None:
    """The recovery path: the refused key can be removed without hand-editing the file."""
    overlay = _write_overlay(tmp_path, {"rover-001": _entry()})

    assert unregister_robot("rover-001") is True

    assert "rover-001" not in json.loads(overlay.read_text())["robots"]
    assert get_robot("so100") is not None


def test_a_folded_overlay_key_still_loads_and_answers_every_spelling(tmp_path: Path) -> None:
    """Control: the refusal is about the key's spelling, not about hand-written overlays."""
    _write_overlay(tmp_path, {"rover_001": _entry("folded")})

    for query in ("rover_001", "rover-001", "ROVER-001"):
        entry = get_robot(query)
        assert entry is not None, f"{query!r} reached no robot"
        assert entry["description"] == "folded"


def test_register_robot_refuses_to_write_while_an_unfolded_key_is_present(tmp_path: Path) -> None:
    """The write-time check mirrors the load: it does not persist into a registry that cannot load."""
    asset_dir = tmp_path / "assets" / "my_arm"
    asset_dir.mkdir(parents=True)
    (asset_dir / "arm.xml").write_text("<mujoco/>")
    path = _write_overlay(tmp_path, {"rover-001": _entry()})
    before = path.read_text()

    with pytest.raises(ValueError, match="rename it to 'rover_001'"):
        register_robot("my_arm", model_xml="arm.xml", overwrite=True)

    assert path.read_text() == before
