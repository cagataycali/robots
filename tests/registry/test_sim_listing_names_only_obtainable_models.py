"""``list_robots(mode="sim")`` and ``has_sim`` name only models ``Robot(name)`` can obtain.

An entry declaring ``auto_download: false`` is never fetched, so until its model
file is placed by hand no ``Robot(name)`` can spawn it. The catalog a user reads
first must not promise it; once the file is on disk, it is listed again.
"""

from __future__ import annotations

import os

import pytest

from strands_robots.registry import get_robot, has_sim, list_robots
from strands_robots.registry.loader import _load

NEVER_FETCHED = sorted(
    name
    for name, info in _load("robots")["robots"].items()
    if isinstance(info.get("asset"), dict) and info["asset"].get("auto_download") is False
)


def test_the_registry_still_ships_never_fetched_entries() -> None:
    assert NEVER_FETCHED, "no auto_download=false entry left; this pin guards nothing"


@pytest.mark.parametrize("name", NEVER_FETCHED)
def test_a_never_fetched_model_is_listed_only_once_it_is_on_disk(name, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # the cwd/assets search path must not hold a checkout's files

    def listed() -> dict[str, bool]:
        return {
            "has_sim": has_sim(name),
            "in_sim_list": name in {r["name"] for r in list_robots(mode="sim")},
            "row_has_sim": next(r["has_sim"] for r in list_robots() if r["name"] == name),
        }

    assert listed() == {"has_sim": False, "in_sim_list": False, "row_has_sim": False}

    info = get_robot(name)
    assert info is not None
    asset = info["asset"]
    model = tmp_path / "assets" / asset["dir"] / asset["model_xml"]
    model.parent.mkdir(parents=True)
    model.write_text("<mujoco/>")
    assert os.environ["STRANDS_ASSETS_DIR"] == str(tmp_path / "assets")

    assert listed() == {"has_sim": True, "in_sim_list": True, "row_has_sim": True}
