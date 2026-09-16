"""A fetched tree that does not hold the declared model is reported, not counted.

Every reader of a fetched asset tree that names a file names the registry's
``asset.model_xml``:
:func:`~strands_robots.assets.manager.resolve_model_path` resolves that entry and
nothing else. Both clone routes -
:func:`~strands_robots.assets.download._download_via_git` for Menagerie robots
and :func:`~strands_robots.assets.download._download_from_github` for a custom
source - fetch their repository at HEAD, so an upstream rename retires the
declared name while the directory around it survives and the copy still
succeeds. Reported as ``downloaded``, that leaves the download and the resolver
disagreeing about one registry entry: the download says the robot is here, and
the resolver's refusal names the download that just ran as the remedy.

The fixtures below are the shape measured on ``google-deepmind/mujoco_menagerie``
at HEAD for the ``so101`` entry, which declares ``so101_new_calib.xml`` in a
directory that ships ``so101.xml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from strands_robots.assets import download as dl

_MOD = "strands_robots.assets.download"

#: The registry entry's declared model, and the files the fetched tree ships.
_DECLARED = "so101_new_calib.xml"
_ARRIVED = ("scene.xml", "so101.xml")


def _entry(*, source: bool) -> dict[str, Any]:
    """Registry entry for the fixture robot, on one of the two clone routes."""
    asset: dict[str, Any] = {"dir": "robotstudio_so101", "model_xml": _DECLARED, "scene_xml": "scene.xml"}
    if source:
        asset["source"] = {"type": "github", "repo": "owner/repo", "subdir": "robotstudio_so101"}
    return {"asset": asset, "category": "arm"}


def _fetch(*, source: bool, ships: tuple[str, ...], dest: Path) -> str:
    """Run one clone route against a fake clone shipping *ships*, return its result."""
    info = _entry(source=source)
    subdir = "robotstudio_so101"

    def _fake_clone(repo_url: str, clone_dir: str, **kw: object) -> None:
        tree = Path(clone_dir) / subdir
        tree.mkdir(parents=True, exist_ok=True)
        for name in ships:
            (tree / name).write_text("<mujoco/>")

    with patch(f"{_MOD}._shallow_clone", side_effect=_fake_clone):
        if source:
            return dl._download_from_github("so101", info, dest)
        return dl._download_via_git({"so101": info}, dest)["so101"]


@pytest.mark.parametrize("source", [False, True], ids=["menagerie_clone", "custom_github_source"])
def test_a_route_that_fetches_a_tree_without_the_declared_model_reports_it(source: bool, tmp_path: Path) -> None:
    """The verdict names the declared model and the model files that did arrive.

    And the fetched files stay on disk: they are a usable tree - this entry's
    ``scene_xml`` is in it - and the cache directory is where a user's own files
    live, so the refusal changes the verdict and undoes nothing.
    """
    result = _fetch(source=source, ships=_ARRIVED, dest=tmp_path)

    assert result.startswith("failed: ")
    assert _DECLARED in result
    for arrived in _ARRIVED:
        assert arrived in result
    assert sorted(p.name for p in (tmp_path / "robotstudio_so101").iterdir()) == list(_ARRIVED)


@pytest.mark.parametrize("source", [False, True], ids=["menagerie_clone", "custom_github_source"])
def test_a_route_that_fetches_the_declared_model_still_reports_a_download(source: bool, tmp_path: Path) -> None:
    """The same tree with the declared file in it is a download, unchanged."""
    assert _fetch(source=source, ships=(*_ARRIVED, _DECLARED), dest=tmp_path) == "downloaded"


def test_the_call_that_fetched_it_counts_it_failed_and_says_what_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``download_robots`` reports 0 downloaded / 1 failed, with the detail.

    This is the caller-visible half: the reported counts are what the
    ``download_assets`` tool prints, and a download reported as succeeding is
    what makes the resolver's refusal - "fetch it with the download_assets
    tool" - name a command that has already run.
    """
    registry = {"so101": _entry(source=False)}
    monkeypatch.setattr(f"{_MOD}.get_user_assets_dir", lambda: tmp_path)
    monkeypatch.setattr(f"{_MOD}.registry_list_robots", lambda mode: [{"name": n} for n in registry])
    monkeypatch.setattr(f"{_MOD}.get_robot", registry.get)
    monkeypatch.setattr(f"{_MOD}.resolve_robot_name", lambda n: n)
    monkeypatch.setattr(f"{_MOD}._needs_download", lambda *a, **k: True)
    monkeypatch.setattr(f"{_MOD}._robot_descriptions_available", lambda: False)

    def _fake_clone(repo_url: str, clone_dir: str, **kw: object) -> None:
        tree = Path(clone_dir) / "robotstudio_so101"
        tree.mkdir(parents=True, exist_ok=True)
        for name in _ARRIVED:
            (tree / name).write_text("<mujoco/>")

    with patch(f"{_MOD}._shallow_clone", side_effect=_fake_clone):
        result = dl.download_robots(names=["so101"])

    assert (result["downloaded"], result["failed"]) == (0, 1)
    assert result["failed_names"] == ["so101"]
    assert _DECLARED in result["failed_details"]["so101"]
