"""A distributed session's workers clone one description one at a time.

Every worker of a distributed run collects the whole tree, so the module-level
``pytest.importorskip("robot_descriptions.<x>_description")` in 20-odd files
reaches ONE shared cache directory in every worker at once. The upstream cache
takes no lock (see :mod:`tests.description_clone_lock`), so on a cold cache the
loser's ``git`` raised inside a collected module and the whole session ERRORed
out with a message about the cache.
"""

from __future__ import annotations

import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest

from tests.description_clone_lock import INSTALLED, LOCK_NAME, cache_dir, serialize_description_clones

#: Long enough that a second unserialized caller joins the first inside the
#: clone, short enough to pay once when the lock holds it out.
MEETING_TIMEOUT = 0.25


@pytest.fixture
def upstream_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    """The real upstream cache, pointed at a cache directory of our own."""
    cache = pytest.importorskip("robot_descriptions._cache")
    monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path / "cache"))
    return cache


def _local_description(tmp_path: Path) -> tuple[str, str]:
    """Return the URL and commit of a one-file git repository to clone."""
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "robot.xml").write_text("<mujoco/>")
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-C", str(remote)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "robot.xml"], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    commit = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return f"file://{remote}", commit.stdout.strip()


def test_the_session_runs_every_description_clone_under_the_lock(upstream_cache: ModuleType) -> None:
    """conftest installed it, and installing again does not stack a second lock."""
    assert getattr(upstream_cache.clone_to_cache, INSTALLED, False), (
        "tests/conftest.py should have installed the clone lock for the session"
    )
    installed = upstream_cache.clone_to_cache
    assert serialize_description_clones() is True
    assert upstream_cache.clone_to_cache is installed


def test_one_caller_clones_at_a_time(upstream_cache: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two callers meet inside the clone without the lock, and never with it."""
    meeting = threading.Barrier(2)
    live, peak = 0, 0
    counting = threading.Lock()

    def clone(description_name: str, commit: str | None = None) -> str:
        nonlocal live, peak
        with counting:
            live += 1
            peak = max(peak, live)
        try:
            meeting.wait(MEETING_TIMEOUT)  # a second caller arrives here, or does not
        except threading.BrokenBarrierError:
            pass
        with counting:
            live -= 1
        return f"/cache/{description_name}"

    monkeypatch.setattr(upstream_cache, "clone_to_cache", clone)
    assert serialize_description_clones() is True
    guarded = upstream_cache.clone_to_cache

    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = [future.result(timeout=30) for future in [pool.submit(guarded, "mujoco_menagerie") for _ in range(2)]]

    assert peak == 1, f"{peak} callers were inside the clone at once"
    assert paths == ["/cache/mujoco_menagerie"] * 2
    assert (cache_dir() / LOCK_NAME).exists()


def test_concurrent_importers_of_one_description_both_get_the_clone(
    upstream_cache: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real clone, run from two workers at once on a cold cache."""
    repositories = pytest.importorskip("robot_descriptions._repositories")
    url, commit = _local_description(tmp_path)
    monkeypatch.setitem(
        upstream_cache.REPOSITORIES,
        "menagerie_stand_in",
        repositories.Repository(cache_path="menagerie_stand_in", commit=commit, url=url),
    )
    start = threading.Barrier(2)

    def clone() -> str:
        start.wait(30)
        return str(upstream_cache.clone_to_cache("menagerie_stand_in"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        clones = [future.result(timeout=120) for future in [pool.submit(clone) for _ in range(2)]]

    assert clones == [str(cache_dir() / "menagerie_stand_in")] * 2
    assert (Path(clones[0]) / "robot.xml").read_text() == "<mujoco/>"
