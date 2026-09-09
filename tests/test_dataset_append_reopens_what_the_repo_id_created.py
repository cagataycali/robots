"""The ``repo_id`` that created a dataset reopens it for appending.

``DatasetRecorder.create`` resolves the dataset directory once, through
:func:`~strands_robots.dataset_recorder.resolve_dataset_dir`, and forwards the
result to LeRobot as an explicit ``root``. ``resume`` -- the multi-episode append
entry point, and the only writable one for an existing dataset -- forwarded the
caller's ``root`` unresolved instead. An absent one therefore stayed absent, and
``LeRobotDataset.resume`` refuses that outright::

    resume() requires an explicit 'root' directory because it creates a
    DatasetWriter. Writing into the revision-safe Hub snapshot cache (used when
    root=None) would corrupt the shared cache. Please provide a local directory
    path.

The refusal is correct about LeRobot's own resolution and asks for a directory
this repo already names: the one ``create`` wrote the dataset to. So append was
unreachable on exactly the arguments ``create`` accepts, which is the documented
default path. Measured end to end against lerobot 0.6.2 on a MuJoCo scene --
``start_recording(repo_id="probe/append_two_episodes", root=None,
overwrite=True)``, a 30-step rollout, ``stop_recording()``, then the same
``start_recording`` call again with the default ``overwrite=False``:

* the first session wrote one episode, every call ``status="success"``;
* the second returned ``status="error"``, "Dataset init failed: resume()
  requires an explicit 'root' directory ...", having already resolved that
  directory to decide an append was wanted and stashed it as
  ``last_dataset_root``;
* ``verify_dataset_episodes(expected=2)`` then reported 1 episode / 30 frames.

With ``resume`` resolving once, the same script records 2 episodes / 60 frames
and every call reports success.

Every existing fake dataset class accepts ``root=None`` silently, so none of them
could observe this; the double below reproduces the one property that decides the
outcome -- a writer refuses an absent root -- and is what makes the cells here
fail on the unresolved forward rather than merely describe it.

An explicit ``root`` is unaffected: it was, and still is, used verbatim.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from strands_robots import dataset_recorder as dr


class _Meta:
    total_episodes = 1
    total_frames = 30


class _WriterThatRefusesAnAbsentRoot:
    """A LeRobotDataset double shaped like the writer, for both entry points.

    ``resume`` and ``create`` record the ``root`` they were handed. ``resume``
    additionally refuses an absent one with LeRobot's own wording, because that
    refusal -- not the value of the kwarg -- is what a caller actually hits.
    """

    resume_kwargs: dict[str, Any] = {}
    create_kwargs: dict[str, Any] = {}

    def __init__(self, repo_id: str, root: str | None = None) -> None:
        self.repo_id = repo_id
        self.root = Path(root) if root else None
        self.meta = _Meta()

    @classmethod
    def resume(cls, repo_id: str, root: str | None = None, streaming_encoding: bool = True) -> Any:
        if not root:
            raise ValueError(
                "resume() requires an explicit 'root' directory because it creates a "
                "DatasetWriter. Writing into the revision-safe Hub snapshot cache "
                "(used when root=None) would corrupt the shared cache. Please provide "
                "a local directory path."
            )
        cls.resume_kwargs = {"repo_id": repo_id, "root": root}
        return cls(repo_id, root=root)

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int = 30,
        root: str | None = None,
        robot_type: str = "unknown",
        features: dict[str, Any] | None = None,
        use_videos: bool = True,
        image_writer_threads: int = 4,
    ) -> Any:
        cls.create_kwargs = {"repo_id": repo_id, "root": root}
        return cls(repo_id, root=root)


@pytest.fixture
def writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> type[_WriterThatRefusesAnAbsentRoot]:
    """Inject the double, relocate the dataset home, and anchor the cwd."""
    module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    module.LeRobotDataset = _WriterThatRefusesAnAbsentRoot  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", module)
    home = tmp_path / "home"
    monkeypatch.setattr(dr, "_lerobot_home", lambda: home)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    _WriterThatRefusesAnAbsentRoot.resume_kwargs = {}
    _WriterThatRefusesAnAbsentRoot.create_kwargs = {}
    return _WriterThatRefusesAnAbsentRoot


class _Case(NamedTuple):
    repo_id: str
    # Where resolve_dataset_dir says this id lives, relative to the fixture's
    # dataset home ("home") or the working directory ("cwd").
    anchor: str
    relative: str


@pytest.mark.parametrize(
    "case",
    [
        _Case(repo_id="user/data", anchor="home", relative="user/data"),
        _Case(repo_id="bare_name", anchor="cwd", relative="bare_name"),
        _Case(repo_id="./dotted", anchor="cwd", relative="dotted"),
    ],
    ids=lambda case: case.repo_id,
)
def test_an_absent_root_appends_into_the_directory_the_repo_id_names(
    writer: type[_WriterThatRefusesAnAbsentRoot], tmp_path: Path, case: _Case
) -> None:
    """An append opens the dataset this id resolves to, rather than being refused.

    The path-like ids are the same divergence ``create`` resolves: LeRobot reads
    any absent root as ``$HF_LEROBOT_HOME/{repo_id}`` whatever the id looks like,
    while this repo reads an id that is itself a path as the directory. So the
    resolution has to happen here for the append to land where the recording did.
    """
    recorder = dr.DatasetRecorder.resume(repo_id=case.repo_id, root=None)

    anchor = (tmp_path / "home") if case.anchor == "home" else Path.cwd()
    # Resolved on both sides: a path-like id is forwarded relative, exactly as
    # ``create`` forwards it, so the effective directory is what has to match.
    assert Path(writer.resume_kwargs["root"]).resolve() == (anchor / case.relative).resolve()
    assert recorder.episode_count == 1


def test_an_explicit_root_is_still_forwarded_verbatim(
    writer: type[_WriterThatRefusesAnAbsentRoot], tmp_path: Path
) -> None:
    """The control: naming a root replaces the resolution, it is not joined to it."""
    chosen = tmp_path / "chosen"

    dr.DatasetRecorder.resume(repo_id="user/data", root=str(chosen))

    assert Path(writer.resume_kwargs["root"]) == chosen


def test_the_id_that_created_a_dataset_reopens_it(writer: type[_WriterThatRefusesAnAbsentRoot]) -> None:
    """One id, one directory: the two writable entry points must agree on it.

    They resolve through the same function, so agreement is by construction --
    which is the point. Before, only ``create`` resolved, and the pair disagreed
    for every id whose append is worth having.
    """
    dr.DatasetRecorder.create(repo_id="bare_name", root=None, joint_names=["j1"], task="pick")
    dr.DatasetRecorder.resume(repo_id="bare_name", root=None)

    assert Path(writer.resume_kwargs["root"]) == Path(writer.create_kwargs["root"])
