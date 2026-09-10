"""The ``repo_id`` that recorded a dataset reads it back.

Every writing entry point resolves the dataset directory through
:func:`~strands_robots.dataset_recorder.resolve_dataset_dir` and hands LeRobot
the result as an explicit ``root``, because a ``repo_id`` that is itself a path
(absolute, ``./``-prefixed, or with no ``owner/name`` slash) is a local
directory here and is **not** one to LeRobot, which resolves any absent root to
``$HF_LEROBOT_HOME/{repo_id}`` whatever the id looks like.

Both read-back entry points forwarded the caller's ``root`` unresolved instead,
so a read by the recording id looked for the dataset somewhere it had never
been. The miss then falls through to a Hub lookup for a name that only ever
meant a directory. Measured end to end against lerobot 0.6.2 on a MuJoCo scene,
``HF_LEROBOT_HOME`` pointed at an empty directory and the working directory
fresh::

    sim.start_recording(repo_id="sim_recording", task="pick the cube", fps=30,
                        root=None, overwrite=True)   # status="success"
    sim.run_policy(policy_provider="mock", n_steps=45, control_frequency=30.0)
    sim.stop_recording()                             # status="success"
    # 7 files under ./sim_recording (parquet + 2 per-camera MP4);
    # $HF_LEROBOT_HOME/sim_recording does not exist.

    sim.replay_episode(repo_id="sim_recording", episode=0)
    # status="error": "Cannot reach
    #   https://huggingface.co/api/datasets/sim_recording/refs"
    strands_robots.stream_dataset("sim_recording")
    # RepositoryNotFoundError: 404 ... /api/datasets/sim_recording/refs

With both reads resolving the same rule, the same script replays 45/45 frames
under ``status="success"`` and streams 45 frames from ``./sim_recording``.

Only that rule is resolved on the read side, and the cells below pin the
boundary as well as the fix. An ``owner/name`` id keeps its absent root: that is
how LeRobot selects the revision-safe Hub snapshot cache for a download
(``snapshot_download(cache_dir=HF_LEROBOT_HUB_CACHE)``), and naming a directory
instead switches it to a plain ``local_dir=`` materialization. A Hub id needs no
help anyway -- the directory LeRobot derives for it is already the one a local
recording under that id wrote to.

The existing dataset doubles all accept any ``root`` silently, so none of them
could observe this. The reader below reproduces the one property that decides the
outcome: it finds a dataset only where it is told to look, and falls back to the
Hub for an absent root exactly as LeRobot does. It stands in at BOTH read sites,
so the three properties are asserted of every entry point that opens a dataset
by id rather than of whichever one was written first.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from strands_robots import dataset_recorder as dr
from strands_robots import streaming_dataset as sd


class _OfflineHub(Exception):
    """Stands in for the hub error a read of a non-existent dataset raises."""


class _Meta:
    total_episodes = 1
    episodes = [{"dataset_from_index": 0, "dataset_to_index": 45}]


class _ReaderThatFindsOnlyWhereItLooks:
    """A dataset double shaped like the reader.

    It records the ``root`` it was handed and resolves an absent one the way
    LeRobot does -- ``$HF_LEROBOT_HOME/{repo_id}``, then the Hub -- so a read
    aimed at the wrong directory fails the way a caller actually experiences it
    rather than merely carrying a different kwarg.

    The two-parameter signature is also what keeps it usable as the streaming
    double: ``StreamingDatasetReader.open`` forwards a kwarg only when the
    constructor declares it, so the streaming knobs are skipped and the call
    reduces to the ``(repo_id, root)`` pair under test.
    """

    home: Path = Path()
    kwargs: dict[str, Any] = {}
    num_frames = 45

    def __init__(self, repo_id: str, root: str | None = None, **_forwarded: Any) -> None:
        type(self).kwargs = {"repo_id": repo_id, "root": root}
        looked_in = Path(root) if root else type(self).home / repo_id
        if not (looked_in / "meta" / "info.json").exists():
            if root:
                raise FileNotFoundError(f"No LeRobotDataset at {looked_in}")
            raise _OfflineHub(f"Cannot reach https://huggingface.co/api/datasets/{repo_id}/refs")
        self.repo_id = repo_id
        self.root = looked_in
        self.meta = _Meta()


def plant(directory: Path) -> Path:
    """Write the minimum on-disk shape a dataset read looks for."""
    (directory / "meta").mkdir(parents=True, exist_ok=True)
    (directory / "meta" / "info.json").write_text(json.dumps({"fps": 30}), encoding="utf-8")
    return directory


@pytest.fixture
def reader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> type[_ReaderThatFindsOnlyWhereItLooks]:
    """Inject the double at both read sites, relocate the home, anchor the cwd."""
    module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    module.LeRobotDataset = _ReaderThatFindsOnlyWhereItLooks  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", module)
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _ReaderThatFindsOnlyWhereItLooks, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(dr, "_lerobot_home", lambda: home)
    _ReaderThatFindsOnlyWhereItLooks.home = home
    _ReaderThatFindsOnlyWhereItLooks.kwargs = {}
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return _ReaderThatFindsOnlyWhereItLooks


def _replay(repo_id: str, root: str | None) -> tuple[Any, Any]:
    """Read an episode the way ``Simulation.replay_episode`` does."""
    ds, start, length = dr.load_lerobot_episode(repo_id, 0, root)
    return ds, (start, length)


def _stream(repo_id: str, root: str | None) -> tuple[Any, Any]:
    """Read frames the way ``Simulation.stream_dataset`` does."""
    reader = sd.stream_dataset(repo_id, root=root)
    return reader.dataset, reader.num_frames


class _Read(NamedTuple):
    """A read-back entry point, and what it reports about what it opened."""

    name: str
    open: Callable[[str, str | None], tuple[Any, Any]]
    reports: Any


READS = [
    _Read("replay", _replay, (0, 45)),
    _Read("stream", _stream, 45),
]


class _PathLikeId(NamedTuple):
    repo_id: str
    # The directory that spelling names, relative to the working directory.
    name: str


PATH_LIKE_IDS = [
    _PathLikeId(repo_id="sim_recording", name="sim_recording"),
    _PathLikeId(repo_id="./dotted", name="dotted"),
]


@pytest.mark.parametrize("read", READS, ids=lambda read: read.name)
@pytest.mark.parametrize("case", PATH_LIKE_IDS, ids=lambda case: case.repo_id)
def test_a_path_like_repo_id_reads_the_directory_it_recorded_to(
    reader: type[_ReaderThatFindsOnlyWhereItLooks], read: _Read, case: _PathLikeId
) -> None:
    """The read lands where ``resolve_dataset_dir`` put the recording.

    The directory is planted at the *writer's* resolution, so this asserts the
    two sides agree rather than restating the reader's own arithmetic.

    The relative spellings are the whole exposure. An ABSOLUTE path-like id is
    not tested here because it cannot be broken: ``$HF_LEROBOT_HOME / "/abs/dir"``
    is ``/abs/dir`` -- joining an absolute right-hand operand discards the left --
    so LeRobot's own resolution already lands on it, with or without this fix.
    """
    recorded_at = plant(dr.resolve_dataset_dir(case.repo_id, None))

    ds, reported = read.open(case.repo_id, None)

    assert Path(ds.root).resolve() == (Path.cwd() / case.name).resolve()
    assert Path(recorded_at).resolve() == Path(ds.root).resolve()
    assert reported == read.reports


@pytest.mark.parametrize("read", READS, ids=lambda read: read.name)
def test_a_hub_id_is_left_to_lerobots_own_resolution(
    reader: type[_ReaderThatFindsOnlyWhereItLooks], read: _Read, tmp_path: Path
) -> None:
    """An ``owner/name`` id reads with no root, and still finds a local recording.

    The boundary, and the reason the read side does not simply apply
    ``resolve_dataset_dir``: an absent root is how LeRobot selects the
    revision-safe Hub snapshot cache for a download, so naming the directory for
    a Hub id would move every download out of that cache. Nothing is lost by
    leaving it -- the home LeRobot derives is the one a local recording under
    that id wrote to, which is why this read succeeds.
    """
    plant(tmp_path / "home" / "user" / "data")

    ds, _ = read.open("user/data", None)

    assert reader.kwargs["root"] is None
    assert Path(ds.root) == tmp_path / "home" / "user" / "data"


@pytest.mark.parametrize("read", READS, ids=lambda read: read.name)
def test_an_explicit_root_is_read_verbatim(
    reader: type[_ReaderThatFindsOnlyWhereItLooks], read: _Read, tmp_path: Path
) -> None:
    """The control: naming a root replaces the resolution, it is not joined to it."""
    chosen = plant(tmp_path / "chosen")

    ds, _ = read.open("user/data", str(chosen))

    assert Path(reader.kwargs["root"]) == chosen
    assert Path(ds.root) == chosen
