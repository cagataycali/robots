"""A dataset is written into the directory ``create`` prepared for it.

``DatasetRecorder.create`` does two things with a dataset directory. It resolves
one through :func:`~strands_robots.dataset_recorder.resolve_dataset_dir` and
hands it to ``_prepare_create_target``, which inspects it and - under
``overwrite=True`` - deletes it; then it calls ``LeRobotDataset.create``, which
resolves a directory of its own from ``repo_id`` whenever ``root`` is absent.
Those two resolutions are not the same one. ``resolve_dataset_dir`` reads a
``repo_id`` that is itself a path - no ``owner/name`` slash, or ``./``-prefixed -
as a local directory, while LeRobot resolves any absent root to
``$HF_LEROBOT_HOME/{repo_id}`` whatever the id looks like. For those ids the
guard therefore inspected one directory and the writer wrote another.

Measured against lerobot 0.6.2 with the dataset home and the working directory
both redirected into a scratch tree, ``repo_id="sim_recording"``, ``root=None``:

* ``overwrite=True`` with an unrelated ``./sim_recording`` holding one file:
  the file's directory was **deleted** and the dataset was created under
  ``$HF_LEROBOT_HOME/sim_recording``. The delete and the write named different
  places, so what ``overwrite`` consumed was not what it replaced.
* ``overwrite=True`` with a dataset already at ``$HF_LEROBOT_HOME/sim_recording``
  raised ``FileExistsError: [Errno 17] File exists`` and the stale dataset
  survived: the flag could not overwrite, because the wipe was aimed elsewhere.
* ``overwrite=False`` in the same arrangement raised the same bare ``[Errno 17]``
  rather than the message naming ``overwrite=True`` and :meth:`resume` -
  ``_prepare_create_target`` exists to replace that error and was reading a
  directory the write never touched.

``create`` now resolves once and forwards the result as an explicit ``root``, so
the inspected directory is the written directory by construction. The ordinary
``owner/name`` id is unaffected: both resolutions already agreed on
``$HF_LEROBOT_HOME/{repo_id}`` there, which the controls below hold.

``DatasetRecorder.resume`` resolves the same way, for a different reason: it
prepares nothing, so it has no second resolution to disagree with, but LeRobot
refuses an absent root outright there - a writer must not open the Hub snapshot
cache - which left the append entry point unreachable on the arguments ``create``
accepts. See
``tests/test_dataset_append_reopens_what_the_repo_id_created.py``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import pytest

from strands_robots import dataset_recorder as dr
from strands_robots.dataset_recorder import DatasetRecorder, resolve_dataset_dir


class _Writer:
    """A ``create`` shaped like ``LeRobotDataset.create``.

    Two properties matter and both decide where a dataset ends up: an absent
    ``root`` resolves to ``$HF_LEROBOT_HOME/{repo_id}``, and the directory is
    made with ``exist_ok=False`` so an existing one raises the bare
    ``FileExistsError`` that ``_prepare_create_target`` exists to pre-empt. A
    double that ignored ``root`` could not see this defect at all.
    """

    #: Stands in for ``$HF_LEROBOT_HOME``; the fixture points it at a scratch dir.
    home: ClassVar[Path] = Path()

    def __init__(self, repo_id: str, root: Path) -> None:
        self.repo_id = repo_id
        self.root = root

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int = 30,
        root: str | None = None,
        robot_type: str | None = None,
        features: dict[str, Any] | None = None,
        use_videos: bool = True,
        image_writer_threads: int = 0,
    ) -> _Writer:
        resolved = Path(root) if root is not None else cls.home / repo_id
        resolved.mkdir(parents=True, exist_ok=False)
        (resolved / "meta").mkdir()
        return cls(repo_id, resolved)


class _Bench(NamedTuple):
    """A private dataset home and working directory, plus the writer."""

    home: Path
    work: Path


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Bench:
    """Redirect both halves of the resolution into a scratch tree.

    ``_lerobot_home`` is patched rather than ``$HF_LEROBOT_HOME`` because
    lerobot reads that variable once at import; the working directory is
    redirected too, since a bare ``repo_id`` resolves relative to it.
    """
    home = tmp_path / "lerobot_home"
    home.mkdir()
    work = tmp_path / "workdir"
    work.mkdir()
    monkeypatch.setattr(dr, "_lerobot_home", lambda: home)
    monkeypatch.setattr(_Writer, "home", home)
    monkeypatch.chdir(work)
    module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    module.LeRobotDataset = _Writer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", module)
    return _Bench(home=home, work=work)


def _create(**kwargs: Any) -> _Writer:
    """Record through ``create``, with the schema every case shares.

    ``repo_id`` rides in ``kwargs`` rather than being named here so each case
    states its own id beside its own ``root`` - which is also what the
    shared-cache rule in ``test_recording_root_is_not_the_shared_cache`` reads.
    """
    recorder = DatasetRecorder.create(joint_names=["shoulder_pan"], **kwargs)
    dataset = recorder.dataset
    assert isinstance(dataset, _Writer)
    return dataset


def _plant_dataset(directory: Path) -> Path:
    """An existing LeRobotDataset: a ``meta/`` dir with an ``info.json`` in it."""
    (directory / "meta").mkdir(parents=True)
    info = directory / "meta" / "info.json"
    info.write_text('{"fps": 30}', encoding="utf-8")
    return info


# A repo_id that is itself a path, in both forms resolve_dataset_dir names.
PATH_LIKE = ("bare_name", "./dot_slash_name")


@pytest.mark.parametrize("repo_id", PATH_LIKE)
def test_the_dataset_is_written_into_the_directory_that_was_resolved(bench: _Bench, repo_id: str) -> None:
    """The writer's directory is the one ``resolve_dataset_dir`` named."""
    dataset = _create(repo_id=repo_id, root=None)

    assert dataset.root.resolve() == resolve_dataset_dir(repo_id, None).resolve()
    assert dataset.root.resolve() == (bench.work / Path(repo_id).name).resolve()


def test_an_overwrite_replaces_the_directory_it_deleted(bench: _Bench) -> None:
    """What ``overwrite=True`` deletes is what the dataset is then written into."""
    target = bench.work / "bare_name"
    target.mkdir()
    (target / "unrelated_notes.txt").write_text("a file the caller cares about", encoding="utf-8")

    dataset = _create(repo_id="bare_name", root=None, overwrite=True)

    assert dataset.root.resolve() == target.resolve(), (
        "overwrite deleted one directory and the dataset was written to another"
    )
    assert (target / "meta").is_dir()
    assert not (bench.home / "bare_name").exists()


def test_a_dataset_outside_the_resolved_directory_does_not_refuse_the_create(bench: _Bench) -> None:
    """An unrelated dataset under the dataset home neither blocks nor is touched.

    Pre-fix this create reached the writer's ``exist_ok=False`` mkdir on that
    very directory, so it died on the bare ``FileExistsError`` the guard exists
    to replace - for a dataset the caller had not addressed.
    """
    unrelated = _plant_dataset(bench.home / "bare_name")

    dataset = _create(repo_id="bare_name", root=None)

    assert dataset.root.resolve() == (bench.work / "bare_name").resolve()
    assert unrelated.read_text(encoding="utf-8") == '{"fps": 30}'


def test_an_owner_name_repo_id_still_resolves_under_the_dataset_home(bench: _Bench) -> None:
    """The ordinary id: both resolutions agreed already, and still do."""
    dataset = _create(repo_id="user/data", root=None)

    assert dataset.root.resolve() == (bench.home / "user" / "data").resolve()


def test_an_explicit_root_is_still_used_verbatim(bench: _Bench) -> None:
    """A caller-named root is the target for both halves, as before."""
    chosen = bench.work / "chosen_dir"

    dataset = _create(repo_id="user/data", root=str(chosen))

    assert dataset.root.resolve() == chosen.resolve()


def test_an_existing_dataset_at_the_target_still_names_overwrite_and_resume(bench: _Bench) -> None:
    """The actionable refusal survives: it is now aimed at the written directory."""
    _plant_dataset(bench.work / "bare_name")

    with pytest.raises(FileExistsError, match="overwrite=True.*resume|resume.*overwrite=True"):
        _create(repo_id="bare_name", root=None)
