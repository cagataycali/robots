# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A calibration write that fails leaves the measurement already on disk readable.

A calibration is not derived data. Its homing offsets and joint travel limits are
recorded by disabling torque and moving *one physical arm* by hand, so a stored
one that is lost costs the procedure, not a re-run.

Both writers of that store used to encode straight into the destination -
``save_calibration`` through ``json.dump``, ``restore_calibrations`` through
``shutil.copy2`` - which truncates the stored file before the first byte of the
replacement lands. A value the encoder rejects, or a filesystem that refuses the
write, therefore left a prefix where the measurement was. Nothing downstream says
so: ``calibration_exists`` answers ``True`` for a prefix, ``load_calibration``
answers ``None``, and the tool's ``view`` renders the path, size and timestamp
under ``status="success"`` with no motors in it.

``restore_calibrations`` is the sharper of the two, because it is the path a lost
calibration is recovered on: with ``overwrite=True`` a refused write destroyed the
calibration being replaced *and* did not install the backup, so an operator who
restored a backup over a working arm could end up with neither.

These cells pin the commit instead: the calibration is serialized in full before
the stored file is touched, committed through a temp file plus ``os.replace``, and
a failed write leaves the stored measurement byte-identical, loadable, and with no
uncommitted file beside it.

The filesystem refusal is a real one. ``RLIMIT_FSIZE`` is enforced by the kernel on
the write itself, which is what makes it reach both writers: ``shutil.copy2``
copies with ``sendfile`` on Linux, so a wrapper around Python's ``open`` never sees
it, and ``EFBIG`` is not one of the errnos ``shutil`` gives up on and retries.

``backup_calibrations`` also writes JSON, and is deliberately not in this roster:
its manifest goes into a directory it has just created, so there is no previously
stored document for a failed write to lose.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import json
import resource
import signal
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.tools.lerobot_calibrate import LeRobotCalibrationManager

#: The motors of an SO-101 follower, in the order lerobot records them.
_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

_DEVICE = ("robots", "so101_follower", "orange_arm")


def _measured_calibration() -> dict[str, dict[str, int]]:
    """A calibration in the shape the procedure produces: per-motor ints."""
    return {
        name: {
            "id": index + 1,
            "drive_mode": 0,
            "homing_offset": -1024 + index * 7,
            "range_min": 800 + index,
            "range_max": 3200 - index,
        }
        for index, name in enumerate(_MOTORS)
    }


def _bus_reading(value: int) -> Any:
    """A motor reading as the NumPy scalar a bus read produces.

    Annotated ``Any`` because that is the honest static picture: ``CalibrationData``
    is a type alias, so nothing between a caller holding this and the encoder
    rejecting it narrows the value to ``int``.
    """
    return np.int64(value)


@pytest.fixture
def stored(tmp_path: Path) -> tuple[LeRobotCalibrationManager, Path, bytes]:
    """A manager with one calibration already measured and stored."""
    manager = LeRobotCalibrationManager(tmp_path)
    assert manager.save_calibration(*_DEVICE, _measured_calibration()) is True
    path = manager.get_calibration_path(*_DEVICE)
    return manager, path, path.read_bytes()


@pytest.fixture
def refusing_filesystem() -> Iterator[Callable[[int], contextlib.AbstractContextManager[None]]]:
    """Refuse any write that would take a file past ``ceiling`` bytes.

    A real ``RLIMIT_FSIZE`` ceiling rather than a patched writer, so the refusal
    reaches whichever call the implementation makes. The default disposition for
    ``SIGXFSZ`` terminates the process, so it is ignored for the duration and the
    previous handler and limit are both put back - the window is one call wide on
    purpose, because the limit applies to every write this process makes.
    """

    @contextlib.contextmanager
    def _ceiling(ceiling: int) -> Iterator[None]:
        soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
        previous = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (ceiling, hard))
        try:
            yield
        finally:
            resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
            signal.signal(signal.SIGXFSZ, previous)

    yield _ceiling


def _assert_measurement_survived(manager: LeRobotCalibrationManager, path: Path, before: bytes, how: str) -> None:
    """The stored calibration is the one that was measured, and it stands alone."""
    after = path.read_bytes()
    assert after == before, f"{how} left the stored measurement changed: {after[:80]!r}"
    assert json.loads(after), f"{how} left the stored measurement unparseable"
    loaded = manager.load_calibration(*_DEVICE)
    assert loaded is not None, f"{how} left a calibration that reads as absent"
    assert set(loaded) == set(_MOTORS), f"{how} lost motors: {sorted(loaded)}"
    # An uncommitted file must not be left under a name the ``*.json`` globs that
    # enumerate this store would pick up as a calibration of its own.
    strays = sorted(entry.name for entry in path.parent.iterdir() if entry.name != path.name)
    assert strays == [], f"{how} left an uncommitted file beside the calibration: {strays}"


def test_a_value_json_cannot_encode_leaves_the_stored_calibration_intact(
    stored: tuple[LeRobotCalibrationManager, Path, bytes],
) -> None:
    """A NumPy scalar in a re-measured calibration is refused before the stored one is touched.

    ``CalibrationData`` is a type alias, not a runtime check, so a reading that is
    not a plain ``int`` reaches the writer from any caller. ``json.dump`` encodes
    into the stream it is given, so such a value used to raise only after a prefix
    of the new calibration had replaced the stored one.
    """
    manager, path, before = stored
    remeasured = {name: dict(motor) for name, motor in _measured_calibration().items()}
    remeasured["wrist_roll"]["homing_offset"] = _bus_reading(-1031)

    assert manager.save_calibration(*_DEVICE, remeasured) is False
    _assert_measurement_survived(manager, path, before, "a rejected value")


@pytest.mark.parametrize("writer", ["save", "restore"])
def test_a_write_the_filesystem_refuses_leaves_the_stored_calibration_intact(
    stored: tuple[LeRobotCalibrationManager, Path, bytes],
    refusing_filesystem: Callable[[int], contextlib.AbstractContextManager[None]],
    writer: str,
) -> None:
    """Both writers of the store keep the measurement when the write cannot complete.

    Parametrized over which one performs it because both replace the same file and
    both used to truncate it first. ``restore`` is driven with ``overwrite=True``,
    which is the recovery case: the calibration it destroyed was the one it was
    asked to write over, and the backup did not land either.
    """
    manager, path, before = stored
    ceiling = len(before) // 2

    if writer == "save":
        with refusing_filesystem(ceiling):
            reported: object = manager.save_calibration(*_DEVICE, _measured_calibration())
        assert reported is False
    else:
        ok, _where, count = manager.backup_calibrations(output_dir=path.parent.parent.parent / "bk")
        assert (ok, count) == (True, 1), "the backup this restore reads was not written"
        with refusing_filesystem(ceiling):
            ok, message, restored = manager.restore_calibrations(path.parent.parent.parent / "bk", overwrite=True)
        assert ok is False and restored == 0, message

    _assert_measurement_survived(manager, path, before, f"a refused {writer}")


def test_an_uncommitted_calibration_is_not_named_like_one(
    stored: tuple[LeRobotCalibrationManager, Path, bytes],
) -> None:
    """A temp file that outlives its process is not enumerated as a calibration.

    The commit removes its own temp file, but a kill between the write and the
    rename cannot. The name is taken from the commit rather than restated here, so
    the choice of suffix is what is graded: were it to keep ``.json``, the
    ``*.json`` globs that enumerate this store would read a partial write as a
    measurement, and ``backup_calibrations`` - which reads that enumeration -
    would preserve it as one.
    """
    from strands_robots.tools.lerobot_calibrate import _commit_calibration

    manager, path, _before = stored
    written: list[Path] = []

    def _record_then_die(tmp: Path) -> None:
        written.append(tmp)
        tmp.write_bytes(b'{"partial":')
        raise RuntimeError("killed before the rename")

    with pytest.raises(RuntimeError):
        _commit_calibration(path, _record_then_die)
    assert written, "the commit handed its fill no path to write"
    # The commit cleaned its own up; a kill at the same point could not, so put
    # one back under exactly the name the commit chose.
    written[0].write_bytes(b'{"partial":')

    structure = manager.get_calibration_structure()
    assert structure["robots"]["so101_follower"] == ["orange_arm"], (
        f"an uncommitted write was enumerated as a calibration: {structure}"
    )
    ok, where, count = manager.backup_calibrations(output_dir=path.parent.parent.parent / "bk2")
    assert (ok, count) == (True, 1), f"an uncommitted write was backed up as a calibration: {where}"


def test_a_blocked_write_target_is_reported_rather_than_raised(tmp_path: Path) -> None:
    """CONTROL: a write that cannot start at all still reports failure.

    Holds either way. It is the fail-soft contract callers already had, kept so
    this change is about what survives a failure and not about how one is
    reported.
    """
    manager = LeRobotCalibrationManager(tmp_path)
    manager.get_calibration_path(*_DEVICE).mkdir(parents=True)
    assert manager.save_calibration(*_DEVICE, _measured_calibration()) is False


def test_an_ordinary_save_and_restore_still_round_trip(tmp_path: Path) -> None:
    """CONTROL: the working paths still store and recover a calibration.

    Holds either way, so a commit that refused every write - or one that never
    renamed its temp file into place - would not satisfy it.
    """
    manager = LeRobotCalibrationManager(tmp_path)
    measured = _measured_calibration()
    assert manager.save_calibration(*_DEVICE, measured) is True
    assert manager.load_calibration(*_DEVICE) == measured

    ok, where, count = manager.backup_calibrations(output_dir=tmp_path / "bk")
    assert (ok, count) == (True, 1)

    changed = {name: dict(motor) for name, motor in measured.items()}
    changed["gripper"]["range_max"] = 1
    assert manager.save_calibration(*_DEVICE, changed) is True
    assert manager.load_calibration(*_DEVICE) == changed

    ok, message, restored = manager.restore_calibrations(Path(where), overwrite=True)
    assert (ok, restored) == (True, 1), message
    assert manager.load_calibration(*_DEVICE) == measured, "the restored calibration is not the backed-up one"


def test_both_calibration_writers_commit_through_the_one_owner() -> None:
    """Neither writer spells the commit itself, so the sequence lives in one place.

    Graded on the calls each writer makes rather than on the text: the rule is
    which function performs the replacement, and a second copy of ``tmp`` plus
    ``os.replace`` is exactly the drift that leaves one of the two destructive.
    """
    for method, forbidden in (
        (LeRobotCalibrationManager.save_calibration, "json.dump"),
        (LeRobotCalibrationManager.restore_calibrations, "shutil.copy2"),
    ):
        source = textwrap.dedent(inspect.getsource(method))
        called: set[str] = {
            ast.unparse(node.func) for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
        }
        assert "_commit_calibration" in called, (
            f"{method.__qualname__} does not commit through the shared owner: {sorted(called)}"
        )
        assert forbidden not in called, (
            f"{method.__qualname__} still writes {forbidden} straight into the stored calibration"
        )


def test_the_commit_owner_removes_its_temp_file_when_the_commit_does_not_happen(tmp_path: Path) -> None:
    """A failure inside the owner leaves neither a changed destination nor a temp file."""
    from strands_robots.tools.lerobot_calibrate import _commit_calibration

    destination = tmp_path / "arm.json"
    destination.write_text('{"kept": true}', encoding="utf-8")

    def _fill(tmp: Path) -> Any:
        tmp.write_text("partial", encoding="utf-8")
        raise RuntimeError("the write could not finish")

    with pytest.raises(RuntimeError):
        _commit_calibration(destination, _fill)

    assert json.loads(destination.read_text()) == {"kept": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["arm.json"]
