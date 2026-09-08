# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A pose library that cannot be written keeps the poses it already held.

``PoseManager`` rewrites the WHOLE library on every change: ``store_pose`` and
``delete_pose`` alter one entry and store the document back. Encoding that
document straight into its own destination makes a rejected write destructive -
the stream is truncated first, so an encoder error or a full disk leaves a
prefix where the library was, and ``_load_poses`` reports an unparseable file as
*no poses*. One failed save therefore lost every pose the arm had, and
``pose_tool``'s ``store_pose`` still reported the new pose as stored - one line
below the refusal that declines to persist a partially-read arm because "a
stored pose is a named posture every later load_pose drives towards".

These cells pin the commit instead: the document is serialized before the
destination is touched, committed through a temp file plus ``os.replace``, and a
failure is reported to the caller with the stored library - and the in-memory
one - left as they were.
"""

from __future__ import annotations

import ast
import builtins
import errno
import inspect
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.tools.pose_tool import PoseManager, pose_tool

_REAL_OPEN = io.open

_MOTORS = {"shoulder_pan": 1.0, "shoulder_lift": 2.0, "elbow_flex": 3.0}


class _FullDisk:
    """A text file that accepts ``allowance`` characters, then reports ENOSPC.

    A full disk fails the write that exhausts it *after* the bytes before it
    landed, which is what makes an in-place rewrite destructive and a
    temp-file commit harmless. Everything else delegates, so the file the
    implementation chose to open is the file that fills up.
    """

    def __init__(self, fh: Any, allowance: int) -> None:
        self._fh = fh
        self._left = allowance

    def write(self, data: str) -> int:
        if len(data) <= self._left:
            self._left -= len(data)
            return self._fh.write(data)
        if self._left:
            self._fh.write(data[: self._left])
            self._left = 0
        raise OSError(errno.ENOSPC, "No space left on device")

    def __enter__(self) -> _FullDisk:
        return self

    def __exit__(self, *exc: object) -> None:
        self._fh.close()

    def close(self) -> None:
        self._fh.close()

    def flush(self) -> None:
        self._fh.flush()


@pytest.fixture
def full_disk(monkeypatch: pytest.MonkeyPatch):
    """Fill the filesystem under any file whose name starts with a prefix.

    Both spellings are covered on purpose - the library itself and any
    ``.tmp`` sibling a commit writes - so the injection describes the disk
    rather than the implementation, and hits whichever file is written.
    """

    def _install(prefix: str, allowance: int = 40) -> None:
        def _open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            fh = _REAL_OPEN(file, mode, *args, **kwargs)
            if "w" in mode and Path(str(file)).name.startswith(prefix):
                return _FullDisk(fh, allowance)
            return fh

        monkeypatch.setattr(io, "open", _open)
        monkeypatch.setattr(builtins, "open", _open)

    return _install


@pytest.fixture
def write_opens(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the name of every file opened for writing."""
    seen: list[str] = []

    def _open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if "w" in mode or "a" in mode:
            seen.append(Path(str(file)).name)
        return _REAL_OPEN(file, mode, *args, **kwargs)

    monkeypatch.setattr(io, "open", _open)
    monkeypatch.setattr(builtins, "open", _open)
    return seen


def _stocked(robot_id: str = "arm") -> PoseManager:
    """A manager holding two stored poses, committed to disk."""
    mgr = PoseManager(robot_id)
    mgr.store_pose("home", dict(_MOTORS))
    mgr.store_pose("ready", dict(_MOTORS), description="staged")
    return mgr


class TestTheStoredLibrarySurvivesAFailedWrite:
    def test_a_full_disk_leaves_every_pose_it_already_stored(self, cwd_tmp, full_disk) -> None:
        mgr = _stocked()
        before = mgr.pose_file.read_bytes()
        full_disk(mgr.pose_file.name)

        with pytest.raises(OSError) as raised:
            mgr.store_pose("third", dict(_MOTORS))

        assert raised.value.errno == errno.ENOSPC
        assert mgr.pose_file.read_bytes() == before
        assert sorted(json.loads(before)) == ["home", "ready"]
        # The library that reloads is the library that was stored, and the
        # in-memory one agrees with it rather than holding a pose no reader can
        # find.
        assert sorted(PoseManager("arm").list_poses()) == ["home", "ready"]
        assert sorted(mgr.list_poses()) == ["home", "ready"]

    def test_no_temp_file_is_left_behind(self, cwd_tmp, full_disk) -> None:
        mgr = _stocked()
        full_disk(mgr.pose_file.name)
        with pytest.raises(OSError):
            mgr.store_pose("third", dict(_MOTORS))
        assert [p.name for p in mgr.storage_dir.iterdir()] == [mgr.pose_file.name]

    def test_a_full_disk_leaves_a_deletion_undone(self, cwd_tmp, full_disk) -> None:
        mgr = _stocked()
        before = mgr.pose_file.read_bytes()
        full_disk(mgr.pose_file.name)

        with pytest.raises(OSError):
            mgr.delete_pose("home")

        assert mgr.pose_file.read_bytes() == before
        assert sorted(mgr.list_poses()) == ["home", "ready"]

    def test_a_value_json_cannot_encode_is_refused_before_the_library_is_touched(self, cwd_tmp, write_opens) -> None:
        mgr = _stocked()
        before = mgr.pose_file.read_bytes()
        write_opens.clear()

        # A joint angle read through NumPy is the way this arrives: np.float32
        # is not a JSON type, and json.dump raises on it mid-document.
        numpy_angle: dict[str, Any] = {"shoulder_pan": np.float32(1.5)}
        with pytest.raises(ValueError, match="not JSON-serializable"):
            mgr.store_pose("third", numpy_angle)

        assert write_opens == [], f"the library was opened for writing: {write_opens}"
        assert mgr.pose_file.read_bytes() == before
        assert sorted(mgr.list_poses()) == ["home", "ready"]


class TestTheToolDoesNotReportAStoreItCouldNotMake:
    def test_store_pose_reports_the_library_it_could_not_store(self, cwd_tmp, reading_serial, full_disk) -> None:
        first = pose_tool(action="store_pose", robot_id="hw_arm", port="/dev/ttyTEST", pose_name="home")
        assert first["status"] == "success"
        library = Path.cwd() / ".strands_robots" / "poses" / "hw_arm_poses.json"
        full_disk(library.name)

        result = pose_tool(action="store_pose", robot_id="hw_arm", port="/dev/ttyTEST", pose_name="ready")

        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert text.startswith("Not storing 'ready'"), text
        assert "1 previously stored poses are unchanged" in text
        assert sorted(json.loads(library.read_text(encoding="utf-8"))) == ["home"]

    def test_delete_pose_reports_the_library_it_could_not_store(self, cwd_tmp, full_disk) -> None:
        mgr = _stocked("hw_arm")
        full_disk(mgr.pose_file.name)

        result = pose_tool(action="delete_pose", robot_id="hw_arm", pose_name="home")

        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert text.startswith("Not deleting 'home'"), text
        assert sorted(json.loads(mgr.pose_file.read_text(encoding="utf-8"))) == ["home", "ready"]


class TestTheCommitItself:
    def test_a_stored_pose_round_trips_and_leaves_no_temp_file(self, cwd_tmp) -> None:
        """The control: an ordinary store still lands, whole, with nothing beside it."""
        mgr = _stocked()
        mgr.delete_pose("ready")
        reloaded = PoseManager("arm")
        assert sorted(reloaded.list_poses()) == ["home"]
        home = reloaded.get_pose("home")
        assert home is not None and home.positions == _MOTORS
        assert [p.name for p in mgr.storage_dir.iterdir()] == [mgr.pose_file.name]

    def test_the_library_is_never_encoded_into_its_own_destination(self) -> None:
        """No ``json.dump`` in the module: encoding into the destination is the defect."""
        source = Path(inspect.getsourcefile(PoseManager) or "")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        streamed = [
            f"line {node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "dump"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "json"
        ]
        assert streamed == [], (
            f"json.dump encodes into the stream it is given, so these writes ({streamed}) truncate the "
            "pose library before a value they cannot encode can be reported; serialize with json.dumps "
            "and commit through os.replace instead"
        )
