# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pose library's directory is created by the one write that needs it.

``PoseManager.__init__`` used to run::

    self.storage_dir.mkdir(parents=True, exist_ok=True)

and every verb of ``pose_tool`` constructs a manager - including the ones that
only read the library (``list_poses``, ``show_pose``, ``delete_pose``) and
``emergency_stop``, which is never gated because stopping is never gated. So
asking what poses exist created ``.strands_robots/poses`` in whatever directory
the caller happened to be in, and where that directory is read-only - an edge
container's rootfs - the same construction raised ``PermissionError`` out of
line 1324 of the tool, which wraps it for ``ValueError`` only: an emergency stop
refused by an exception this tool's error envelope does not cover, because the
arm could not be told about a pose file nobody asked to write.

The cells below grade both halves: a verb that only reads creates nothing and
still answers, and the writer creates the directory it needs or reports the one
it could not make.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from strands_robots.tools.pose_tool import PoseManager, pose_tool

_MOTORS = {"shoulder_pan": 0.0, "gripper": 10.0}

#: An explicit fake port. ``pose_tool``'s ``port`` defaults to ``/dev/ttyACM0``,
#: so a portless call would reach an arm plugged into the machine running this.
_PORT = "/dev/ttyTEST"

#: Every library verb, with the answer it owes on an empty library. None of them
#: stores a pose, so none of them has a directory to create.
_READING_VERBS = [
    pytest.param({"action": "list_poses"}, "success", "No poses stored", id="list_poses"),
    pytest.param({"action": "show_pose", "pose_name": "home"}, "error", "not found", id="show_pose"),
    pytest.param({"action": "delete_pose", "pose_name": "home"}, "error", "not found", id="delete_pose"),
]


def _unmakeable(root: Path) -> Path:
    """Put a FILE where the pose directory's parent belongs, and name the pair.

    A file rather than a permission bit: ``mkdir(parents=True)`` underneath one
    raises ``NotADirectoryError`` on every platform, needs no ``chmod`` the
    session then has to undo, and stands in for the read-only rootfs of an edge
    container without the suite needing one.
    """
    (root / ".strands_robots").write_text("", encoding="utf-8")
    return root / ".strands_robots" / "poses"


def _text(result: dict[str, Any]) -> str:
    """Every ``text`` field of a tool result, joined."""
    return "\n".join(item["text"] for item in result["content"] if "text" in item)


class TestAVerbThatOnlyReadsCreatesNothing:
    @pytest.mark.parametrize(("call", "status", "says"), _READING_VERBS)
    def test_the_directory_is_not_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call: dict[str, Any], status: str, says: str
    ) -> None:
        """Reading a library that was never stored leaves the tree as it was."""
        monkeypatch.chdir(tmp_path)

        result = pose_tool(robot_id="arm", **call)

        assert result["status"] == status
        assert says in _text(result)
        assert not (tmp_path / ".strands_robots").exists()

    @pytest.mark.parametrize(("call", "status", "says"), _READING_VERBS)
    def test_it_answers_where_the_directory_cannot_be_made(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call: dict[str, Any], status: str, says: str
    ) -> None:
        """Pre-fix each of these raised ``NotADirectoryError`` from the ctor."""
        monkeypatch.chdir(tmp_path)
        _unmakeable(tmp_path)

        result = pose_tool(robot_id="arm", **call)

        assert result["status"] == status
        assert says in _text(result)


class TestStoppingIsNotBlockedByThePoseLibrary:
    def test_emergency_stop_reaches_the_bus_where_the_directory_cannot_be_made(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_serial: list[Any]
    ) -> None:
        """The one action taken while an arm is moving must not need a directory.

        The bytes that de-energize the motors are pinned in
        ``test_pose_tool_emergency_stop``; what is graded here is that the stop
        runs at all - pre-fix the tool raised before reaching the bus, so no
        packet was written and the caller got an exception instead of a verdict.
        """
        monkeypatch.chdir(tmp_path)
        _unmakeable(tmp_path)

        result = pose_tool(action="emergency_stop", robot_id="arm", port=_PORT)

        assert result["status"] == "success"
        assert len(fake_serial) == 1, "the bus must actually be opened"
        assert fake_serial[0].writes, "no packet reached the motors"


class TestTheWriterMakesTheDirectoryItNeeds:
    def test_a_stored_pose_creates_the_directory_and_reloads(self, tmp_path: Path) -> None:
        """Construction creates nothing; the store creates it and round-trips."""
        store = tmp_path / "absent" / "poses"
        manager = PoseManager("arm", storage_dir=store)
        assert not store.exists()

        manager.store_pose("home", dict(_MOTORS))

        assert manager.pose_file.exists()
        assert PoseManager("arm", storage_dir=store).list_poses() == ["home"]

    def test_a_directory_it_cannot_make_is_reported_as_that_directory(self, tmp_path: Path) -> None:
        """The refusal names the directory, not the temp file never written."""
        store = _unmakeable(tmp_path)
        manager = PoseManager("arm", storage_dir=store)

        with pytest.raises(OSError) as raised:
            manager.store_pose("home", dict(_MOTORS))

        assert str(store) in str(raised.value)
        assert not manager.pose_file.with_suffix(".json.tmp").exists()
