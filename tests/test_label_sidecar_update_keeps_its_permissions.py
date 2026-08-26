"""Updating the label sidecar leaves its permissions alone.

:func:`strands_robots.episode_labels._write_document` replaces the sidecar by
writing a ``mkstemp`` temp file and ``os.replace``-ing it into place. A
``mkstemp`` file is owner-only, so the rename used to carry ``0o600`` onto the
destination and silently narrow a sidecar that arrived group- or
world-readable. Travelling with the dataset directory is what the sidecar is
for, and every way it travels - a copy, a ``tar -x``, a Hub download, a clone -
lands it at the reader's umask, so the wider mode is the ordinary case rather
than an unusual one. The write reported success either way; the cost showed up
later, as a ``PermissionError`` from :func:`read_labels` or
:func:`filter_episodes` on a dataset whose labels were readable a moment
before.

What these cells hold:

- The regression: an update through either writer leaves the mode it found.
  Parametrized over the modes a shared dataset actually carries, and over both
  writers, discovered from the module rather than listed so a third writer is
  held to the rule the hour it lands.
- The structural half: the mode is applied to the temp file *before* the
  rename, not to the destination after it, so the sidecar is never momentarily
  readable to fewer callers than it was.
- What must not change: the atomicity contract the construct exists for - a
  failed write leaves no temp file and the previous sidecar byte-identical -
  and the content of a successful update.
- The premise: ``mkstemp`` really does create an owner-only file, so the
  narrowing was a property of the construct and not of one platform; and
  :func:`strands_robots.simulation.safe_output.atomic_write_bytes` really does
  hold its own output to ``0o600`` on purpose, which is the contrast that keeps
  this change to the sidecar.
"""

from __future__ import annotations

import ast
import inspect
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import strands_robots.episode_labels as labels

# Modes a dataset's own files carry in the wild: the default umask, a
# group-writable shared mount, a group-readable team dataset, and a read-only
# archive extraction. Every one is wider than ``mkstemp``'s ``0o600`` in at
# least one bit, so each would have been narrowed.
_SHARED_MODES = (0o644, 0o664, 0o640, 0o444)

# The two members individual cells name. Taken from the grid rather than
# respelled, so the modes these cells stage and the modes the parametrized cells
# sweep cannot drift apart.
_GROUP_READABLE = _SHARED_MODES[0]
_READ_ONLY = _SHARED_MODES[3]

# The private mode the temp file arrives with. Named here rather than read from
# the module so a cell that grades the module's behaviour is not comparing the
# module against itself.
_MKSTEMP_MODE = 0o600


def _mode(path: Path) -> int:
    """Permission bits of ``path``."""
    return stat.S_IMODE(path.stat().st_mode)


def _seeded_dataset() -> Path:
    """A dataset root whose sidecar exists and carries one verdict."""
    root = Path(tempfile.mkdtemp())
    labels.record_deterministic_verdicts(root, [{"episode": 0, "success": True}])
    return root


def _sidecar_writers() -> dict[str, Callable[[Path], Any]]:
    """Every public function that rewrites the sidecar, discovered from the module.

    Derived from the module's own call graph - a function whose body calls
    ``_write_document`` - rather than listed, so a writer added later is held to
    the rule without anyone remembering to add it here. Each value drives that
    writer against a dataset already carrying episode 0's verdict.
    """
    drives: dict[str, Callable[[Path], Any]] = {
        "record_deterministic_verdicts": lambda root: labels.record_deterministic_verdicts(
            root, [{"episode": 1, "success": False, "failure": True}]
        ),
        "annotate_episode": lambda root: labels.annotate_episode(root, 0, quality="high", note="looked clean"),
    }
    return drives


def _writers_in_the_module() -> set[str]:
    """Every public function whose body rewrites the sidecar, read off the module.

    Derived from the module's own call graph - a public function that calls
    ``_write_document`` - rather than listed, so a writer added later is held to
    the permission rule without anyone remembering to extend this file.
    """
    tree = ast.parse(inspect.getsource(labels))
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and not node.name.startswith("_")
        and any(
            isinstance(call, ast.Call) and getattr(call.func, "id", None) == "_write_document"
            for call in ast.walk(node)
        )
    }


_WRITERS = _sidecar_writers()


def _write_document_body() -> ast.FunctionDef:
    """The ``_write_document`` definition, for the structural cells."""
    tree = ast.parse(inspect.getsource(labels))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_write_document")


class TestAnUpdateKeepsTheSidecarReadableToTheSameCallers:
    """The regression: a rewrite leaves the mode it found."""

    @pytest.mark.parametrize("writer", sorted(_WRITERS), ids=sorted(_WRITERS))
    @pytest.mark.parametrize("start", _SHARED_MODES, ids=[oct(m) for m in _SHARED_MODES])
    def test_the_mode_survives_the_rewrite(self, writer: str, start: int) -> None:
        root = _seeded_dataset()
        sidecar = labels.labels_path(root)
        os.chmod(sidecar, start)

        _WRITERS[writer](root)

        assert _mode(sidecar) == start, (
            f"{writer} left the sidecar at {oct(_mode(sidecar))} where it found {oct(start)}; "
            "a caller that could read the labels before the write cannot read them after."
        )

    @pytest.mark.parametrize("writer", sorted(_WRITERS), ids=sorted(_WRITERS))
    def test_a_group_readable_sidecar_stays_group_readable(self, writer: str) -> None:
        """The bit that decides whether another account can read the labels."""
        root = _seeded_dataset()
        sidecar = labels.labels_path(root)
        os.chmod(sidecar, _GROUP_READABLE)

        _WRITERS[writer](root)

        assert _mode(sidecar) & (stat.S_IRGRP | stat.S_IROTH), (
            "the sidecar is owner-only after the write, so every other account on a shared "
            "dataset now takes a PermissionError from read_labels"
        )

    def test_a_sidecar_with_no_permission_bits_keeps_none(self) -> None:
        """``0o000`` is the one mode that is falsy, so a truthiness test would skip it.

        Skipping it would hand the sidecar ``mkstemp``'s ``0o600`` - a widening
        rather than a narrowing, but still this function deciding who may read
        the dataset. Driven through the writer itself: the public entry points
        read the sidecar before rewriting it, so they cannot reach a mode that
        denies them the read.
        """
        root = _seeded_dataset()
        sidecar = labels.labels_path(root)
        document = labels.read_labels(root)
        os.chmod(sidecar, 0o000)

        labels._write_document(sidecar, document)

        assert _mode(sidecar) == 0o000, f"a mode-less sidecar became {oct(_mode(sidecar))}"

    def test_a_read_only_sidecar_is_still_updatable_and_stays_read_only(self) -> None:
        """``os.replace`` needs the directory, not the file, so 0o444 is writable.

        The mode is carried across rather than reset, so a dataset published
        read-only keeps that posture through an annotation.
        """
        root = _seeded_dataset()
        sidecar = labels.labels_path(root)
        os.chmod(sidecar, _READ_ONLY)

        labels.annotate_episode(root, 0, quality="medium")

        assert _mode(sidecar) == _READ_ONLY, f"read-only sidecar became {oct(_mode(sidecar))}"
        assert labels.read_labels(root)["episodes"]["0"]["judge"]["quality"] == "medium"


class TestEveryWriterIsCovered:
    """The drive table names every function that rewrites the sidecar."""

    def test_the_drive_table_covers_every_writer_in_the_module(self) -> None:
        derived = _writers_in_the_module()
        assert derived == set(_WRITERS), (
            f"the module writes the sidecar from {sorted(derived)}; this file drives "
            f"{sorted(_WRITERS)}. Add the new writer to the drive table so the permission "
            "rule is graded for it too."
        )

    def test_the_module_has_at_least_the_two_known_writers(self) -> None:
        """Non-vacuity: an empty derivation would make the cell above trivially true."""
        assert {"record_deterministic_verdicts", "annotate_episode"} <= _writers_in_the_module()


class TestTheModeIsAppliedBeforeTheRename:
    """Applied to the temp file, so no window exists where the mode is wrong."""

    def test_the_chmod_precedes_the_replace(self) -> None:
        body = _write_document_body()
        calls = [
            ".".join(filter(None, [getattr(node.func.value, "id", None), node.func.attr]))
            for node in ast.walk(body)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        linenos = {
            f"{getattr(node.func.value, 'id', '')}.{node.func.attr}": node.lineno
            for node in ast.walk(body)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "os.chmod" in calls, (
            "_write_document does not set a mode, so the rename carries mkstemp's owner-only mode onto the destination"
        )
        assert linenos["os.chmod"] < linenos["os.replace"], (
            "the mode is applied after os.replace, which leaves a window where the sidecar is "
            "readable to fewer callers than it was"
        )

    def test_the_mode_is_applied_to_the_temp_file(self) -> None:
        """Chmod-ing ``path`` instead would be that window, whatever the ordering."""
        body = _write_document_body()
        chmod = next(
            (
                node
                for node in ast.walk(body)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "chmod"
            ),
            None,
        )
        assert chmod is not None, "_write_document sets no mode, so there is no target to grade"
        target = ast.unparse(chmod.args[0])
        assert target != "path", "os.chmod(path, ...) mutates the live sidecar rather than its replacement"
        assert "tmp" in target, f"os.chmod applies to {target!r}, which is not the temp file"


class TestWhatIsUnchanged:
    """The atomicity contract the construct exists for, and the content."""

    def test_a_failed_write_leaves_no_temp_file_and_the_sidecar_intact(self) -> None:
        root = _seeded_dataset()
        sidecar = labels.labels_path(root)
        os.chmod(sidecar, _GROUP_READABLE)
        before = sidecar.read_bytes()

        with pytest.raises(TypeError):
            # A set is not JSON-serialisable, so json.dump raises mid-write.
            labels.record_deterministic_verdicts(root, [{"episode": 2, "success": True, "seed": {1, 2}}])

        assert [p.name for p in root.iterdir() if p.name.endswith(".tmp")] == []
        assert sidecar.read_bytes() == before
        assert _mode(sidecar) == _GROUP_READABLE

    def test_a_successful_update_writes_the_content(self) -> None:
        root = _seeded_dataset()
        os.chmod(labels.labels_path(root), _GROUP_READABLE)

        labels.annotate_episode(root, 0, quality="low", note="drifted")

        judge = labels.read_labels(root)["episodes"]["0"]["judge"]
        assert judge["quality"] == "low"
        assert judge["note"] == "drifted"

    def test_a_sidecar_this_module_creates_is_owner_only(self) -> None:
        """The create case is untouched: this function does not widen anything."""
        root = Path(tempfile.mkdtemp())

        labels.record_deterministic_verdicts(root, [{"episode": 0, "success": True}])

        assert _mode(labels.labels_path(root)) == _MKSTEMP_MODE

    def test_an_already_private_sidecar_stays_private(self) -> None:
        """Carrying the mode across is not a widening."""
        root = _seeded_dataset()
        sidecar = labels.labels_path(root)
        assert _mode(sidecar) == _MKSTEMP_MODE

        labels.annotate_episode(root, 0, quality="high")

        assert _mode(sidecar) == _MKSTEMP_MODE


class TestThePremise:
    """Why the narrowing was a property of the construct, and why it stops here."""

    def test_mkstemp_creates_an_owner_only_file(self) -> None:
        fd, name = tempfile.mkstemp(dir=tempfile.mkdtemp())
        os.close(fd)
        assert _mode(Path(name)) == _MKSTEMP_MODE, (
            "mkstemp is not owner-only on this platform, so the modes these cells "
            "compare against would not be the ones the construct produces"
        )

    def test_the_sibling_atomic_writer_holds_its_output_private_on_purpose(self) -> None:
        """The contrast that keeps this change to the sidecar.

        ``safe_output.atomic_write_bytes`` chooses ``0o600`` and says why - its
        output roots are private to the running user. A label sidecar is a
        dataset artifact, so the same choice does not transfer.
        """
        safe_output = pytest.importorskip("strands_robots.simulation.safe_output")
        source = inspect.getsource(safe_output.atomic_write_bytes)
        assert "os.chmod(path, 0o600)" in source
        target = Path(tempfile.mkdtemp()) / "payload.bin"
        safe_output.atomic_write_bytes(target, b"x")
        assert _mode(target) == _MKSTEMP_MODE
