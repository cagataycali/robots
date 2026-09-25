"""Importing a session tool does not create a directory beside the caller.

``strands_robots/tools/_process_stop.py`` made its session directory as its
module body ran::

    SESSION_DIR = Path.cwd() / ".strands_robots/.sessions"
    SESSION_DIR.mkdir(parents=True, exist_ok=True)

so ``import strands_robots.tools.lerobot_train`` -- what reaching either session
verb costs -- created two directories in whatever directory the caller happened
to be standing in, before any verb was called and for a caller who may only have
been listing sessions. Measured in an empty temp directory: the bare ``import
strands_robots`` left it empty, importing the training tool left
``.strands_robots/.sessions`` behind.

Where that directory may not be written the same import does not litter, it
FAILS::

    File ".../strands_robots/tools/_process_stop.py", line 89, in <module>
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
    PermissionError: [Errno 13] Permission denied: '/tmp/ro/.strands_robots'

which is a container started with a read-only working directory, and an import
that raises cannot be degraded around: the handler a caller would write sits
around the verb, which was never reached.

A directory a writer needs is the writer's to make, so the two doors that write
into it make it -- :func:`~strands_robots.tools._process_stop.session_log_path`,
whose every caller opens the path it returns for writing, and
:func:`~strands_robots.tools._process_stop.store_sessions`.

``TestMaterializingAToolDoesNotConfigureTheHostProcess`` in
``tests/tools/test_tools_lazy_import.py`` grades the shape - no tool module calls
``mkdir`` at its top level - and held this store as its one declared exception.
Here is the behaviour that shape stands for, from both sides: the import leaves
nothing behind and survives a cwd it may not write, and each door still makes the
directory when there is a session to record, which every other session test
takes for granted by making the directory in its own fixture.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("psutil")

from strands_robots.tools import _process_stop  # noqa: E402

#: The modules a caller loads to reach a session verb. ``_process_stop`` holds the
#: directory and both tools import it, so importing either one used to pay for it.
SESSION_TOOL_MODULES = (
    "strands_robots.tools._process_stop",
    "strands_robots.tools.lerobot_train",
    "strands_robots.tools.lerobot_teleoperate",
)


def _import_only(module: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Import ``module`` in a fresh interpreter whose working directory is ``cwd``.

    A subprocess because the import itself is the subject: this process imported
    these modules at collection time, in the repository's own directory, so an
    assertion made here would grade a module body that has already run.

    Args:
        module: Dotted module name to import, and nothing else.
        cwd: The working directory the interpreter starts in.

    Returns:
        The finished process, with output captured as text.
    """
    return subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("module", SESSION_TOOL_MODULES)
def test_importing_a_session_tool_leaves_the_working_directory_empty(module: str, tmp_path: Path) -> None:
    """Nothing appears beside a caller who has not started a session."""
    done = _import_only(module, tmp_path)

    assert done.returncode == 0, f"import {module} failed: {done.stderr}"
    assert sorted(p.name for p in tmp_path.iterdir()) == [], f"import {module} wrote to the caller's working directory"


@pytest.mark.parametrize("module", SESSION_TOOL_MODULES)
def test_a_session_tool_imports_where_the_working_directory_cannot_be_written(module: str, tmp_path: Path) -> None:
    """A read-only working directory is a deployment, not a broken install."""
    read_only = tmp_path / "read-only"
    read_only.mkdir()
    read_only.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(OSError):
            (read_only / "probe").mkdir()

        done = _import_only(module, read_only)
    finally:
        read_only.chmod(stat.S_IRWXU)

    assert done.returncode == 0, f"import {module} needed a writable working directory: {done.stderr}"


def test_the_log_path_makes_the_directory_it_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first door: a log path is asked for only when a session is starting."""
    session_dir = tmp_path / "never-created" / ".sessions"
    monkeypatch.setattr(_process_stop, "SESSION_DIR", session_dir)

    log = _process_stop.session_log_path("a-session")

    assert log.parent == session_dir and session_dir.is_dir()
    log.write_text("the tool opens this for writing", encoding="utf-8")


def test_the_store_is_written_where_no_directory_existed_yet(tmp_path: Path) -> None:
    """The second door, handed the file it is to replace rather than a directory."""
    store = tmp_path / "never-created" / ".sessions" / "active_sessions.json"
    record = {"arm": {"pid": os.getpid(), "action": "teleoperate", "start_time": 0.0}}

    _process_stop.store_sessions(store, record)

    assert store.is_file()
