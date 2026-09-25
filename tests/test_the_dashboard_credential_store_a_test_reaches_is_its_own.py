"""A test reaches a credential store of its own, never the one this machine holds.

:func:`strands_robots.dashboard.auth._store_path` resolves an unset
``STRANDS_DASH_AUTH_STORE`` to ``~/.strands_dashboard/auth.json`` - the passkey
records and the ``jwt_secret``, and so the file that decides whether the
dashboard on this machine is sealed. Every reader goes through ``auth._load``,
which writes: with no store it creates one, and with a store that will not parse
it renames the operator's file aside and writes a fresh secret, invalidating
every live session token. A test does none of that visibly, so nothing in a
green report says it happened.

Sixteen modules used to redirect the store in an autouse fixture of their own,
which left the guarantee as sixteen authors remembering to. These cells declare
nothing at all - no fixture, no marker - and grade what the session hands a
module that does not ask: a store inside this test's own directory, and none of
the sibling knobs that would let the developer's shell decide a verdict a cell
is grading.

``tests/conftest.py`` holds the fixture; the cells are here because a module
that declares nothing is the thing under test.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from strands_robots.dashboard import auth

_ENV = "STRANDS_DASH_AUTH_"

#: The file a dashboard reads when nothing redirects it.
_THE_OPERATORS_STORE = Path.home() / ".strands_dashboard" / "auth.json"

#: The cell below that the poisoned-environment cell runs in a child session.
_INHERITANCE_PROBE = "test_no_knob_of_the_family_reaches_a_cell"


def test_the_store_is_not_the_file_this_machine_is_sealed_with() -> None:
    """The resolved path is neither the default nor anything under that directory."""
    resolved = auth._store_path()
    assert resolved != _THE_OPERATORS_STORE.resolve(), (
        f"a test resolved the store to {resolved}, the record this machine is sealed with"
    )
    assert ".strands_dashboard" not in resolved.parts, (
        f"a test resolved the store to {resolved}, inside the operator's dashboard directory"
    )


def test_the_store_is_inside_this_tests_own_directory(tmp_path: Path) -> None:
    """Per test, not per session: a store one cell writes is not a later cell's read."""
    assert auth._store_path() == (tmp_path / "auth.json").resolve()


def test_a_read_writes_the_store_it_was_pointed_at(tmp_path: Path) -> None:
    """``_load`` creates what it could not find - here, and nowhere else.

    The redirect is asserted before the read, so a session that lost it fails
    this cell without having written anything.
    """
    assert auth._store_path().parent == tmp_path.resolve()
    assert auth.has_credentials() is False
    assert [path.name for path in tmp_path.iterdir()] == ["auth.json"]


def test_no_knob_of_the_family_reaches_a_cell() -> None:
    """Only the ``STORE`` redirect is set, whatever the environment carried.

    Run by :func:`test_a_knob_in_the_environment_is_not_inherited` in a child
    session whose environment carries two of them; in this session it states
    what that child asserts.
    """
    assert [name for name in os.environ if name.startswith(_ENV)] == [_ENV + "STORE"]


@pytest.mark.timeout(300)
def test_a_knob_in_the_environment_is_not_inherited() -> None:
    """A knob the shell exports does not decide a verdict a cell is grading.

    ``RP_ID`` pins the relying party a passkey is bound to and ``ENABLED`` decides
    whether the dashboard is sealed at all, so either one inherited turns a cell
    that grades a derivation into a cell that reports the environment. Graded in a
    child session because this one's environment was already swept before the
    first cell ran.
    """
    finished = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{Path(__file__).name}::{_INHERITANCE_PROBE}",
            "-q",
            "--no-cov",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
        ],
        cwd=Path(__file__).resolve().parent,
        env={**os.environ, _ENV + "RP_ID": "attacker.example", _ENV + "ENABLED": "1"},
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert finished.returncode == 0, finished.stdout + finished.stderr
