"""A passed test's ``tmp_path`` is removed at its teardown, so the base temp does not grow with the suite.

Every test in this suite creates a ``tmp_path``: the autouse fixtures in
``tests/conftest.py`` that redirect the dashboard credential store and the
session store request it, so the fixture runs for the ~58,000 cells whether or
not a cell reads it. pytest creates each one through ``make_numbered_dir``,
which lists the *whole* base temp to pick the next suffix, and under pytest's
default ``tmp_path_retention_policy = "all"`` every directory it made stays
until the session ends. So the N-th test a worker runs pays a scan of the N-1
directories before it, and a worker running 29,000 tests pays a quadratic
total that no single cell shows: it is ~10-20 ms of setup on each of them.

Measured on ubuntu-latest (2 vCPU, ``-n 2 --dist loadfile``) with a synthetic
suite of trivial tests under one autouse ``tmp_path`` fixture, so the tests
themselves cost nothing and the scan is the whole difference:

    tests     retention=all    retention=failed    no tmp_path at all
    10,000          34.3 s              21.1 s                12.9 s
    20,000          96.7 s              42.7 s                    -

The gap between the first two columns is the retained directories - 13 s at
10,000 and 54 s at 20,000, four times the cost for twice the tests. Read
directly, ``make_numbered_dir`` costs 0.11 ms against an empty base temp and
16.7 ms against 24,000 entries.

The same directory is also the cwd a nested pytest is spawned with
(``_run_pytest`` in tests/test_session_truncation_is_reported.py), and a pytest
whose arguments name no ini file takes the common ancestor of its cwd and its
arguments as rootdir - the base temp - and collects it, which is a scan of
every entry in it. Measured against 24,000 sibling entries: 4.5 s per nested
run, against 0.27 s with ``--rootdir`` naming the target - the shape of the
5-8 s a nested-pytest cell costs on CI and not locally, where the base temp
holds a few hundred entries rather than tens of thousands (#3869).

``failed`` keeps a failed test's directory for inspection and removes a passed
one at its own teardown, when nothing can read it any more. On its own it does
not empty the base temp while the session runs: pytest also writes one
``<name>current`` symlink per distinct test-name prefix and removes those only
at session end, so within a run the scan still grew with the number of distinct
names rather than the number of tests - about 25,600 against 58,000 here,
~13,000 per worker. Measured with 20,000 trivial tests of *distinct* names
under the ``failed`` policy: 101-108 s, against the 42.7 s the 250-name suite
above took, and a mid-run sample of each worker's base temp held 7,000+
symlinks and no directory. ``pytest_runtest_teardown`` in ``tests/conftest.py``
removes the symlink in the same motion as the directory - only once the
directory is gone, so a failed test keeps both - which took the distinct-name
suite to 44-45 s. The two halves are pinned below, on the fixture itself.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import ClassVar

import pytest

from tests.conftest import remove_dead_current_symlink

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_the_policy_is_declared_where_every_run_reads_it() -> None:
    """``pyproject.toml`` states the policy, so a bare ``pytest`` and ``hatch run test`` agree."""
    ini = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]
    assert ini.get("tmp_path_retention_policy") == "failed", (
        f"[tool.pytest.ini_options].tmp_path_retention_policy is {ini.get('tmp_path_retention_policy')!r}; "
        "under pytest's default 'all' every one of the suite's ~58,000 tmp_path directories is kept until "
        "the session ends, and make_numbered_dir lists the whole base temp to create the next one, so a "
        "worker's setup cost grows quadratically with the tests it has run (measured: +13 s at 10,000 "
        "trivial tests, +54 s at 20,000, on the two-vCPU runner the required check uses)"
    )


def test_the_session_runs_under_that_policy(pytestconfig: pytest.Config) -> None:
    """The declared value is the one this session resolved, so nothing on the command line undid it."""
    assert pytestconfig.getini("tmp_path_retention_policy") == "failed"


def _current_symlink_of(tmp_path: Path, test_name: str) -> Path:
    """The ``<prefix>current`` symlink ``make_numbered_dir`` writes beside a test's ``tmp_path``.

    Spelled as ``_pytest.tmpdir._mk_tmp`` spells the prefix - non-word characters
    to ``_``, cut to 30 - so the cell that reads it back checks the spelling
    against what pytest actually wrote, and the cleanup that shares it is graded
    on the same name.
    """
    return tmp_path.parent / (re.sub(r"[\W]", "_", test_name)[:30] + "current")


class TestAPassedTestsDirectoryIsGoneBeforeTheNextTestRuns:
    """Driven on the fixture itself rather than on the option: the first cell passes, the later ones look."""

    #: The ``tmp_path`` the first cell was handed and the ``current`` symlink
    #: pytest wrote for it, read by the cells after. ``--dist loadfile`` keeps
    #: them on one worker, in file order.
    _handed_out: ClassVar[list[tuple[Path, Path]]] = []

    def test_a_cell_that_passes_leaves_its_tmp_path_behind_for_teardown(
        self, tmp_path: Path, request: pytest.FixtureRequest
    ) -> None:
        (tmp_path / "written.txt").write_text("a file the teardown has to remove with the directory", encoding="utf-8")
        assert tmp_path.is_dir()
        link = _current_symlink_of(tmp_path, request.node.name)
        # The control for the name: pytest wrote this symlink, for this
        # directory. A prefix spelled differently from pytest's would leave a
        # symlink the teardown never looks at, and the third cell would then
        # be asserting the absence of a name nothing ever created.
        assert link.is_symlink() and link.resolve() == tmp_path.resolve(), (
            f"{link} is not the current symlink pytest wrote for {tmp_path}; the prefix spelling here "
            "has drifted from _pytest.tmpdir._mk_tmp, so the cleanup keyed on it removes nothing"
        )
        self._handed_out.append((tmp_path, link))

    def test_that_directory_no_longer_exists(self) -> None:
        if not self._handed_out:
            pytest.skip("the cell before this one did not run in this process, so there is nothing to look for")
        ((earlier, _link),) = self._handed_out
        assert not earlier.exists(), (
            f"{earlier} survived the teardown of the passed test it was created for; under "
            "tmp_path_retention_policy = 'failed' a passed test's directory is removed at its own teardown, "
            "and one that stays is one more entry every later tmp_path in this worker has to scan past"
        )

    def test_nor_does_its_current_symlink(self) -> None:
        if not self._handed_out:
            pytest.skip("the first cell did not run in this process, so there is nothing to look for")
        ((earlier, link),) = self._handed_out
        assert not link.is_symlink(), (
            f"{link} survived the teardown that removed {earlier}; pytest removes a dead current symlink only "
            "at session end, so one stays per distinct test name for the whole run and every later tmp_path "
            "in this worker scans past it - pytest_runtest_teardown in tests/conftest.py removes it with the "
            "directory"
        )


class TestTheCurrentSymlinkIsRemovedOnlyOnceItsDirectoryIsGone:
    """Both directions of the cleanup, on a staged base temp rather than the session's own."""

    @staticmethod
    def _stage(base: Path, test_name: str) -> tuple[Path, Path]:
        """A numbered directory and its ``current`` symlink, as ``make_numbered_dir`` leaves them."""
        base.mkdir()
        directory = base / f"{test_name}0"
        directory.mkdir()
        link = _current_symlink_of(directory, test_name)
        link.symlink_to(directory, target_is_directory=True)
        return directory, link

    def test_a_directory_that_is_gone_takes_its_symlink_with_it(self, tmp_path: Path) -> None:
        directory, link = self._stage(tmp_path / "base", "test_passed")
        directory.rmdir()
        remove_dead_current_symlink(directory, "test_passed")
        assert not link.is_symlink()
        assert list((tmp_path / "base").iterdir()) == []

    def test_a_directory_that_is_kept_keeps_its_symlink(self, tmp_path: Path) -> None:
        """A failed test's directory stays under the policy, and so does the link that names it."""
        directory, link = self._stage(tmp_path / "base", "test_failed")
        remove_dead_current_symlink(directory, "test_failed")
        assert directory.is_dir()
        assert link.is_symlink() and link.resolve() == directory.resolve()

    def test_a_parametrized_name_is_spelled_as_pytest_spells_it(self, tmp_path: Path) -> None:
        """``test_x[a-b]`` becomes ``test_x_a_b_``, cut to 30: the same prefix pytest named the directory by."""
        name = "test_a_long_parametrized_name[value-1]"
        directory, link = self._stage(tmp_path / "base", re.sub(r"[\W]", "_", name)[:30])
        directory.rmdir()
        remove_dead_current_symlink(directory, name)
        assert not link.is_symlink()

    def test_no_symlink_is_not_an_error(self, tmp_path: Path) -> None:
        """A platform that could not write the symlink, or a second call, has nothing to remove."""
        base = tmp_path / "base"
        base.mkdir()
        remove_dead_current_symlink(base / "test_bare0", "test_bare")
        assert list(base.iterdir()) == []
