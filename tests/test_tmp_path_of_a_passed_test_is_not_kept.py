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
one at its own teardown, when nothing can read it any more. It does not empty
the base temp while the session runs: pytest also writes one ``<name>current``
symlink per distinct test-name prefix and removes those only at session end,
so within a run the scan still grows with the number of distinct names rather
than the number of tests - about 25,600 against 58,000 here.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import ClassVar

import pytest

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


class TestAPassedTestsDirectoryIsGoneBeforeTheNextTestRuns:
    """Driven on the fixture itself rather than on the option: the first cell passes, the second looks."""

    #: The ``tmp_path`` the first cell was handed, read by the second. ``--dist
    #: loadfile`` keeps the two on one worker, in file order.
    _handed_out: ClassVar[list[Path]] = []

    def test_a_cell_that_passes_leaves_its_tmp_path_behind_for_teardown(self, tmp_path: Path) -> None:
        (tmp_path / "written.txt").write_text("a file the teardown has to remove with the directory", encoding="utf-8")
        assert tmp_path.is_dir()
        self._handed_out.append(tmp_path)

    def test_that_directory_no_longer_exists(self) -> None:
        if not self._handed_out:
            pytest.skip("the cell before this one did not run in this process, so there is nothing to look for")
        (earlier,) = self._handed_out
        assert not earlier.exists(), (
            f"{earlier} survived the teardown of the passed test it was created for; under "
            "tmp_path_retention_policy = 'failed' a passed test's directory is removed at its own teardown, "
            "and one that stays is one more entry every later tmp_path in this worker has to scan past"
        )
