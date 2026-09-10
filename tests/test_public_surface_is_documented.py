# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""No public surface in the package ships without a docstring, anywhere in it.

A public module, class, method or function with no docstring at all is the one
documentation defect a caller cannot work around: there is nothing to read but
the source, and an agent driving this SDK blind has no source to read.

The rule used to be asserted by fourteen copies of one AST walk under ``tests``,
one per package, each rooted at its own ``<pkg>.__file__``. Fourteen copies
covered fourteen of the package's forty-one directories, and all twenty-two
undocumented surfaces in the tree sat in a directory none of them walked: :mod:`strands_robots.dashboard` (eighteen, including the
WebAuthn ceremony verbs), :mod:`strands_robots.rtps.idl` (two) and
``strands_robots.tools.g1`` (three - ``list_services``, ``list_operations`` and
``describe_operation``, the discovery verbs an agent calls precisely because it
does not know the surface). A per-package guard reports a clean sweep over the
packages it happens to name, so the copies could not have said otherwise.

So the rule has one owner now, and it is the linter: ``ruff`` selects the
pydocstyle PRESENCE codes (``D100``-``D103``, ``D106``) over the whole package,
inside the merge-blocking lint gate. A linter cannot be vacuous the way a walk
rooted at an importable module can - it reports on the paths it is handed, and
``hatch run lint`` hands it ``strands_robots`` by name.

What is pinned here is that arrangement, not a second implementation of the
rule: the codes are selected, no ignore takes them back for the package, the
lint gate names the package, and the codes really do report the shape. The
FORMAT codes (``D205`` summary layout, ``D401`` imperative mood, ``D413``
section spacing) are deliberately not selected - 2,051 findings on this tree,
none of them a missing docstring. Whether a docstring that exists is *complete*
is graded elsewhere, against the blocks it already has:
:mod:`tests.test_args_docstring_completeness` and
:mod:`tests.test_raises_docstring_completeness`.
"""

from __future__ import annotations

import inspect
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import strands_robots as package

#: The presence codes, and only those.
PRESENCE_CODES = ("D100", "D101", "D102", "D103", "D106")

#: Derived from the imported package rather than a path literal, so a moved
#: package cannot leave this pointing at an empty tree.
PACKAGE_ROOT = Path(inspect.getfile(package)).parent

REPO_ROOT = PACKAGE_ROOT.parent


def _pyproject() -> dict:
    """The repository's parsed ``pyproject.toml``."""
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _ruff(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ``ruff`` with *args*, skipping the test when it is not installed."""
    if shutil.which("ruff") is None and not (Path(sys.executable).parent / "ruff").exists():
        pytest.skip("ruff is not installed in this environment")
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


class TestTheLinterOwnsTheRule:
    """The presence codes are selected for the package, in the gate that blocks a merge."""

    def test_every_presence_code_is_selected(self) -> None:
        selected = _pyproject()["tool"]["ruff"]["lint"]["select"]
        missing = [code for code in PRESENCE_CODES if code not in selected]
        assert missing == [], f"presence codes not selected: {missing}"

    def test_no_ignore_takes_them_back_for_the_package(self) -> None:
        """A per-file ignore for the package would make the selection decorative."""
        lint = _pyproject()["tool"]["ruff"]["lint"]
        assert not [code for code in PRESENCE_CODES if code in lint.get("ignore", [])]
        for pattern, codes in lint.get("per-file-ignores", {}).items():
            if pattern.startswith(("tests/", "tests_integ/", "examples/", "scripts/", ".github/")):
                continue
            assert not [code for code in PRESENCE_CODES if code in codes], (
                f"per-file-ignores pattern {pattern!r} takes the presence codes back"
            )

    def test_the_lint_gate_hands_ruff_the_package(self) -> None:
        """A selection ruff is never pointed at the package with would grade nothing."""
        lint = _pyproject()["tool"]["hatch"]["envs"]["default"]["scripts"]["lint"]
        checks = [step for step in lint if step.startswith("ruff check")]
        assert checks, lint
        assert all(package.__name__ in step for step in checks), checks


class TestThePackageIsClean:
    """The rule holds over the whole package, which is the behaviour itself."""

    def test_no_public_surface_is_undocumented(self) -> None:
        result = _ruff(
            "check", "--no-cache", "--output-format=concise", f"--select={','.join(PRESENCE_CODES)}", str(PACKAGE_ROOT)
        )
        assert result.returncode == 0, result.stdout or result.stderr

    def test_the_tree_it_grades_is_populated(self) -> None:
        """Non-vacuity of the target: a clean sweep over nothing reads identically."""
        modules = [p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts]
        assert len(modules) > 200, len(modules)


class TestTheCodesReportTheShape:
    """Non-vacuity of the invocation: each code fires on a planted omission."""

    def test_a_planted_omission_is_reported_by_every_code(self, tmp_path: Path) -> None:
        planted = tmp_path / "undocumented.py"
        planted.write_text(
            "def loose_function():\n"
            "    return 1\n"
            "\n"
            "\n"
            "class Undocumented:\n"
            "    class Nested:\n"
            "        pass\n"
            "\n"
            "    def method(self):\n"
            "        return 2\n",
            encoding="utf-8",
        )
        result = _ruff(
            "check",
            "--no-cache",
            "--isolated",
            "--output-format=concise",
            f"--select={','.join(PRESENCE_CODES)}",
            str(planted),
        )
        reported = {code for code in PRESENCE_CODES if code in result.stdout}
        assert reported == set(PRESENCE_CODES), result.stdout
