# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A gate on an optional dependency skips the cells it names, not the whole module.

``pytest.importorskip`` *raises* ``Skipped`` when the module is absent -- it
never returns ``None``. A decorator expression is evaluated while the module is
being imported, so a gate written there raises during import, the raise leaves
the collector, and pytest skips the entire file instead of the one test or class
the decorator sits on. The comparison such a gate writes,
``importorskip(...) is None``, is unreachable on both arms: absent it raised,
present it is a module.

The failure is silent in the reassuring direction -- a muted file reports
``1 skipped``, which reads like one test. Two gates in this tree were written to
protect a single cell each and were deleting every sibling cell in their file
whenever the extra was missing. Measured on an interpreter without the extra,
running each file on its own:

======================================================== =============== ======================
module                                                   gate as found   gate reading the spec
======================================================== =============== ======================
``tests/policies/...misspelled_provider_kwarg...``       none collected  17 passed, 1 skipped
``tests/simulation/test_sim_engine_describe_discovery``  none collected  33 passed, 17 skipped
======================================================== =============== ======================

Fifty cells needing neither ``zmq`` nor ``mujoco`` were being dropped by gates
protecting the eighteen that do. A merge check installs every extra, so it
collected all of them and read green; only an environment missing the extra --
the environment the gate exists for -- loses them, which is why this is graded
here rather than left to a required check.

The convention the rest of the suite keeps reads presence as a *spec* and
imports nothing::

    _HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None

    @pytest.mark.skipif(not _HAS_MUJOCO, reason="MuJoCo not installed")

A bare module-scope ``pytest.importorskip("x")`` is not in scope and is not
flagged: it mutes its module deliberately, which is what a file wholly about
``x`` wants. Nor is a call inside a test body, which skips that test at run
time. The shape refused here is the single one whose stated scope and actual
scope disagree.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: The two test trees a merge gates on.
_TEST_ROOTS = ("tests", "tests_integ")


def _test_modules() -> list[Path]:
    """Every Python module under the test trees, ``__pycache__`` excluded."""
    return sorted(
        path for root in _TEST_ROOTS for path in (_REPO_ROOT / root).rglob("*.py") if "__pycache__" not in path.parts
    )


def _calls_importorskip(func: ast.expr) -> bool:
    """Whether ``func`` names ``importorskip``, qualified or bare."""
    if isinstance(func, ast.Attribute):
        return func.attr == "importorskip"
    return isinstance(func, ast.Name) and func.id == "importorskip"


def _gates_in_a_decorator(source: str) -> list[int]:
    """Lines where ``source`` evaluates ``importorskip`` in a decorator.

    Args:
        source: Python source text of one test module.

    Returns:
        The line number of each such call, ascending. Empty for source that
        cannot be parsed, which the syntax the interpreter enforces already
        grades.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return sorted(
        call.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        for decorator in node.decorator_list
        for call in ast.walk(decorator)
        if isinstance(call, ast.Call) and _calls_importorskip(call.func)
    )


def test_both_test_trees_are_scanned() -> None:
    """Guard: the sweep walked both test trees, not one subtree."""
    modules = _test_modules()
    assert len(modules) > 1000, f"only {len(modules)} test modules found; the sweep is too narrow"
    assert set(_TEST_ROOTS) <= {path.relative_to(_REPO_ROOT).parts[0] for path in modules}


def test_no_optional_dependency_gate_is_written_in_a_decorator() -> None:
    """A gate in a decorator mutes its whole module; none may be written that way."""
    offenders = [
        f"{path.relative_to(_REPO_ROOT)}:{line}"
        for path in _test_modules()
        for line in _gates_in_a_decorator(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "pytest.importorskip raises while the module is imported, so a gate "
        "written in a decorator skips the entire file rather than the test it "
        "names. Read presence as a spec instead - "
        '_HAS_X = importlib.util.find_spec("x") is not None, then '
        "@pytest.mark.skipif(not _HAS_X, ...):\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    ("label", "source"),
    [
        (
            "on a function",
            'import pytest\n\n\n@pytest.mark.skipif(pytest.importorskip("x") is None, reason="x")\ndef test_a():\n    pass\n',
        ),
        (
            "on a class",
            'import pytest\n\n\n@pytest.mark.skipif(not pytest.importorskip("x"), reason="x")\nclass TestA:\n    pass\n',
        ),
        (
            "imported bare",
            'from pytest import importorskip, mark\n\n\n@mark.skipif(importorskip("x") is None, reason="x")\ndef test_a():\n    pass\n',
        ),
    ],
)
def test_a_gate_written_in_a_decorator_is_reported(label: str, source: str) -> None:
    """The predicate catches every spelling of the shape this guard refuses."""
    assert _gates_in_a_decorator(source), f"the {label} spelling is no longer caught"


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("module scope, muting on purpose", 'import pytest\n\npytest.importorskip("x")\n'),
        ("inside a test body", 'import pytest\n\n\ndef test_a():\n    pytest.importorskip("x")\n'),
        (
            "presence read as a spec",
            'import importlib.util\n\nimport pytest\n\n_HAS_X = importlib.util.find_spec("x") is not None\n\n\n@pytest.mark.skipif(not _HAS_X, reason="x")\ndef test_a():\n    pass\n',
        ),
    ],
)
def test_a_gate_with_the_scope_it_states_is_not_reported(label: str, source: str) -> None:
    """The rule is about scope disagreement, not about ``importorskip`` itself."""
    assert _gates_in_a_decorator(source) == [], f"the {label} case is wrongly flagged"


def test_importorskip_raises_rather_than_returning_none() -> None:
    """The mechanism: the comparison a decorator gate writes is never reached."""
    with pytest.raises(pytest.skip.Exception):
        pytest.importorskip("a_module_no_environment_installs")


#: A module with two cells needing nothing and one needing an absent dependency.
_TWO_FREE_CELLS_AND_A_GATED_ONE = '''\
"""Two cells that need no dependency, and one gated on a module nothing installs."""

{prelude}

def test_first_free_cell():
    assert True


def test_second_free_cell():
    assert True


{gate}
def test_the_gated_cell():
    raise AssertionError("must not run without the dependency")
'''

_ABSENT = "a_module_no_environment_installs"


@pytest.mark.parametrize(
    ("label", "prelude", "gate", "expected"),
    [
        (
            "gate in the decorator",
            "import pytest",
            f'@pytest.mark.skipif(pytest.importorskip("{_ABSENT}") is None, reason="absent")',
            (0, 1),
        ),
        (
            "presence read as a spec",
            f'import importlib.util\n\nimport pytest\n\n_HAS_IT = importlib.util.find_spec("{_ABSENT}") is not None',
            '@pytest.mark.skipif(not _HAS_IT, reason="absent")',
            (2, 1),
        ),
    ],
)
def test_only_the_gated_cell_is_skipped_when_the_gate_states_its_scope(
    tmp_path: Path, label: str, prelude: str, gate: str, expected: tuple[int, int]
) -> None:
    """Behavioural pin for the rationale: the decorator gate deletes its siblings.

    Runs one synthetic module in its own interpreter, outside this repository so
    no shared configuration applies, and reads the counts pytest reports. The
    decorator spelling collects nothing at all; the spec spelling runs both
    independent cells and skips only the gated one.
    """
    module = tmp_path / "test_synthetic_gate.py"
    module.write_text(_TWO_FREE_CELLS_AND_A_GATED_ONE.format(prelude=prelude, gate=gate), encoding="utf-8")
    finished = subprocess.run(
        [sys.executable, "-m", "pytest", module.name, "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    report = finished.stdout + finished.stderr
    counted = {
        outcome: int(found.group(1))
        for outcome in ("passed", "skipped")
        if (found := re.search(rf"(\d+) {outcome}", report))
    }
    assert (counted.get("passed", 0), counted.get("skipped", 0)) == expected, f"{label} reported:\n{report}"
