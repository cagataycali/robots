"""A helper that mixes ``return <value>`` with a terminal ``pytest.fail(...)`` has no implicit return.

CodeQL's ``py/mixed-returns`` reports a function that returns a value on some
paths and falls off the end on another, because the fall-through returns
``None``. Test helpers here end in a pytest *outcome* by design::

    def _self_calls(module: str, method: str) -> set[str]:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == method:
                return {...}
        pytest.fail(f"{method} not found in {module}")

``pytest.fail``, ``pytest.skip``, ``pytest.exit`` and ``pytest.xfail`` are each
declared ``-> NoReturn`` (``_pytest/outcomes.py``), as the ``__call__`` of an
outcome class - which is why CodeQL's Python analysis does not model them. The
only way past that line is an exception, so the ``None`` the alert describes
cannot be produced. It has been reported three times - alerts 823
(``tests/simulation/test_recording_rate_matches_control_frequency.py``,
2026-07-29), 1140 (``tests/drivers/ur/test_ur_sim_joint_order_matches_the_wire.py``,
2026-08-31) and 1206 (``tests/mesh/test_mesh_guide_opening_block_starts_the_mesh.py``,
2026-09-12, whose review thread gated #3551 under
``required_review_thread_resolution``) - and all three are false positives.
``AGENTS.md`` records that adjudication so a fourth instance is read rather than
re-derived.

What this module grades is the precondition the adjudication rests on. ``mypy``
grades the counterfactual for an annotated helper: replace the ``pytest.fail``
with any call that can return and it reports ``Missing return statement
[return]``. But that grader has two holes in the test trees, and one of the three
adjudicated sites sits in one of them. A helper declared ``-> Any`` is not
checked for a missing return at all, which is alert 1140's shape; and
``[tool.mypy]`` relaxes ``disallow_untyped_defs`` for ``tests.*`` and
``tests_integ.*``, so an unannotated helper's body is never read. In either
shape a helper whose terminal call *can* return - ``print``, ``logger.warning``,
a cleanup - is exactly the defect the query names, with nothing else to report
it. So the exemption is not "a mixed-return helper in a test is fine" - it holds
while the terminal call is one that cannot return, and that is the property
asserted below, at every site and whatever the annotation.

The population is derived from the tree rather than written down, so a helper
added by a later change is graded on arrival and one that stops qualifying
leaves without an edit here.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# The trees a pytest outcome can legitimately terminate a helper in. The
# package is out of scope: it does not import pytest, and every function there
# is annotated with a concrete return type, so ``mypy`` grades it without a hole.
_TREES = ("tests", "tests_integ")

# Callables declared ``-> NoReturn`` by their own module. The pytest four are
# ``_pytest/outcomes.py``; ``sys.exit`` and ``os._exit`` are the interpreter's
# and are the two CodeQL itself models.
_NO_RETURN_CALLEES = frozenset(
    {
        ("pytest", "fail"),
        ("pytest", "skip"),
        ("pytest", "exit"),
        ("pytest", "xfail"),
        ("sys", "exit"),
        ("os", "_exit"),
    }
)


class _MixedReturnHelper(NamedTuple):
    """One function that returns a value somewhere and ends in a bare call."""

    path: str
    name: str
    line: int
    terminal_call: str


def _imported_callees(tree: ast.Module) -> dict[str, tuple[str, str]]:
    """Map each module-level import alias to the ``(module, name)`` it binds.

    ``import pytest`` binds ``pytest`` to the module; ``from pytest import fail``
    and ``from pytest import fail as bail`` bind a callable directly, and both
    spellings have to resolve to the same ``("pytest", "fail")`` key.
    """
    aliases: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name] = (alias.name, "")
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = (node.module, alias.name)
    return aliases


def _callee_key(call: ast.Call, aliases: dict[str, tuple[str, str]]) -> tuple[str, str] | None:
    """Resolve ``pytest.fail(...)`` and ``fail(...)`` to ``("pytest", "fail")``; ``None`` for anything else."""
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        module, name = aliases.get(func.value.id, (func.value.id, ""))
        return (module, func.attr) if not name else None
    if isinstance(func, ast.Name):
        module, name = aliases.get(func.id, ("", ""))
        return (module, name) if name else None
    return None


def _returns_a_value(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether ``function``'s own body - not a nested def or lambda - has a ``return <expr>``."""
    stack: list[ast.AST] = list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Return):
            if node.value is not None:
                return True
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return False


def _call_tails(body: list[ast.stmt]) -> list[ast.Call]:
    """The bare calls control can fall off the end of ``body`` through.

    Walks the last statement only, because that is the one a fall-through
    leaves by: a call there is a tail; a ``return`` or ``raise`` is not; an
    ``if``, ``try`` or ``with`` is entered and its own last statements read the
    same way. Alert 1140's shape is the ``try`` case - ``return`` inside the
    body, ``pytest.skip(...)`` closing the handler. A branch that ends in
    anything else (an assignment, a loop, an empty ``else``) is a *silent*
    fall-through, which is a mixed return too but not one this exemption
    covers, so it is left to the query rather than counted here.
    """
    if not body:
        return []
    last = body[-1]
    if isinstance(last, ast.Expr) and isinstance(last.value, ast.Call):
        return [last.value]
    if isinstance(last, ast.If):
        return _call_tails(last.body) + _call_tails(last.orelse)
    if isinstance(last, ast.Try):
        tails = _call_tails(last.orelse or last.body)
        for handler in last.handlers:
            tails += _call_tails(handler.body)
        return tails
    if isinstance(last, (ast.With, ast.AsyncWith)):
        return _call_tails(last.body)
    return []


def _mixed_return_helpers(path: Path) -> list[tuple[_MixedReturnHelper, tuple[str, str] | None]]:
    """Every function in ``path`` that returns a value and can fall off its end through a bare call.

    That is the one shape of ``py/mixed-returns`` the adjudication exempts - a
    fall-through whose last act is a call - so it is the one shape graded here,
    one entry per such call.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases = _imported_callees(tree)
    found: list[tuple[_MixedReturnHelper, tuple[str, str] | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        tails = _call_tails(node.body)
        if not tails or not _returns_a_value(node):
            continue
        for call in tails:
            helper = _MixedReturnHelper(
                path=str(path.relative_to(REPO_ROOT)),
                name=node.name,
                line=call.lineno,
                terminal_call=ast.unparse(call.func),
            )
            found.append((helper, _callee_key(call, aliases)))
    return found


def _population() -> list[tuple[_MixedReturnHelper, tuple[str, str] | None]]:
    """Every mixed-return helper ending in a call, across the trees pytest outcomes live in."""
    return [
        entry
        for tree in _TREES
        for path in sorted((REPO_ROOT / tree).rglob("*.py"))
        for entry in _mixed_return_helpers(path)
    ]


class TestAMixedReturnHelperEndsInACallThatCannotReturn:
    """The exemption's premise, asserted at every site regardless of annotation."""

    def test_every_terminal_call_is_a_declared_no_return(self) -> None:
        """A terminal call that can return is the implicit ``None`` the query names, and nothing else reports it."""
        offenders = [
            f"{helper.path}:{helper.line} {helper.name}() ends in {helper.terminal_call}(...)"
            for helper, key in _population()
            if key not in _NO_RETURN_CALLEES
        ]
        assert offenders == [], (
            "a helper that returns a value elsewhere ends in a call that can return, "
            "so the fall-through returns None; either return after it, raise, or end in a "
            "pytest outcome (fail / skip / exit / xfail):\n  " + "\n  ".join(offenders)
        )


class TestTheClassifierReadsTheShapeItGrades:
    """Controls on the AST read, so the sweep above cannot pass by finding nothing."""

    @staticmethod
    def _classify(source: str) -> list[tuple[str, tuple[str, str] | None]]:
        tree = ast.parse(source)
        aliases = _imported_callees(tree)
        out: list[tuple[str, tuple[str, str] | None]] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and _returns_a_value(node):
                out.extend((node.name, _callee_key(call, aliases)) for call in _call_tails(node.body))
        return out

    def test_a_terminal_call_that_can_return_is_an_offender(self) -> None:
        """``print`` is the query's finding, and the annotation does not excuse it - ``-> Any`` and none are the mypy holes."""
        source = (
            "from typing import Any\n"
            "def typed(xs: list[str]) -> str:\n    for x in xs:\n        if x:\n            return x\n    print('none')\n"
            "def any_typed(xs: list[str]) -> Any:\n    for x in xs:\n        if x:\n            return x\n    print('none')\n"
            "def untyped(xs):\n    for x in xs:\n        if x:\n            return x\n    print('none')\n"
        )
        classified = self._classify(source)
        assert [name for name, _ in classified] == ["typed", "any_typed", "untyped"]
        assert all(key not in _NO_RETURN_CALLEES for _, key in classified)

    def test_every_spelling_of_a_pytest_outcome_is_accepted(self) -> None:
        """Attribute form, bare import and aliased import all resolve to the same declared ``NoReturn``."""
        source = (
            "import pytest\nimport sys\nfrom pytest import skip\nfrom pytest import fail as bail\n"
            "def a(xs: list[str]) -> str:\n    for x in xs:\n        return x\n    pytest.fail('none')\n"
            "def b(xs: list[str]) -> str:\n    for x in xs:\n        return x\n    skip('none')\n"
            "def c(xs: list[str]) -> str:\n    for x in xs:\n        return x\n    bail('none')\n"
            "def d(xs: list[str]) -> str:\n    for x in xs:\n        return x\n    sys.exit(1)\n"
        )
        keys = [key for _, key in self._classify(source)]
        assert keys == [("pytest", "fail"), ("pytest", "skip"), ("pytest", "fail"), ("sys", "exit")]
        assert all(key in _NO_RETURN_CALLEES for key in keys)

    def test_a_handler_closing_in_an_outcome_is_the_try_shape(self) -> None:
        """``return`` in the ``try`` body and ``pytest.skip`` closing the handler is alert 1140's shape; a ``print`` there is the defect."""
        source = (
            "import pytest\n"
            "def skips(robot: str):\n    try:\n        return build(robot)\n"
            "    except OSError as exc:\n        pytest.skip(str(exc))\n"
            "def leaks(robot: str):\n    try:\n        return build(robot)\n"
            "    except OSError as exc:\n        print(exc)\n"
        )
        assert self._classify(source) == [("skips", ("pytest", "skip")), ("leaks", None)]

    def test_a_bare_return_or_a_nested_def_is_not_a_value_return(self) -> None:
        """``return`` with no value is not mixed with a fall-through, and a nested function's return is its own."""
        source = (
            "def bare(xs: list[str]) -> None:\n    for x in xs:\n        return\n    print('none')\n"
            "def nested(xs: list[str]) -> None:\n    def inner() -> int:\n        return 1\n    print(inner())\n"
        )
        assert self._classify(source) == []


class TestThePopulationIsDerivedFromTheTree:
    """The sweep reads the trees it names, and finds the idiom it grades."""

    def test_every_named_tree_ships_python(self) -> None:
        """A tree that stops shipping Python is a stale name here, not a silently empty walk."""
        for tree in _TREES:
            assert any((REPO_ROOT / tree).rglob("*.py")), f"{tree}/ ships no Python; drop it from _TREES"

    def test_the_sweep_finds_the_idiom_it_grades(self) -> None:
        """At least one helper in the tree ends in a pytest outcome - three did when this was written."""
        accepted = [helper for helper, key in _population() if key in _NO_RETURN_CALLEES]
        assert accepted, "no mixed-return helper ends in a pytest outcome; the sweep is reading the wrong shape"

    def test_mypy_grades_the_annotated_half_inside_the_required_check(self) -> None:
        """The counterfactual the adjudication cites is run by ``hatch run lint`` over both test trees."""
        with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)
        lint = config["tool"]["hatch"]["envs"]["default"]["scripts"]["lint"]
        mypy_steps = [step for step in lint if step.split()[0] == "mypy"]
        assert mypy_steps, "lint no longer runs mypy, so nothing grades the counterfactual"
        for tree in _TREES:
            assert any(tree in step.split() for step in mypy_steps), f"mypy is not run over {tree}/"
        assert config["tool"]["mypy"].get("warn_no_return", True) is True
