"""A ``cast("X", ...)`` string is a use of ``X``, and the exemption that says so has a precondition.

``typing.cast``'s first argument is an ordinary runtime expression, and the idiom
here writes it as a *string* so the name it refers to need exist only for a type
checker::

    if TYPE_CHECKING:
        from strands_robots.training.rl import SimEnv
    ...
    env_factory=lambda: cast("SimEnv", _FakeTermEnv(terminated_flag))

CodeQL's ``py/unused-import`` reads bare ``Name`` loads, and a string carries
none, so it reports the import as dead. It has done so three times - alerts 599
(``tests/training/test_rl_truncation_bootstrap.py``, 2026-07-02), 1138
(``tests/drivers/robotiq/test_robotiq_gripper_moves_over_modbus_tcp.py``,
2026-08-31) and 1160 (``strands_robots/training/rl/fast_td3.py``, 2026-09-05) -
and all three were dismissed as false positives, the last one after its review
thread had held a merge for twelve hours under
``required_review_thread_resolution``. ``AGENTS.md`` records that adjudication so
a fourth instance is read rather than re-derived.

What this module grades is the precondition the adjudication rests on, which is
the half no other gate can see. Taking the alert's advice is already refused:
delete one of these imports and ``ruff`` reports ``F821 Undefined name`` at the
cast while ``mypy`` reports ``name-defined``, both inside
``call-test-lint / Test and Lint``. The direction with no gate is the opposite
one. A ``TYPE_CHECKING`` import of a name the module *also* binds at runtime is
genuinely dead: the cast string still resolves, so ``ruff`` and ``mypy`` stay
silent, and there ``py/unused-import`` is right and the import should go. So the
exemption is not "a cast string always excuses the import" - it holds while the
``TYPE_CHECKING`` import is the name's only binding, and that is the property
asserted below.

Stated the other way round, this is the discriminator a reader needs at the
alert: run the counterfactual rather than the query. ``F821`` means load-bearing
and the alert is a false positive; a clean ``ruff check`` means the name is bound
at runtime too and the alert is correct. The rule is not reliably wrong here -
seven of its twelve alerts on ``main`` are open and unadjudicated - so an
exemption that did not name its own boundary would suppress live signal.

The population is derived from the tree rather than written down, so a site added
by a later change is graded on arrival and one that stops qualifying leaves
without an edit here.
"""

from __future__ import annotations

import ast
import builtins
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every top-level directory that ships Python. The idiom this grades is not
# confined to one area: two of the three alerts above are under `tests/` and one
# is in the package, and an `examples/` copy would be the shape a reader
# reproduces. A directory added later that ships Python is swept by
# `_swept_areas()` without an edit here.
_AREAS = ("strands_robots", "tests", "tests_integ", "examples", "scripts")

# A name that resolves to a builtin needs no import, so it is not evidence about
# one. `cast("dict[str, Any]", ...)` contributes `dict` here and `Any`, which is
# imported, is what the assertion is about.
_BUILTIN_NAMES = frozenset(dir(builtins))


class _CastStringName(NamedTuple):
    """One name a ``cast`` string refers to, in the module whose import supplies it."""

    path: str
    name: str
    cast_line: int
    import_line: int


def _module_level_bindings(tree: ast.Module) -> dict[str, set[str]]:
    """Map each module-level name to the kinds of binding that supply it.

    ``"typing"`` for a binding inside an ``if TYPE_CHECKING:`` block, which does
    not exist at runtime, and ``"runtime"`` for every other module-level
    binding - an import, a ``def``, a ``class``, or an assignment. A name can
    carry both, and that is exactly the case this module refuses.
    """
    bindings: dict[str, set[str]] = {}

    def bind(name: str, kind: str) -> None:
        bindings.setdefault(name, set()).add(kind)

    def visit(body: list[ast.stmt], *, typing_only: bool) -> None:
        kind = "typing" if typing_only else "runtime"
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bind((alias.asname or alias.name).split(".")[0], kind)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bind(node.name, kind)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bind(target.id, kind)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                bind(node.target.id, kind)
            elif isinstance(node, ast.If):
                # `else:` runs when TYPE_CHECKING is false, so it is a runtime
                # branch however the test reads.
                visit(node.body, typing_only=typing_only or _is_type_checking_test(node.test))
                visit(node.orelse, typing_only=typing_only)
            elif isinstance(node, (ast.Try, ast.With)):
                visit(node.body, typing_only=typing_only)
                for handler in getattr(node, "handlers", []):
                    visit(handler.body, typing_only=typing_only)
                visit(getattr(node, "orelse", []), typing_only=typing_only)
                visit(getattr(node, "finalbody", []), typing_only=typing_only)

    visit(tree.body, typing_only=False)
    return bindings


def _is_type_checking_test(test: ast.expr) -> bool:
    """Both spellings: a bare ``TYPE_CHECKING`` and a qualified ``typing.TYPE_CHECKING``."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _type_checking_import_lines(tree: ast.Module) -> dict[str, int]:
    """Map a name imported under ``TYPE_CHECKING`` to the line that imports it."""
    lines: dict[str, int] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If) and _is_type_checking_test(node.test)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Import, ast.ImportFrom)):
                for alias in sub.names:
                    lines.setdefault((alias.asname or alias.name).split(".")[0], sub.lineno)
    return lines


def _names_in_cast_strings(tree: ast.Module) -> set[tuple[str, int]]:
    """Every ``(name, line)`` a string-literal first argument to ``cast`` refers to.

    The string is parsed rather than split, so a subscripted reference
    (``cast("Mapping[str, SimEnv]", ...)``) contributes both names and a
    dotted one contributes its root, which is the name an import binds.
    """
    found: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if called != "cast" or not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        try:
            reference = ast.parse(first.value, mode="eval")
        except SyntaxError:
            # Not a type expression at all. mypy reports that itself, and it is
            # not evidence about any import.
            continue
        for sub in ast.walk(reference):
            if isinstance(sub, ast.Name):
                found.add((sub.id, node.lineno))
    return found


@lru_cache(maxsize=1)
def _cast_string_names() -> tuple[_CastStringName, ...]:
    """Every name a ``cast`` string refers to that a ``TYPE_CHECKING`` import supplies."""
    collected: list[_CastStringName] = []
    # The walk is spelled with the area held in the loop variable, which is the
    # form `scripts/check_whole_tree_graders.py` resolves: this grader's input is
    # the rest of the repository, so a diff-scoped selector collects it only
    # through that roster.
    for area in _AREAS:
        for path in sorted((REPO_ROOT / area).rglob("*.py")):
            source = path.read_text(encoding="utf-8", errors="ignore")
            if "cast(" not in source:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            import_lines = _type_checking_import_lines(tree)
            if not import_lines:
                continue
            for name, cast_line in sorted(_names_in_cast_strings(tree)):
                if name in _BUILTIN_NAMES or name not in import_lines:
                    continue
                collected.append(
                    _CastStringName(
                        path=str(path.relative_to(REPO_ROOT)),
                        name=name,
                        cast_line=cast_line,
                        import_line=import_lines[name],
                    )
                )
    return tuple(collected)


class TestACastStringImportIsTheNamesOnlyBinding:
    """The exemption for ``py/unused-import`` holds only while nothing binds the name at runtime."""

    def test_no_module_binds_a_cast_string_name_both_ways(self) -> None:
        """A name bound at runtime too makes the ``TYPE_CHECKING`` import genuinely dead.

        This is the direction ``ruff`` and ``mypy`` cannot report: the cast
        string resolves either way, so both stay silent while the import has
        stopped carrying anything. ``py/unused-import`` is correct on that shape,
        and dismissing it as the same false positive as alerts 599 / 1138 / 1160
        would suppress a true finding. The remedy is to delete the
        ``TYPE_CHECKING`` import, which is what the alert asks for.
        """
        redundant = []
        for reference in _cast_string_names():
            source = (REPO_ROOT / reference.path).read_text(encoding="utf-8")
            bindings = _module_level_bindings(ast.parse(source))
            if "runtime" in bindings.get(reference.name, set()):
                redundant.append(
                    f"{reference.path}:{reference.import_line} imports {reference.name!r} under "
                    f"TYPE_CHECKING, and the module also binds it at runtime; the cast at line "
                    f"{reference.cast_line} resolves without the import"
                )
        assert not redundant, (
            "A TYPE_CHECKING import is only load-bearing behind a cast string while it is the "
            "name's sole binding. These carry a runtime binding as well, so the import is dead "
            "and py/unused-import is right about them - delete the import rather than dismissing "
            "the alert: " + "; ".join(redundant)
        )


class TestThePopulationIsDerivedFromTheTree:
    """A sweep that reaches nothing passes, so what it reaches is asserted separately."""

    def test_every_named_area_ships_python(self) -> None:
        """An area renamed or mistyped would silently drop out of the sweep."""
        for area in _AREAS:
            path = REPO_ROOT / area
            assert path.is_dir(), f"{area} is named in _AREAS and is not a directory"
            assert any(path.rglob("*.py")), f"{area} is named in _AREAS and ships no Python"

    def test_the_sweep_finds_the_idiom_it_grades(self) -> None:
        """The idiom is in the tree, so an empty population means the derivation broke.

        Deliberately weaker than a count or a path list: the sites are ordinary
        code that may be rewritten for reasons of their own, and pinning them
        here would turn an unrelated refactor into a failure of this file. What
        cannot happen quietly is the derivation matching nothing at all, which
        is how a grader keeps passing after it has stopped grading.
        """
        assert _cast_string_names(), (
            "No cast-string reference to a TYPE_CHECKING import was found anywhere in "
            f"{', '.join(_AREAS)}. Either the idiom has left the tree - in which case delete this "
            "module and the AGENTS.md entry it pins - or the derivation above no longer matches it."
        )
