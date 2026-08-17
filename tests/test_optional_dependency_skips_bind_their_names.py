"""An optional-dependency skip must bind its names on every path to their use.

The suite gates on optional dependencies constantly, and the shortest way to
write that gate is a ``try`` whose handler calls :func:`pytest.skip`::

    try:
        from libero.libero.envs.robots.mounted_panda import MountedPanda
    except ImportError:
        pytest.skip("libero does not expose MountedPanda")
    qpos = MountedPanda().init_qpos          # <- bound only on the try path

That runs correctly, because ``pytest.skip`` raises. But the binding's
liveness is then a property of pytest's control flow rather than of the
enclosing function, so ``py/uninitialized-local-variable`` reports the use --
and every code-scanning alert opens a review thread that has to be resolved
before a merge, on whichever pull request happens to touch the file next.

Three sites of this shape carried five of those alerts on ``main`` for long
enough to become the lowest-numbered open alerts in the repository, and the
shape was reintroduced once after being fixed, which is what this check is
for: the same mistake now fails in a fraction of a second locally instead of
arriving as an alert after the merge.

The remedy is the suite's own idiom, and it is shorter than what it replaces:

* a missing **module** -- :func:`pytest.importorskip`, already used at ~890
  call sites here;
* a missing **attribute** on a module that does import -- ``getattr(module,
  name, None)`` plus an explicit skip, which keeps an upstream rename a skip
  rather than converting it into an ``AttributeError``;
* a **value** that has to be built (a model load, a decode) -- a module-level
  ``*_or_skip`` helper that returns it, so the caller's binding is
  unconditional. :func:`pytest.skip` is annotated ``NoReturn``, so a helper
  needs no fallback return.

Scope is deliberately the skip-handler shape and nothing wider. A handler
that ends in ``return``, ``raise`` or :func:`pytest.fail` leaves the same
name conditionally bound, and none of those is reported: measured against
``main``, the tree carries 20 ``return`` sites, 4 ``pytest.fail`` sites and 1
``raise`` site of the same structure with zero alerts between them. Those
forms are idiomatic here -- a ``pytest.fail`` handler asserts the ``try`` body
succeeds, rather than gating on an environment -- so widening this check to
them would trade a bounded rule for a large mechanical rewrite that no gate
asks for.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEST_ROOTS = ("tests", "tests_integ")

# A sweep that reaches nothing would pass silently, so require that it saw a
# plausible amount of the tree. These are floors, not measurements.
_MINIMUM_FILES_SCANNED = 400
_MINIMUM_TRY_STATEMENTS_SCANNED = 200

# ``importorskip`` is deliberately absent: called from a handler it raises only
# when its own module is missing, so such a handler can fall through.
_SKIPPING_CALLS = frozenset({"skip"})


def _leaves_via_skip(handler: ast.ExceptHandler) -> bool:
    """Whether this handler's only exit is a :func:`pytest.skip` call.

    Args:
        handler: Exception handler to classify.

    Returns:
        ``True`` when the handler reaches a skip before any ``return`` or
        ``raise``. A handler that returns or raises is out of scope (see the
        module docstring).
    """
    for statement in handler.body:
        if isinstance(statement, (ast.Return, ast.Raise)):
            return False
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            func = statement.value.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in _SKIPPING_CALLS:
                return True
            if name in {"fail", "xfail", "exit"}:
                return False
    return False


def _statements_excluding_nested_scopes(node: ast.AST) -> list[ast.AST]:
    """Walk ``node`` without descending into a nested function or class.

    A name assigned inside a nested definition is that scope's local, not the
    one under test, so counting it would report bindings that were never in
    question.

    Args:
        node: Statement to walk.

    Returns:
        Every descendant node in the same scope, including ``node`` itself.
    """
    collected: list[ast.AST] = []
    queue: list[ast.AST] = [node]
    while queue:
        current = queue.pop()
        collected.append(current)
        for child in ast.iter_child_nodes(current):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            queue.append(child)
    return collected


def _names_bound_by(statements: list[ast.stmt]) -> set[str]:
    """Collect the plain names ``statements`` bind in their own scope.

    Args:
        statements: Statements to inspect.

    Returns:
        Every name bound by an import, assignment or ``with ... as``.
    """
    bound: set[str] = set()
    for statement in statements:
        for node in _statements_excluding_nested_scopes(statement):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bound.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                bound.add(node.target.id)
            elif isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
                bound.add(node.optional_vars.id)
    return bound


def _conditionally_bound_names(source: str) -> list[tuple[int, str]]:
    """Report every name a skipping ``try`` binds and later code reads.

    Args:
        source: Python source text.

    Returns:
        ``(line, name)`` pairs, sorted, one per offending name. A name already
        bound earlier in the same statement list is excluded: pre-binding is a
        valid fix for this shape and is in use in the tree.
    """
    tree = ast.parse(source)
    findings: set[tuple[int, str]] = set()
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(parent, field, None)
            if not isinstance(statements, list):
                continue
            for index, statement in enumerate(statements):
                if not isinstance(statement, ast.Try) or not statement.handlers:
                    continue
                if not all(_leaves_via_skip(handler) for handler in statement.handlers):
                    continue
                candidates = _names_bound_by(statement.body) - _names_bound_by(list(statements[:index]))
                if not candidates:
                    continue
                for later in statements[index + 1 :]:
                    for node in ast.walk(later):
                        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in candidates:
                            findings.add((statement.lineno, node.id))
    return sorted(findings)


def _count_try_statements(source: str) -> int:
    """Count ``try`` statements in ``source``.

    Args:
        source: Python source text.

    Returns:
        The number of ``try`` statements, used as a non-vacuity floor.
    """
    return sum(1 for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Try))


def _test_sources() -> list[tuple[Path, str]]:
    """Read every Python file under the test roots.

    Returns:
        ``(path, source)`` pairs, sorted by path.
    """
    sources: list[tuple[Path, str]] = []
    for root in _TEST_ROOTS:
        for path in sorted((_REPO_ROOT / root).rglob("*.py")):
            sources.append((path, path.read_text(encoding="utf-8")))
    return sources


class TestASkippingGuardBindsItsNames:
    """No test binds a name only inside a ``try`` whose handler skips."""

    def test_no_name_is_bound_only_on_the_non_skipping_path(self) -> None:
        """Every name a skipping guard binds is bound on all paths to its use."""
        sources = _test_sources()
        assert len(sources) >= _MINIMUM_FILES_SCANNED, (
            f"scan reached only {len(sources)} files under {_TEST_ROOTS}; "
            f"expected at least {_MINIMUM_FILES_SCANNED}, so a clean result here proves nothing"
        )
        tries = sum(_count_try_statements(source) for _, source in sources)
        assert tries >= _MINIMUM_TRY_STATEMENTS_SCANNED, (
            f"scan reached only {tries} try statements; expected at least "
            f"{_MINIMUM_TRY_STATEMENTS_SCANNED}, so a clean result here proves nothing"
        )

        offenders = [
            f"{path.relative_to(_REPO_ROOT)}:{line} binds {name!r}"
            for path, source in sources
            for line, name in _conditionally_bound_names(source)
        ]
        assert not offenders, (
            "a name bound inside a try whose handler calls pytest.skip is read after the block, so it is "
            "bound only on the success path as far as any analysis of the enclosing function can tell "
            "(py/uninitialized-local-variable):\n  "
            + "\n  ".join(offenders)
            + "\nUse pytest.importorskip for a missing module, getattr(module, name, None) plus an explicit "
            "skip for a missing attribute, or a module-level *_or_skip helper that returns the value."
        )

    def test_the_detector_reports_a_planted_skipping_guard(self) -> None:
        """A clean sweep means the tree is clean, not that the rule is inert."""
        planted = (
            "import pytest\n"
            "def test_x():\n"
            "    try:\n"
            "        from somewhere import Thing\n"
            "    except ImportError:\n"
            "        pytest.skip('absent')\n"
            "    assert Thing\n"
        )
        assert _conditionally_bound_names(planted) == [(3, "Thing")]

    def test_the_detector_reports_a_planted_value_built_under_a_skip(self) -> None:
        """The shape covers a built value, not only an import."""
        planted = (
            "import pytest\n"
            "def test_x():\n"
            "    try:\n"
            "        policy = load()\n"
            "    except Exception as exc:\n"
            "        pytest.skip(f'no model: {exc}')\n"
            "    return wrap(policy)\n"
        )
        assert _conditionally_bound_names(planted) == [(3, "policy")]

    @pytest.mark.parametrize(
        ("label", "handler_body"),
        [
            ("fail", "        pytest.fail('must not happen')"),
            ("return", "        return"),
            ("raise", "        raise AssertionError('leaked')"),
        ],
    )
    def test_a_non_skipping_handler_is_out_of_scope(self, label: str, handler_body: str) -> None:
        """Only the reported shape is in scope; see the module docstring."""
        source = (
            "import pytest\n"
            "def test_x():\n"
            "    try:\n"
            "        value = compute()\n"
            "    except ValueError:\n"
            f"{handler_body}\n"
            "    assert value\n"
        )
        assert _conditionally_bound_names(source) == [], f"{label} handler must not be reported"

    def test_pre_binding_before_the_try_resolves_the_shape(self) -> None:
        """Binding the name first is a valid fix and must not be reported."""
        source = (
            "import pytest\n"
            "def test_x():\n"
            "    cls = None\n"
            "    try:\n"
            "        cls = resolve()\n"
            "    except ImportError:\n"
            "        pytest.skip('absent')\n"
            "    assert cls\n"
        )
        assert _conditionally_bound_names(source) == []

    def test_a_name_the_try_never_binds_is_not_reported(self) -> None:
        """A guard that binds nothing has nothing to report."""
        source = (
            "import pytest\n"
            "def test_x(module):\n"
            "    try:\n"
            "        module.check()\n"
            "    except ImportError:\n"
            "        pytest.skip('absent')\n"
            "    assert module\n"
        )
        assert _conditionally_bound_names(source) == []
