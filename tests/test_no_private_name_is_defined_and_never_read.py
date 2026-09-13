"""A private module-level name or method must have a reader.

This is the module-scope half of the invariant
:mod:`tests.test_no_private_state_is_written_and_never_read` holds for instance
attributes. A ``_name`` bound at module scope - a constant, a function, a
class - or a ``_method`` defined in a class body, that nothing in the tree
refers to, is dead code by the project's own rule ("if it's not called and not
part of base class, delete it"), and it is dead in the direction that costs a
reader: a private constant spelling a vocabulary reads as the place that
vocabulary is owned, so the reader goes looking for the call sites that
consult it and finds that every consumer spells the literal instead.

Two instances on the tree this arrived in, out of 714 private methods and
every module-level private name in the package. ``_VALID_MODES`` in
:mod:`strands_robots.robot` named the three mode spellings while the three
sites that decide or report a mode (``_auto_detect_mode``'s membership test,
its warning and the factory's ``ValueError``) each spelled them inline, so the
constant documented an owner that did not exist. ``_PEERS_VERSION`` in
:mod:`strands_robots.mesh.session` was the other kind - a registry version
counter incremented under the lock at every insert, eviction, prune and
clear, and consulted by nothing, so four write sites and three ``global``
declarations maintained a change signal no cache ever read.

A reader is any of

* a load of the identifier anywhere in the package - a ``Name`` in ``Load``
  context, an ``Attribute`` reached through ``self`` or a module alias, or an
  ``import`` of the name from another module;
* a string literal containing the identifier anywhere in the package, which
  covers the ``getattr(module, "_name")`` shape and a ``__all__``-style
  ledger that names the symbol;
* any mention of the identifier under ``tests/``, ``tests_integ/``,
  ``examples/`` or ``scripts/`` - an example that reaches a backend's private
  helper through ``getattr(sim, "_name", None)`` is a reader the package
  cannot see, and it ships in the same tree. This file's own text is left out
  of that read, so naming an offender here does not clear it.

Dunder names are out of scope (``__all__``, ``__getattr__`` and friends are
protocol, not state), and so is the bare ``_`` throwaway. A name that is
defined in several modules counts as read if any of them reads it, the same
simplification the instance-state grader makes for a mixin's state.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "strands_robots"
MENTION_TREES = (REPO_ROOT / "tests", REPO_ROOT / "tests_integ", REPO_ROOT / "examples", REPO_ROOT / "scripts")

# Private names that are defined and never read, and correctly so. One reason
# per entry; an entry without a reason that survives review is a bug in review.
DEFINED_WITHOUT_A_READER_BY_DESIGN: dict[str, str] = {}

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _is_private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__") and name != "_"


def _definitions(tree: ast.Module) -> list[tuple[str, int]]:
    """Private names bound at module scope, plus private methods in class bodies.

    Descends through module-level ``if`` / ``try`` / ``with`` blocks, which is
    where ``TYPE_CHECKING`` mirrors and optional-import fallbacks bind names,
    and stops at a function body: a name bound there is a local, not a symbol.
    """
    found: list[tuple[str, int]] = []

    def visit(statements: list[ast.stmt], *, in_class: bool) -> None:
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _is_private(node.name):
                    found.append((node.name, node.lineno))
            elif isinstance(node, ast.ClassDef):
                if not in_class and _is_private(node.name):
                    found.append((node.name, node.lineno))
                visit(node.body, in_class=True)
            elif in_class:
                continue
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    for name_node in ast.walk(target):
                        if isinstance(name_node, ast.Name) and _is_private(name_node.id):
                            found.append((name_node.id, node.lineno))
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                if isinstance(node.target, ast.Name) and _is_private(node.target.id):
                    found.append((node.target.id, node.lineno))
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                for field in ("body", "orelse", "finalbody"):
                    visit(getattr(node, field, []), in_class=False)
                for handler in getattr(node, "handlers", []):
                    visit(handler.body, in_class=False)

    visit(tree.body, in_class=False)
    return found


def _readers(tree: ast.Module) -> set[str]:
    """Every identifier the module refers to other than by defining it."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.rsplit(".", 1)[-1])
            if node.asname:
                names.add(node.asname)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.update(_IDENTIFIER.findall(node.value))
    return names


def _scan() -> tuple[dict[str, list[str]], int, int]:
    """Return (unread private name -> definition sites, definitions seen, files parsed)."""
    files = sorted(PACKAGE.rglob("*.py"))
    definitions: dict[str, list[str]] = defaultdict(list)
    read: set[str] = set()
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name, lineno in _definitions(tree):
            definitions[name].append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
        read |= _readers(tree)

    own = Path(__file__).resolve()
    mention_text = "\n".join(
        path.read_text(encoding="utf-8")
        for tree in MENTION_TREES
        if tree.is_dir()
        for path in tree.rglob("*.py")
        if path.resolve() != own
    )
    read |= set(_IDENTIFIER.findall(mention_text))

    unread = {
        name: sites
        for name, sites in definitions.items()
        if name not in read and name not in DEFINED_WITHOUT_A_READER_BY_DESIGN
    }
    return unread, sum(len(sites) for sites in definitions.values()), len(files)


def test_every_private_name_defined_in_the_package_has_a_reader() -> None:
    unread, _, _ = _scan()
    assert not unread, "Private names defined and never read:\n" + "\n".join(
        f"  {name}: {', '.join(sites)}" for name, sites in sorted(unread.items())
    )


def test_the_scan_reaches_the_package() -> None:
    """A scan that walked nothing would pass the check above vacuously."""
    _, definitions_seen, file_count = _scan()
    assert file_count > 200, f"only {file_count} package files parsed"
    assert definitions_seen > 1000, f"only {definitions_seen} private definitions seen"


def test_every_exemption_states_a_reason() -> None:
    for name, reason in DEFINED_WITHOUT_A_READER_BY_DESIGN.items():
        assert len(reason.split()) >= 5, f"{name} needs a reason, not a label: {reason!r}"


def test_a_module_constant_nothing_reads_is_reported() -> None:
    """The grader's own verdict on the shape it was written for, without the tree."""
    source = (
        "_VOCABULARY = ('sim', 'real')\n"
        "_USED = 3\n"
        "def _helper():\n"
        "    return _USED\n"
        "class _Owner:\n"
        "    def _unused_method(self):\n"
        "        pass\n"
        "    def _called(self):\n"
        "        return self._called\n"
        "def public():\n"
        "    return _helper(), _Owner()\n"
    )
    tree = ast.parse(source)
    defined = {name for name, _ in _definitions(tree)}
    unread = defined - _readers(tree)
    assert defined == {"_VOCABULARY", "_USED", "_helper", "_Owner", "_unused_method", "_called"}
    assert unread == {"_VOCABULARY", "_unused_method"}
