"""A test may not leave ``sys.modules`` missing a module a sibling patches.

Removing an entry from ``sys.modules`` and not putting it back does not undo
an import - it *orphans* every reference already bound to that module. A test
module that does ``import boto3`` at collection time keeps the original module
object; once the entry is gone, the next ``import boto3`` executes the package
again and returns a **different** object. So::

    monkeypatch.setattr(boto3, "client", fake)   # patches the orphan
    exec(deployed_source)                        # its `import boto3` gets the
                                                 # fresh module - the real SDK

Measured consequence, in the ordering the full suite actually collects:
``tests/mesh/test_iot_camera_offload.py`` used to end
``test_boto3_missing_returns_none`` with a bare ``sys.modules.pop("boto3",
None)``. Three tests in ``tests/mesh/test_mesh_role_attribute_is_reserved.py``
then reached the real SDK instead of their double and failed with
``botocore.exceptions.NoRegionError`` - a unit test attempting a real AWS call,
and one that passes in isolation and on any host where a default region happens
to be configured. The pop was also redundant: the ``builtins.__import__`` block
on the line above already makes the function-scope ``import boto3`` raise
``ImportError``, which is what that test asserts, and ``monkeypatch`` undoes it.

The rule graded here is derived from the tree rather than listed:

* **Protected** is every module a test file binds with a module-level import -
  ``import a.b`` *or* ``from a import b`` - *and* then patches an attribute on
  (``monkeypatch.setattr(<binding>, ...)``). Those are exactly the references a
  removal can orphan; a file that only *reads* ``mod.attr`` is unaffected by
  getting a fresh copy.
* **Reported** is a removal of a protected module with no restoration in the
  same function.

Both import statements have to be read, because ``from a import b`` is how this
tree binds nearly every submodule - a submodule is what a test wants to patch
attributes on. Reading ``import`` alone left the protected set at 43 modules of
95, and the 52 it could not see included two removals with live symptoms:
``tests/tools/g1/test_motion_switcher_decoder.py`` dropped
``strands_robots.tools.g1._motion_switcher``, orphaning the reference
``tests/drivers/test_motion_switcher_open_is_under_the_shared_dds_lock.py``
patches ``_load_motion_switcher_client`` on, so the driver's lazy import reached
the real loader and the open returned ``None`` - that file's cells grade whether
an RPC client is constructed under the shared DDS lock, and they reported "the
open returned None instead of the recorder, so this cell graded nothing";
``tests/simulation/test_policy_runner.py`` dropped
``strands_robots.simulation.policy_runner``, and four cells in
``tests/simulation/test_recording_frame_loss_is_not_tolerated.py`` then failed
with ``KeyError`` on that entry. Both pass in isolation.

It is deliberately one-directional and under-reports rather than over-reports:

* Only a **literal** key is graded. A removal whose key is a variable (a loop
  purging ``mujoco*``, say) is out of reach of a static read and is not claimed.
* Every name bound to ``sys`` in the file is followed, so an aliased
  ``import sys as _sys`` is graded on both sides of the rule - four files use
  that spelling, all of them with the restoring idiom.
* Any ``finally``, ``patch.dict``, ``monkeypatch.setitem`` or re-assignment in
  the same function counts as restoring, without checking that it restores the
  same key.
* Purging a module **no test patches** stays legal. That is a deliberate
  cache-invalidation idiom here - ``tests/policies/lerobot_local/
  test_resolution.py`` drops ``lerobot.*`` to force re-registration, and it
  cannot orphan a patched reference. ``tests/simulation/test_policy_runner.py``
  used to be described as the same idiom; it is not, because two sibling
  modules patch the runner through a ``from`` import, so it is graded.

``monkeypatch.setitem(sys.modules, name, None)`` is the idiom for "make
``import name`` raise ``ImportError``": it has the same effect and it restores.
``tests/mesh/test_iot_camera_offload.py`` uses it for ``cv2`` in the same file.

A second rule lives here, for the cells that remove an entry in order to
**import the module again**. ``importlib.import_module`` binds a submodule in
two places - the ``sys.modules`` entry, and an attribute of the same name on its
parent package - and ``monkeypatch.delitem`` restores only the first. Restoring
one of the two is worse than restoring neither: with neither restored both halves
hold the fresh module and agree, while with only the entry restored the two
disagree for the rest of the session, and which one a spelling reaches is not
visible at the call site. Measured after
``test_policy_runner_import_does_not_pull_in_mujoco``, with the entry restored::

    import a.b.c as m                 -> the fresh module
    import a.b.c ; a.b.c              -> the fresh module
    from a.b import c as m            -> the fresh module
    sys.modules["a.b.c"]              -> the original
    from a.b.c import f               -> the original's f
    <the live code's own globals>     -> the original

So a cell that binds the module *inside* its body and patches an attribute on it
patches the fresh copy, while production's ``from a.b.c import f`` reads the
original: the stub is silently inert and the cell grades nothing. That is not
hypothetical. With ``strands_robots.simulation.policy_runner`` split this way,
``tests/tools/test_run_policy.py`` binds it in-cell to install a ``PolicyRunner``
that raises, and the two cells pinning that
``strands_robots.tools.run_policy._finalize_episode`` tolerates a construction
error and a save error keep passing with both ``except Exception`` guards deleted
from production - they alone report the deletion.

:func:`tests._module_reimport.reimport` records both bindings, and
:class:`TestAReimportPutsTheParentBindingBack` grades that every re-importing
cell goes through it or an equivalent ``setattr``. Two cells are in scope. The
two others that remove-and-import are not, for reasons the scan reads rather
than lists: ``tests/test_dashboard_extra_is_declared.py`` drives an import that
*raises*, which binds nothing, and ``tests/test_device_connect_drivers.py``
never restores the entry, so both halves agree on the fresh module.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import strands_robots

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_TEST_TREES = ("tests", "tests_integ")

#: A protected set smaller than this means the scan stopped reaching the tree.
_MINIMUM_PROTECTED = 10


def _module_level_bindings(tree: ast.Module) -> dict[str, str]:
    """Map each name a module-level import binds to the dotted path it names.

    Both import statements bind a module object a removal can orphan:

    * ``import a.b`` binds ``a`` to ``a``, and ``import a.b as c`` binds ``c``
      to ``a.b``.
    * ``from a import b`` binds ``b`` to ``a.b`` - the spelling this tree uses
      for almost every submodule, because a submodule is what a test wants to
      patch attributes on (``from strands_robots.simulation import
      policy_runner``).

    A ``from`` import can also name something that is not a module
    (``from a import SomeClass`` records ``a.SomeClass``). Recording it is
    harmless rather than a source of false reports: the rule only ever
    intersects these names with the literal keys a test removes from
    ``sys.modules``, and a class is not registered there under that spelling,
    so an entry that is not a module can never be matched by a removal.

    Star imports bind no inspectable name and relative imports cannot be
    resolved to a dotted path from the file alone, so neither is claimed.
    """
    bindings: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bindings


def _graded_files() -> list[Path]:
    """Every Python file under the test trees."""
    return sorted(p for tree in _TEST_TREES for p in (_REPO_ROOT / tree).rglob("*.py"))


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):  # pragma: no cover - defensive
        return None


def _patched_module_level_imports(tree: ast.Module) -> set[str]:
    """Dotted names this module binds at import time and patches attributes on."""
    bindings = _module_level_bindings(tree)
    if not bindings:
        return set()
    patched: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setattr"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in bindings
        ):
            dotted = bindings[node.args[0].id]
            if dotted.split(".")[0] not in sys.stdlib_module_names:
                patched.add(dotted)
    return patched


def protected_modules() -> dict[str, set[str]]:
    """Modules whose identity a removal can orphan, and the files that patch them."""
    protected: dict[str, set[str]] = {}
    for path in _graded_files():
        tree = _parse(path)
        if tree is None:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for dotted in _patched_module_level_imports(tree):
            protected.setdefault(dotted, set()).add(rel)
    return protected


def _sys_aliases(tree: ast.Module) -> set[str]:
    """Every name bound to the ``sys`` module in this file, at any scope."""
    aliases = {"sys"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(alias.asname or "sys" for alias in node.names if alias.name == "sys")
    return aliases


def _own_scope_removals(fn: ast.FunctionDef | ast.AsyncFunctionDef, registries: set[str]) -> list[tuple[int, str]]:
    """``(lineno, key)`` for each literal-key removal in *fn*'s own scope."""
    found: list[tuple[int, str]] = []

    def visit(node: ast.AST, *, top: bool = False) -> None:
        if not top and isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            return
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "pop"
            and ast.unparse(node.func.value) in registries
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.append((node.lineno, node.args[0].value))
        if isinstance(node, ast.Delete):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and ast.unparse(target.value) in registries
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    found.append((node.lineno, target.slice.value))
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in fn.body:
        visit(statement, top=True)
    return found


def _restores(fn: ast.FunctionDef | ast.AsyncFunctionDef, registries: set[str]) -> bool:
    """Whether *fn* puts something back. Permissive on purpose - see the module docstring."""
    source = ast.unparse(fn)
    if "finally" in source or "patch.dict" in source:
        return True
    return any(
        f"setitem({registry}" in source
        or f"{registry}.update" in source
        or (f"{registry}[" in source and "] =" in source)
        for registry in registries
    )


def unrestored_removals(tree: ast.Module) -> list[tuple[int, str, str]]:
    """``(lineno, function, key)`` for each literal removal *tree* never undoes."""
    registries = {f"{alias}.modules" for alias in _sys_aliases(tree)}
    reported: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) or _restores(node, registries):
            continue
        reported.extend((lineno, node.name, key) for lineno, key in _own_scope_removals(node, registries))
    return reported


def orphaning_removals() -> list[str]:
    """Every removal of a protected module the removing function does not undo."""
    protected = protected_modules()
    offenders: list[str] = []
    for path in _graded_files():
        tree = _parse(path)
        if tree is None:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for lineno, function, key in unrestored_removals(tree):
            if key in protected:
                holders = ", ".join(sorted(protected[key]))
                offenders.append(f"{rel}:{lineno} in {function}() removes {key!r}, which is patched by {holders}")
    return offenders


class TestNoRemovalOrphansAPatchedModule:
    """The rule."""

    def test_no_protected_module_is_removed_without_being_restored(self) -> None:
        offenders = orphaning_removals()
        assert offenders == [], (
            "a test removes a module a sibling test module patches attributes on, and "
            "does not put it back - the sibling's reference is orphaned, so its patch "
            "is invisible to the next import and the real package is used instead. "
            "Use monkeypatch.setitem(sys.modules, name, None) to make `import name` "
            "raise ImportError with restoration, or restore the entry in a finally:\n  " + "\n  ".join(offenders)
        )

    def test_the_protected_set_is_derived_from_the_test_tree(self) -> None:
        """So a clean result means the scan looked, rather than found nothing to look at."""
        protected = protected_modules()
        assert len(protected) >= _MINIMUM_PROTECTED, (
            f"only {len(protected)} modules read as protected; the scan is no longer "
            f"reaching {_TEST_TREES} under {_REPO_ROOT}"
        )
        assert "boto3" in protected, (
            "boto3 is bound at module level and attribute-patched by the IoT fan-out tests, "
            f"so it must be protected; got {sorted(protected)}"
        )

    def test_purging_a_module_no_test_patches_stays_legal(self) -> None:
        """The cache-invalidation idiom is not reported - it cannot orphan a patch."""
        protected = protected_modules()
        purges = ["lerobot.policies.vla_jepa.processor_vla_jepa"]
        assert [key for key in purges if key in protected] == [], (
            "these are deliberately dropped to force a re-import; if a test starts "
            "patching one through a module-level binding the rule reaches it, but "
            f"today it must not: {sorted(protected)}"
        )

    def test_a_module_patched_through_a_from_import_is_protected(self) -> None:
        """The binding this tree uses for a submodule is the one to reach.

        Both of these are bound with ``from <package> import <module>`` and
        attribute-patched, so a removal of either orphans a double. They are
        named rather than derived because a rule that lost this statement would
        empty the population it grades and still report a clean tree.
        """
        protected = protected_modules()
        expected = {
            "strands_robots.simulation.policy_runner",
            "strands_robots.tools.g1._motion_switcher",
        }
        assert expected <= set(protected), (
            "a module bound by `from package import module` and then attribute-"
            f"patched must be protected; missing {sorted(expected - set(protected))}"
        )


class TestTheScanIsSpecific:
    """Planted sources, so a clean tree means the rule works rather than accepts anything."""

    def test_an_unrestored_literal_removal_is_reported(self) -> None:
        source = "\n".join(
            [
                "import sys",
                "def test_x(monkeypatch):",
                "    sys.modules.pop('boto3', None)",
            ]
        )
        assert unrestored_removals(ast.parse(source)) == [(3, "test_x", "boto3")]

    def test_a_del_statement_is_reported_too(self) -> None:
        source = "\n".join(["import sys", "def test_x():", "    del sys.modules['boto3']"])
        assert unrestored_removals(ast.parse(source)) == [(3, "test_x", "boto3")]

    def test_a_restored_removal_is_accepted(self) -> None:
        source = "\n".join(
            [
                "import sys",
                "def test_x():",
                "    held = sys.modules.pop('boto3')",
                "    try:",
                "        pass",
                "    finally:",
                "        sys.modules['boto3'] = held",
            ]
        )
        assert unrestored_removals(ast.parse(source)) == []

    def test_the_restoring_monkeypatch_idiom_is_accepted(self) -> None:
        source = "\n".join(
            ["import sys", "def test_x(monkeypatch):", "    monkeypatch.setitem(sys.modules, 'boto3', None)"]
        )
        assert unrestored_removals(ast.parse(source)) == []

    def test_an_aliased_sys_is_followed_on_both_sides(self) -> None:
        """Four files spell it ``import sys as _sys``; the rule must not lose them."""
        removal = "\n".join(["import sys as _sys", "def test_x():", "    _sys.modules.pop('boto3', None)"])
        restored = "\n".join(
            [
                "import sys as _sys",
                "def test_x(monkeypatch):",
                "    monkeypatch.setitem(_sys.modules, 'boto3', None)",
            ]
        )
        assert unrestored_removals(ast.parse(removal)) == [(3, "test_x", "boto3")]
        assert unrestored_removals(ast.parse(restored)) == []

    def test_a_dynamic_key_is_not_claimed(self) -> None:
        """A key a static read cannot resolve is out of scope rather than guessed at."""
        source = "\n".join(
            [
                "import sys",
                "def test_x():",
                "    for name in [m for m in sys.modules if m.startswith('mujoco')]:",
                "        del sys.modules[name]",
            ]
        )
        assert unrestored_removals(ast.parse(source)) == []

    def test_a_from_import_binding_is_followed(self) -> None:
        """``from a import b`` binds the module ``a.b``, under either spelling."""
        plain = "\n".join(
            [
                "from pkg import mod",
                "def test_x(monkeypatch):",
                "    monkeypatch.setattr(mod, 'attr', None)",
            ]
        )
        aliased = "\n".join(
            [
                "from pkg import mod as aliased",
                "def test_x(monkeypatch):",
                "    monkeypatch.setattr(aliased, 'attr', None)",
            ]
        )
        assert _patched_module_level_imports(ast.parse(plain)) == {"pkg.mod"}
        assert _patched_module_level_imports(ast.parse(aliased)) == {"pkg.mod"}

    def test_a_star_or_relative_from_import_is_not_claimed(self) -> None:
        """Neither resolves to a dotted path from the file alone."""
        star = "\n".join(
            ["from pkg import *", "def test_x(monkeypatch):", "    monkeypatch.setattr(mod, 'attr', None)"]
        )
        relative = "\n".join(
            ["from . import mod", "def test_x(monkeypatch):", "    monkeypatch.setattr(mod, 'attr', None)"]
        )
        assert _patched_module_level_imports(ast.parse(star)) == set()
        assert _patched_module_level_imports(ast.parse(relative)) == set()

    def test_a_from_imported_non_module_is_recorded_but_matches_no_removal(self) -> None:
        """Why recording ``pkg.SomeClass`` costs nothing.

        The rule intersects the protected names with the literal keys a test
        removes from ``sys.modules``. A class is not registered there under that
        spelling, so the entry can never be matched - which is what lets the
        binding collector stay a static read with no import side effects.
        """
        source = "\n".join(
            [
                "import sys",
                "from pkg import SomeClass",
                "def test_x(monkeypatch):",
                "    monkeypatch.setattr(SomeClass, 'attr', None)",
                "def test_y():",
                "    sys.modules.pop('pkg', None)",
            ]
        )
        tree = ast.parse(source)
        assert _patched_module_level_imports(tree) == {"pkg.SomeClass"}
        removed = {key for _, _, key in unrestored_removals(tree)}
        assert removed == {"pkg"}, removed
        assert removed & _patched_module_level_imports(tree) == set()

    def test_a_module_only_read_is_not_protected(self) -> None:
        """Reading ``mod.attr`` survives a fresh import; patching it does not."""
        reader = "\n".join(["import boto3", "def test_x():", "    assert boto3.client is not None"])
        patcher = "\n".join(
            ["import boto3", "def test_x(monkeypatch):", "    monkeypatch.setattr(boto3, 'client', None)"]
        )
        assert _patched_module_level_imports(ast.parse(reader)) == set()
        assert _patched_module_level_imports(ast.parse(patcher)) == {"boto3"}


@pytest.fixture
def probe_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """An importable throwaway module, left out of ``sys.modules`` afterwards."""
    name = "strands_orphan_probe"
    (tmp_path / f"{name}.py").write_text("value = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    try:
        yield name
    finally:
        sys.modules.pop(name, None)


class TestTheOrphaningMechanism:
    """What the rule protects against, driven on a throwaway module."""

    def test_a_removal_orphans_a_reference_taken_before_it(self, probe_module: str) -> None:
        name = probe_module
        held = __import__(name)
        assert sys.modules[name] is held, "premise: the binding is the registered module"

        del sys.modules[name]
        fresh = __import__(name)

        assert fresh is not held, "a fresh import after a removal must be a different module object"

    def test_a_patch_on_the_orphan_is_invisible_to_the_next_import(
        self, probe_module: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact failure mode: the double is installed somewhere nothing will look."""
        name = probe_module
        held = __import__(name)
        monkeypatch.setattr(held, "value", "double")

        del sys.modules[name]
        fresh = __import__(name)

        assert held.value == "double", "premise: the patch reached the object it was applied to"
        assert fresh.value == 1, "the fresh module does not carry the patch - the double is orphaned"


#: A re-importing population smaller than this means the second scan stopped
#: reaching the tree. Two cells re-import a module they removed; both are
#: named in :meth:`TestAReimportPutsTheParentBindingBack.test_the_reimporting_cells_are_found`.
_MINIMUM_REIMPORTS = 2

#: The shared owner of "remove, import again, put both bindings back".
_REIMPORT_HELPER = "reimport"


def _literal_str(node: ast.AST, names: dict[str, str]) -> str | None:
    """The string *node* evaluates to, following simple local bindings."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    return None


def _string_names(*scopes: ast.AST) -> dict[str, str]:
    """``name -> value`` for each ``name = "literal"`` assignment in *scopes*.

    The key a cell removes is as often bound to a local as written inline
    (``runner = "strands_robots.simulation.policy_runner"``), so a scan that
    reads only ``ast.Constant`` cannot see the cells this rule exists for.
    Anything more involved than a plain string assignment is out of reach of a
    static read and is not claimed.
    """
    bound: dict[str, str] = {}
    for scope in scopes:
        for node in ast.walk(scope):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bound[target.id] = node.value.value
    return bound


def _lines_under_raises(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[int]:
    """Line numbers inside a ``with pytest.raises(...)`` block in *fn*.

    An import that raises binds nothing - not the entry and not the attribute -
    so a cell driving a *failing* import has nothing to put back.
    ``tests/test_dashboard_extra_is_declared.py`` is that shape.
    """
    lines: set[int] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.With | ast.AsyncWith) and any(
            "raises" in ast.unparse(item.context_expr) for item in node.items
        ):
            lines.update(getattr(child, "lineno", -1) for child in ast.walk(node))
    return lines


def _restored_removal_keys(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, registries: set[str], names: dict[str, str]
) -> set[str]:
    """Keys *fn* removes from ``sys.modules`` through the restoring idiom.

    Only ``monkeypatch.delitem`` is read. A removal that is *not* restored is
    the other rule's business, and it cannot produce the asymmetry this one
    grades: with the entry left holding the fresh module, the parent attribute
    holds the same fresh module and the two paths still agree.
    """
    keys: set[str] = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "delitem"
            and len(node.args) >= 2
            and ast.unparse(node.args[0]) in registries
        ):
            if key := _literal_str(node.args[1], names):
                keys.add(key)
    return keys


def _reimported_keys(fn: ast.FunctionDef | ast.AsyncFunctionDef, names: dict[str, str]) -> dict[str, int]:
    """``key -> lineno`` for every module *fn* imports, outside a ``raises`` block."""
    exempt = _lines_under_raises(fn)
    found: dict[str, int] = {}
    for node in ast.walk(fn):
        if getattr(node, "lineno", -1) in exempt:
            continue
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("import_module") and node.args:
            if key := _literal_str(node.args[0], names):
                found.setdefault(key, node.lineno)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found.setdefault(alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.setdefault(node.module, node.lineno)
    return found


def _delegated_keys(fn: ast.FunctionDef | ast.AsyncFunctionDef, names: dict[str, str]) -> dict[str, int]:
    """``key -> lineno`` for each module *fn* re-imports through the shared owner.

    A delegating cell is *in scope and satisfied*, rather than invisible. That
    is what keeps the population the same size before and after a cell is
    moved onto the owner, so the floor in
    :meth:`TestAReimportPutsTheParentBindingBack.test_the_reimporting_cells_are_found`
    means "the scan is still reaching the tree" rather than "nobody re-imports
    anything today".
    """
    found: dict[str, int] = {}
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and (ast.unparse(node.func) == _REIMPORT_HELPER or ast.unparse(node.func).endswith(f".{_REIMPORT_HELPER}"))
            and len(node.args) >= 2
        ):
            if key := _literal_str(node.args[1], names):
                found.setdefault(key, node.lineno)
    return found


def _restores_the_leaf(fn: ast.FunctionDef | ast.AsyncFunctionDef, leaf: str) -> bool:
    """Whether *fn* records an attribute named *leaf* for restoration itself.

    Permissive in the same way :func:`_restores` is: the point is that the pair
    is kept together, not which spelling keeps it.
    """
    return any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func).endswith("setattr")
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == leaf
        for node in ast.walk(fn)
    )


def reimporting_cells(tree: ast.Module) -> list[tuple[int, str, str, bool]]:
    """``(lineno, function, key, puts_it_back)`` per cell that re-imports what it removed.

    In scope either way it is spelled: the cell removes the entry through
    ``monkeypatch.delitem`` and imports the module again itself, or it asks the
    shared owner to do both.
    """
    registries = {f"{alias}.modules" for alias in _sys_aliases(tree)}
    module_names = _string_names(*tree.body)
    reported: list[tuple[int, str, str, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        names = module_names | _string_names(node)
        delegated = _delegated_keys(node, names)
        reimported = _reimported_keys(node, names)
        hand_rolled = _restored_removal_keys(node, registries, names) & set(reimported)
        for key in sorted(set(delegated) | hand_rolled):
            if "." not in key:
                continue  # a top-level module has no parent package to rebind
            lineno = delegated.get(key) or reimported[key]
            puts_it_back = key in delegated or _restores_the_leaf(node, key.rpartition(".")[2])
            reported.append((lineno, node.name, key, puts_it_back))
    return reported


def splitting_reimports() -> list[str]:
    """Every cell that re-imports a module and leaves the parent binding split."""
    offenders: list[str] = []
    for path in _graded_files():
        tree = _parse(path)
        if tree is None:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for lineno, function, key, restored in reimporting_cells(tree):
            if not restored:
                parent, _, leaf = key.rpartition(".")
                offenders.append(
                    f"{rel}:{lineno} in {function}() re-imports {key!r} without restoring {leaf!r} on {parent}"
                )
    return offenders


class TestAReimportPutsTheParentBindingBack:
    """The second rule: an import rebinds two things, so two go back."""

    def test_no_reimport_leaves_the_parent_binding_split(self) -> None:
        offenders = splitting_reimports()
        assert offenders == [], (
            "a test removes a sys.modules entry, imports the module again, and restores "
            "only the entry - so the fresh module stays bound as an attribute of its "
            "parent package while sys.modules holds the original, and for the rest of "
            "the session `import a.b.c as m` and `from a.b.c import f` name different "
            "objects. A stub then goes on one and the code under test reads the other. "
            "Import through tests._module_reimport.reimport, which records both:\n  " + "\n  ".join(offenders)
        )

    def test_the_reimporting_cells_are_found(self) -> None:
        """So a clean result means the scan looked, rather than found nothing to look at."""
        found = {
            (path.relative_to(_REPO_ROOT).as_posix(), key)
            for path in _graded_files()
            if (tree := _parse(path)) is not None
            for _, _, key, _ in reimporting_cells(tree)
        }
        assert len(found) >= _MINIMUM_REIMPORTS, (
            f"only {len(found)} re-importing cells read as in scope; the scan is no "
            f"longer reaching {_TEST_TREES} under {_REPO_ROOT}: {sorted(found)}"
        )
        expected = {
            ("tests/simulation/test_policy_runner.py", "strands_robots.simulation.policy_runner"),
            ("tests/tools/g1/test_motion_switcher_decoder.py", "strands_robots.tools.g1._motion_switcher"),
        }
        assert expected <= found, (
            "these cells remove an entry and import the module again, so the rule has "
            f"to reach them; missing {sorted(expected - found)}"
        )


class TestTheReimportScanIsSpecific:
    """Planted sources, so a clean tree means the rule works rather than accepts anything."""

    _REMOVE_AND_IMPORT = (
        "import importlib",
        "import sys",
        "def test_x(monkeypatch):",
        "    monkeypatch.delitem(sys.modules, 'a.b.c', raising=False)",
        "    importlib.import_module('a.b.c')",
    )

    def test_restoring_only_the_entry_is_reported(self) -> None:
        source = "\n".join(self._REMOVE_AND_IMPORT)
        assert reimporting_cells(ast.parse(source)) == [(5, "test_x", "a.b.c", False)]

    def test_the_shared_owner_is_accepted(self) -> None:
        source = "\n".join(
            [
                "from tests._module_reimport import reimport",
                "def test_x(monkeypatch):",
                "    reimport(monkeypatch, 'a.b.c')",
            ]
        )
        assert reimporting_cells(ast.parse(source)) == [(3, "test_x", "a.b.c", True)]

    def test_an_explicit_setattr_on_the_parent_is_accepted(self) -> None:
        source = "\n".join(
            [
                *self._REMOVE_AND_IMPORT[:4],
                "    monkeypatch.setattr(a.b, 'c', a.b.c)",
                "    importlib.import_module('a.b.c')",
            ]
        )
        assert reimporting_cells(ast.parse(source)) == [(6, "test_x", "a.b.c", True)]

    def test_a_key_bound_to_a_local_is_read(self) -> None:
        """The spelling the graded cells use, and the one a literal-only scan misses."""
        source = "\n".join(
            [
                "import importlib",
                "import sys",
                "def test_x(monkeypatch):",
                "    name = 'a.b.c'",
                "    monkeypatch.delitem(sys.modules, name, raising=False)",
                "    importlib.import_module(name)",
            ]
        )
        assert reimporting_cells(ast.parse(source)) == [(6, "test_x", "a.b.c", False)]

    def test_an_import_that_raises_is_not_claimed(self) -> None:
        """A failing import binds nothing, so it leaves nothing to put back."""
        source = "\n".join(
            [
                "import importlib",
                "import sys",
                "import pytest",
                "def test_x(monkeypatch):",
                "    monkeypatch.delitem(sys.modules, 'a.b.c', raising=False)",
                "    with pytest.raises(ImportError):",
                "        importlib.import_module('a.b.c')",
            ]
        )
        assert reimporting_cells(ast.parse(source)) == []

    def test_an_unrestored_removal_is_the_other_rule(self) -> None:
        """Both halves then hold the fresh module, so this rule has nothing to say."""
        source = "\n".join(
            [
                "import importlib",
                "import sys",
                "def test_x():",
                "    del sys.modules['a.b.c']",
                "    importlib.import_module('a.b.c')",
            ]
        )
        assert reimporting_cells(ast.parse(source)) == []

    def test_a_top_level_module_is_not_claimed(self) -> None:
        """There is no parent package to rebind."""
        source = "\n".join(
            [
                "import importlib",
                "import sys",
                "def test_x(monkeypatch):",
                "    monkeypatch.delitem(sys.modules, 'boto3', raising=False)",
                "    importlib.import_module('boto3')",
            ]
        )
        assert reimporting_cells(ast.parse(source)) == []
