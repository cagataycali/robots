"""Regression: every ``lerobot`` symbol ``strands_robots`` imports must resolve.

``strands_robots`` integrates LeRobot through 45 deferred imports (policy
inference, dataset recording, teleoperation, motor buses). Every one of them is
lazy: it lives inside a function body, a ``try`` block, or a method, never at
module top level (so optional-dep absence does not break ``import
strands_robots``). The flip side is that a plain ``import strands_robots`` -- or
any test that only imports the package -- exercises NONE of these import
statements. A LeRobot release that renames or moves a symbol therefore stays
invisible until the exact code path runs, which for several paths only happens
on GPU or attached hardware.

This guard closes that gap statically. It parses every module under the
``strands_robots`` package, collects each ``from lerobot... import NAME`` and
``import lerobot...`` statement, imports the referenced LeRobot module, and
asserts the named symbol resolves -- either as a top-level attribute of the
module or as an importable submodule (``from lerobot.transport import
services_pb2`` binds a submodule, which is absent from the parent namespace
until first imported). A renamed/removed LeRobot attribute fails here, in the
fast unit suite, instead of at runtime in a deferred branch.

The scan is conservative: a LeRobot submodule that cannot be imported in the
current environment (an optional extra is absent) is skipped rather than failed,
so the guard never turns an optional-dependency gap into a red test. It only
asserts on symbols belonging to modules that successfully import.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

import strands_robots

pytest.importorskip("lerobot", reason="lerobot not installed - pip install 'strands-robots[lerobot]'")

_PACKAGE_DIR = Path(strands_robots.__file__).resolve().parent

# Symbols strands imports for FORWARD compatibility with a future lerobot:
# they exist on lerobot main but not in any wheel within the currently pinned
# range, and EVERY use is gated behind a try/except (ImportError, TypeError)
# fallback that degrades gracefully when the symbol is absent. Their absence on
# the pinned wheel is therefore expected, not a rename/removal, so the drift
# guard must not flag them. Remove an entry once the pinned lerobot range ships
# the symbol (it then resolves like any other).
#
# A new entry belongs here only when strands starts importing a symbol that
# exists on lerobot main but not yet in the pinned range. The previous entry,
# reanchor_relative_rtc_prefix, left when lerobot 0.6 shipped it.
#
# resolve_episode_indices: the episode-subset resolver that landed on lerobot
# main in 64b23178d beside DatasetConfig.exclude_episodes, absent from 0.6.1 -
# the only release inside the declared range. utils.effective_episode_count
# imports it inside a try/except ImportError and falls back to counting the
# allowlist verbatim, which is what 0.6.1's own make_train_eval_datasets does.
#
# That fallback is what the exemption rests on, so both halves of it are asserted
# rather than trusted: ``test_forward_compat_allowlist_entries_are_actually_imported``
# refuses a stale entry, and ``test_every_forward_compat_entry_is_gated_behind_a_fallback``
# refuses one whose import does not fall back. Without the second, adding a name
# here would silence the drift guard for an UNGUARDED import, which is the
# rename/removal this file exists to catch.
_FORWARD_COMPAT_SYMBOLS: frozenset[tuple[str, str]] = frozenset(
    {
        ("lerobot.datasets.utils", "resolve_episode_indices"),
    }
)

# An import failure for a symbol a module does not export raises ``ImportError``
# (``cannot import name ... from ...``), so an ``ImportError`` handler is what
# makes a forward-compat fallback real. A wider handler is not accepted here: the
# roster's promise is specifically that the symbol's ABSENCE is handled.
_IMPORT_FALLBACK_HANDLERS: frozenset[str] = frozenset({"ImportError", "ModuleNotFoundError"})


def _python_sources() -> list[Path]:
    return sorted(p for p in _PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _collect_lerobot_imports() -> list[tuple[str, str | None, Path, int]]:
    """Return ``(module, symbol, file, lineno)`` for every lerobot import.

    ``symbol`` is ``None`` for plain ``import lerobot.foo`` statements (only the
    module needs to resolve). Star imports are skipped (no single symbol to
    check).
    """
    found: list[tuple[str, str | None, Path, int]] = []
    for f in _python_sources():
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if not node.module or not node.module.startswith("lerobot"):
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    found.append((node.module, alias.name, f, node.lineno))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "lerobot" or alias.name.startswith("lerobot."):
                        found.append((alias.name, None, f, node.lineno))
    return found


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    """Child -> parent for every node in ``tree`` (``ast`` keeps no back-links)."""
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _guarded_by_an_import_fallback(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """True when ``node`` sits in the body of a ``try`` that handles an import failure.

    Walks outward so a nested statement (the shipped site imports inside an ``if``
    inside a function inside the ``try``) is credited to the enclosing handler.
    Only the ``try``'s BODY counts -- a statement in an ``except``/``finally``
    clause is not protected by that same handler.
    """
    current = node
    while current in parents:
        parent = parents[current]
        if isinstance(parent, ast.Try) and current in parent.body:
            for handler in parent.handlers:
                caught = handler.type
                if isinstance(caught, ast.Name):
                    names = {caught.id}
                elif isinstance(caught, ast.Tuple):
                    names = {e.id for e in caught.elts if isinstance(e, ast.Name)}
                else:
                    names = set()
                if names & _IMPORT_FALLBACK_HANDLERS:
                    return True
        current = parent
    return False


def _forward_compat_import_sites() -> dict[tuple[str, str], list[tuple[Path, int, bool]]]:
    """Where each rostered symbol is imported, and whether a fallback guards it."""
    sites: dict[tuple[str, str], list[tuple[Path, int, bool]]] = {key: [] for key in _FORWARD_COMPAT_SYMBOLS}
    for f in _python_sources():
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        parents = _parent_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            for alias in node.names:
                key = (node.module, alias.name)
                if key in sites:
                    sites[key].append((f, node.lineno, _guarded_by_an_import_fallback(node, parents)))
    return sites


def _resolves_in(module: object, module_name: str, symbol: str) -> bool:
    """Return True if ``from module_name import symbol`` resolves.

    ``from pkg import name`` succeeds in two distinct cases: ``name`` is a
    top-level attribute of ``pkg`` (a class/function/constant), OR ``name`` is a
    submodule of ``pkg``. A not-yet-imported submodule is absent from the parent
    package namespace -- ``hasattr(pkg, name)`` is False until something imports
    it -- so an attribute-only check both false-positives on valid submodule
    imports (e.g. ``from lerobot.transport import services_pb2``) and makes the
    result order-dependent: it passes only once an earlier test happens to
    import the submodule and bind it on the parent. Treat the symbol as resolved
    when it names an importable submodule, using ``find_spec`` so the check
    stays side-effect-free (it never actually imports the submodule).
    """
    if hasattr(module, symbol):
        return True
    try:
        return importlib.util.find_spec(f"{module_name}.{symbol}") is not None
    except (ImportError, AttributeError, ValueError):
        return False


# The scan below walks the same few hundred small ``strands_robots`` .py files
# with ``ast.parse`` + ``Path.read_text`` and completes in well under a second.
# Its only way to exceed the global ``--timeout=120`` budget is a transient
# runner I/O stall on ``read_text`` - an environmental hiccup, not an
# algorithmic hang. With the suite running fail-fast (``-x``), one such stall
# aborts the entire job and red-flags otherwise-green PRs. Disable the per-test
# timeout here (``timeout(0)``) so this deterministic meta-guard is never
# governed by the wall-clock budget; the strict 120s budget still protects
# every other test - including the importing scan below - from genuine hangs.
@pytest.mark.timeout(0)
def test_scan_found_lerobot_imports() -> None:
    """Meta-guard: the scan must actually find lerobot imports.

    If a refactor moves the imports or the AST walk regresses, this catches the
    guard silently becoming a no-op.
    """
    imports = _collect_lerobot_imports()
    assert len(imports) > 20, f"expected many lerobot imports, found {len(imports)}"


def test_imported_lerobot_symbols_resolve() -> None:
    """Every imported lerobot symbol must exist in the installed lerobot.

    Modules that cannot be imported in this environment (missing optional
    extra) are skipped; only symbols from importable modules are asserted.
    """
    drift: list[str] = []
    module_cache: dict[str, object | None] = {}

    for module, symbol, f, lineno in _collect_lerobot_imports():
        if symbol is not None and (module, symbol) in _FORWARD_COMPAT_SYMBOLS:
            continue  # intentionally forward-compat, gated behind a fallback
        if module not in module_cache:
            try:
                module_cache[module] = importlib.import_module(module)
            except Exception:  # noqa: BLE001 - optional extra absent: skip, do not fail
                module_cache[module] = None
        mod = module_cache[module]
        if mod is None:
            continue
        if symbol is None:
            continue  # plain `import lerobot.x` already validated by import_module
        if not _resolves_in(mod, module, symbol):
            rel = f.relative_to(_PACKAGE_DIR.parent)
            drift.append(f"  {rel}:{lineno}: `from {module} import {symbol}` - symbol not found")

    assert not drift, (
        "lerobot API drift: strands_robots imports symbols that no longer exist in the "
        f"installed lerobot ({len(drift)} broken):\n" + "\n".join(drift)
    )


def test_every_forward_compat_entry_is_gated_behind_a_fallback() -> None:
    """Every import of a rostered symbol must degrade when the symbol is absent.

    An entry in ``_FORWARD_COMPAT_SYMBOLS`` exempts a symbol from the resolve
    check on the stated grounds that its absence is already handled at every use.
    ``test_forward_compat_allowlist_entries_are_actually_imported`` keeps the
    roster from going stale, but nothing checked the fallback the exemption rests
    on -- so a name added here also made an UNGUARDED import of a missing symbol
    invisible, which is the rename/removal this file exists to catch. That import
    resolves on a newer lerobot and raises ``ImportError`` on the pinned one, in a
    deferred branch, with the guard reporting no drift.

    An import site counts as guarded only when it sits in the BODY of a ``try``
    whose handler catches an import failure; the site of the fallback is what the
    caller reaches when the symbol is gone.
    """
    unguarded: list[str] = []

    for (module, symbol), sites in _forward_compat_import_sites().items():
        for f, lineno, guarded in sites:
            if not guarded:
                rel = f.relative_to(_PACKAGE_DIR.parent)
                unguarded.append(f"  {rel}:{lineno}: `from {module} import {symbol}` - no ImportError fallback")

    assert not unguarded, (
        "_FORWARD_COMPAT_SYMBOLS exempts a symbol whose import is not guarded, so the drift guard "
        f"is silenced for an import that will raise ({len(unguarded)}):\n" + "\n".join(unguarded)
    )


def test_lerobot_scan_disables_global_timeout() -> None:
    """Guard the flake fix: the import scan must opt out of the global per-test timeout.

    ``test_scan_found_lerobot_imports`` is a deterministic, sub-second AST/IO
    sweep whose only way to exceed the global ``--timeout=120`` budget is a
    transient runner I/O stall on ``Path.read_text``. Under fail-fast (``-x``),
    one such stall aborts the whole suite. We pin ``@pytest.mark.timeout(0)`` so
    the wall-clock budget cannot govern it. This regression asserts that opt-out
    stays in place; it fails if the marker is dropped or set to a finite budget.
    """
    pytestmark = getattr(test_scan_found_lerobot_imports, "pytestmark", [])
    marks = [m for m in pytestmark if m.name == "timeout"]
    assert marks, "expected a @pytest.mark.timeout marker on the lerobot import scan"
    assert marks[0].args == (0,), f"expected timeout(0) to disable the budget, got {marks[0].args!r}"


def test_forward_compat_allowlist_entries_are_actually_imported() -> None:
    """Keep the forward-compat allowlist honest - no stale entries.

    Every ``_FORWARD_COMPAT_SYMBOLS`` pair must correspond to a real
    ``from lerobot... import NAME`` the scanner finds in ``strands_robots``.
    A drifted/removed import would otherwise leave a dead allowlist entry that
    silently widens the drift guard's blind spot; this fails so the entry is
    removed alongside the import.
    """
    found_pairs = {(module, symbol) for module, symbol, _f, _lineno in _collect_lerobot_imports() if symbol}
    stale = sorted(pair for pair in _FORWARD_COMPAT_SYMBOLS if pair not in found_pairs)
    assert not stale, (
        "stale forward-compat allowlist entries (no matching lerobot import in "
        f"strands_robots): {stale} - remove them from _FORWARD_COMPAT_SYMBOLS"
    )


def test_submodule_imports_are_not_flagged_as_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``from pkg import submodule`` import must not be reported as API drift.

    ``strands_robots`` imports ``services_pb2`` / ``services_pb2_grpc`` as
    submodules of ``lerobot.transport``. A submodule is absent from its parent
    package namespace until something imports it, so an attribute-only drift
    check (``hasattr``) both false-positives on these valid imports and makes
    the guard order-dependent -- green only after an earlier test happens to
    import the submodule and bind it on the parent, red when run in isolation.

    This forces the cold state (submodule not yet imported, attribute unbound)
    so the assertion holds regardless of collection order: the drift guard must
    resolve the submodule via ``find_spec`` and report no drift. On the
    attribute-only implementation this fails; with submodule-aware resolution it
    passes deterministically.
    """
    transport = importlib.import_module("lerobot.transport")
    submods = ("services_pb2", "services_pb2_grpc")
    imports = _collect_lerobot_imports()
    present = {sym for mod, sym, _f, _ln in imports if mod == "lerobot.transport" and sym in submods}
    if not present:
        pytest.skip("strands_robots no longer imports lerobot.transport submodules")

    import sys

    # Force the cold state: drop cached submodules and unbind them from the
    # parent so hasattr(transport, name) is False, exactly as when the guard
    # runs before anything imports them.
    for name in present:
        sys.modules.pop(f"lerobot.transport.{name}", None)
        if name in transport.__dict__:
            monkeypatch.delitem(transport.__dict__, name, raising=False)
        assert not hasattr(transport, name), f"failed to force cold state for {name!r}"
        assert _resolves_in(transport, "lerobot.transport", name), (
            f"submodule import `from lerobot.transport import {name}` was flagged as "
            "drift even though it is an importable submodule"
        )

    # The full guard must also be clean: no submodule import is reported.
    test_imported_lerobot_symbols_resolve()
