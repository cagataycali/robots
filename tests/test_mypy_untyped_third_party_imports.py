"""Regression tests for third-party imports that ship no type information.

Some runtime dependencies (notably ``pyarrow``, which the LeRobot dataset
writer and verifier read parquet through) do not ship a ``py.typed`` marker or
stub package. When such a module is imported without a matching
``ignore_missing_imports`` mypy override, ``mypy strands_robots tests
tests_integ`` fails with ``import-untyped``/``import-not-found`` errors and the
whole lint gate goes red - even though nothing in first-party code changed.

``pyarrow`` 25.0.0 dropped the ``py.typed`` marker it previously shipped, which
turned every ``import pyarrow.parquet`` into a hard mypy failure across the
repo. These tests pin the override so a future dependency bump (or an
accidental removal of the override) is caught here instead of in CI.

A vendor SDK that cannot be installed at all is the same failure with a
different cause. ``unitree_sdk2py`` is the Unitree G1's DDS SDK: it is not
published on PyPI, it is installed from a source checkout on the robot, and it
therefore cannot be declared as a dependency or an extra. The G1 DDS layer
imports it inside function bodies so that a machine without it can still build
the driver and run every test against a mocked bus - but a lazy import is still
an import to mypy, so the module needs the same override, and its absence reads
as ``import-not-found`` rather than ``import-untyped``.

The two tests per package divide the work deliberately. One reads
``pyproject.toml`` and names the missing entry, which is fast and precise. The
other runs the project's own mypy over exactly the first-party importers, so a
mismatch between the override list and what those modules actually import is
caught even when each looks individually reasonable - and it is indifferent to
how the override is spelled, since a single ``pkg.*`` pattern already covers the
top-level module.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

import strands_robots
from tests._package_ast import parse_file

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# First-party modules that import an untyped third-party package (pyarrow).
_PYARROW_IMPORTERS = (
    "strands_robots/dataset_metadata.py",
    "strands_robots/dataset_recorder.py",
)

# First-party modules that import the Unitree DDS SDK (unitree_sdk2py), which is
# not on PyPI and so is absent from every CI and developer environment.
_UNITREE_IMPORTERS = (
    "strands_robots/drivers/unitree/_common.py",
    "strands_robots/drivers/unitree/_dds_engine.py",
)


def _ignore_missing_imports_modules() -> set[str]:
    """Modules covered by an ``ignore_missing_imports = true`` mypy override."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    covered: set[str] = set()
    for override in data["tool"]["mypy"].get("overrides", []):
        if override.get("ignore_missing_imports") is True:
            modules = override.get("module", [])
            if isinstance(modules, str):
                modules = [modules]
            covered.update(modules)
    return covered


def test_pyarrow_is_declared_untyped_in_mypy_overrides():
    """pyarrow ships no py.typed, so it must stay in the untyped-imports override."""
    covered = _ignore_missing_imports_modules()
    missing = {m for m in ("pyarrow", "pyarrow.*") if m not in covered}
    assert not missing, (
        f"pyarrow imports (dataset_metadata / dataset_recorder) need an "
        f"ignore_missing_imports mypy override; missing entries: {sorted(missing)}"
    )


def test_mypy_clean_on_pyarrow_importing_modules():
    """mypy on the pyarrow-importing modules must not report import-* errors.

    Reproduces the repo-wide lint break: run the project's mypy on exactly the
    first-party modules that import pyarrow and assert there is no residual
    ``import-untyped``/``import-not-found`` diagnostic mentioning pyarrow.
    """
    if shutil.which("mypy") is None and not _mypy_importable():
        pytest.skip("mypy not installed in this environment")

    result = subprocess.run(
        [sys.executable, "-m", "mypy", *_PYARROW_IMPORTERS],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    offending = [
        line
        for line in output.splitlines()
        if "pyarrow" in line and ("import-untyped" in line or "import-not-found" in line)
    ]
    assert not offending, "mypy reported unsilenced pyarrow import errors:\n" + "\n".join(offending)


def test_unitree_sdk_is_declared_untyped_in_mypy_overrides():
    """unitree_sdk2py cannot be installed, so it must stay in the untyped-imports override.

    Matched by prefix rather than against an exact pair of entries: a single
    ``unitree_sdk2py.*`` pattern already covers the top-level module, so
    demanding both spellings would fail a correct override.
    """
    covered = _ignore_missing_imports_modules()
    assert any(m == "unitree_sdk2py" or m.startswith("unitree_sdk2py.") for m in covered), (
        "the G1 DDS layer under strands_robots/tools/g1 imports unitree_sdk2py, "
        "which ships no types and is not installable; it needs an "
        "ignore_missing_imports mypy override and no entry covers it"
    )


def test_mypy_clean_on_unitree_importing_modules():
    """mypy on the G1 DDS layer must not report import-* errors.

    The SDK is imported inside function bodies so that importing the package
    never touches it, but mypy resolves the import anyway - so a missing
    override fails the lint gate on precisely the machines where the SDK could
    not have been installed, which is all of them.
    """
    if shutil.which("mypy") is None and not _mypy_importable():
        pytest.skip("mypy not installed in this environment")

    result = subprocess.run(
        [sys.executable, "-m", "mypy", *_UNITREE_IMPORTERS],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    offending = [
        line
        for line in output.splitlines()
        if "unitree_sdk2py" in line and ("import-untyped" in line or "import-not-found" in line)
    ]
    assert not offending, "mypy reported unsilenced unitree_sdk2py import errors:\n" + "\n".join(offending)


def _modules_mypy_does_not_follow() -> set[str]:
    """Top-level names of the libraries a ``follow_imports = "skip"`` override names."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    skipped: set[str] = set()
    for override in data["tool"]["mypy"].get("overrides", []):
        if override.get("follow_imports") == "skip":
            modules = override.get("module", [])
            skipped.update(m.removesuffix(".*") for m in ([modules] if isinstance(modules, str) else modules))
    return skipped


def _module_scope_imports(tree: ast.Module) -> Iterator[tuple[int, str]]:
    """Yield ``(line, module)`` for every import that runs or types at module scope.

    A function body is the one place an import neither runs at import time nor
    names a type in a signature, so only function bodies are left out: class
    bodies and ``if TYPE_CHECKING:`` blocks count.
    """
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            continue
        if isinstance(node, ast.Import):
            yield from ((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module
        stack.extend(ast.iter_child_nodes(node))


def test_no_library_mypy_skips_types_the_package():
    """The libraries mypy does not follow are ones the package only calls into from a function.

    ``transformers``, ``diffusers``, ``warp`` and ``google`` ship ``py.typed``,
    so without the override mypy analyses their whole source on every lint
    run - a third of its time - for a handful of lazy imports. Skipping a
    library makes everything imported from it ``Any``, which is harmless inside
    a function body and silently untyped anywhere else: a module-scope or
    ``TYPE_CHECKING`` import of a skipped library would put an unchecked type in
    a signature. Such an import belongs on the followed side of the override.
    """
    skipped = _modules_mypy_does_not_follow()
    assert {"transformers", "diffusers"} <= skipped, f"mypy follows the heavy typed libraries again: {sorted(skipped)}"
    planted = ast.parse("import transformers\nclass A:\n    import diffusers\ndef f():\n    import warp\n")
    assert sorted(name for _, name in _module_scope_imports(planted)) == ["diffusers", "transformers"]

    root = Path(strands_robots.__file__).resolve().parent
    typed_by_a_skip = [
        f"strands_robots/{path.relative_to(root)}:{line} imports {name}"
        for path in sorted(root.rglob("*.py"))
        for line, name in _module_scope_imports(parse_file(path))
        if name.split(".")[0] in skipped
    ]
    assert not typed_by_a_skip, "a library mypy does not follow types the package:\n" + "\n".join(typed_by_a_skip)


def _mypy_importable() -> bool:
    import importlib.util

    return importlib.util.find_spec("mypy") is not None
