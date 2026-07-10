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
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# First-party modules that import an untyped third-party package (pyarrow).
_PYARROW_IMPORTERS = (
    "strands_robots/dataset_recorder.py",
    "strands_robots/verify_dataset.py",
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
        f"pyarrow imports (dataset_recorder / verify_dataset) need an "
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


def _mypy_importable() -> bool:
    import importlib.util

    return importlib.util.find_spec("mypy") is not None
