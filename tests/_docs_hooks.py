"""The shipped ``docs/hooks`` modules, each loaded by path once per process.

A hook is loaded the way mkdocs loads it - from its file, not as a package -
because the docs venv is not the test venv. The hooks cache the scan behind
their tables on the module object (``env_vars.reads`` parses every package
module, about 12 s), so a test that executes the file a second time starts
with an empty cache and pays the scan again. Every test reaches a hook through
:func:`docs_hook`; :mod:`tests.test_docs_hooks_load_once` grades that.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

HOOKS = Path(__file__).resolve().parents[1] / "docs" / "hooks"


@cache
def docs_hook(name: str) -> ModuleType:
    """Return ``docs/hooks/<name>.py``, executed on the first call only.

    Args:
        name: The hook's file stem (``"env_vars"``, ``"providers"``, ...).

    Returns:
        The hook module, registered in ``sys.modules`` as ``docs_hooks_<name>``
        because the hooks' dataclasses resolve their module by name.
    """
    spec = importlib.util.spec_from_file_location(f"docs_hooks_{name}", HOOKS / f"{name}.py")
    assert spec is not None and spec.loader is not None, f"docs/hooks/{name}.py is missing"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
