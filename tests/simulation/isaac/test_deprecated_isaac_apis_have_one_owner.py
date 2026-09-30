"""Deprecated Isaac Sim APIs are imported in one place, and legacy ones nowhere.

Isaac Sim 6.1.0 ships ``isaacsim.core.api``, ``isaacsim.core.prims``,
``isaacsim.core.utils`` and ``isaacsim.sensors.camera`` under
``isaacsim/extsDeprecated/``. The backend imported them at ~25 scattered call
sites, each followed by an ``except ImportError:`` fallback to the 4.x
``omni.isaac.*`` names removed in Isaac Sim 5.0 - dead code for a backend whose
floor is 6.0. The imports now go through ``_deprecated_api``; this pins that no
other module reaches a deprecated module directly and that no ``omni.isaac``
import survives, so a future migration has one place to change.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import types

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac import _deprecated_api  # noqa: E402
from tests._package_ast import parse_file

_PKG = pathlib.Path(_deprecated_api.__file__).parent


def _imports(path: pathlib.Path) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(parse_file(path)):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
    return found


def _backend_files() -> list[pathlib.Path]:
    return sorted(p for p in _PKG.rglob("*.py") if p.name != "_deprecated_api.py")


def test_no_module_imports_a_deprecated_isaac_api_directly() -> None:
    offenders = [
        f"{p.relative_to(_PKG)}:{line} {mod}"
        for p in _backend_files()
        for line, mod in _imports(p)
        if any(mod == d or mod.startswith(d + ".") for d in _deprecated_api.DEPRECATED_MODULES)
    ]
    assert not offenders, "import these through _deprecated_api instead:\n" + "\n".join(offenders)


def test_no_legacy_omni_isaac_import_survives() -> None:
    offenders = [
        f"{p.relative_to(_PKG)}:{line} {mod}"
        for p in [*_backend_files(), _PKG / "_deprecated_api.py"]
        for line, mod in _imports(p)
        if mod == "omni.isaac" or mod.startswith("omni.isaac.")
    ]
    assert not offenders, "\n".join(offenders)


def test_every_source_is_a_deprecated_module() -> None:
    for name, module in _deprecated_api._SOURCES.items():
        assert any(module == d or module.startswith(d + ".") for d in _deprecated_api.DEPRECATED_MODULES), name


def test_a_name_resolves_through_a_faked_module(monkeypatch) -> None:
    fake = types.ModuleType("isaacsim.core.api")
    fake.World = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaacsim.core.api", fake)

    from strands_robots.simulation.isaac._deprecated_api import World

    assert World is object


def test_an_absent_isaac_raises_import_error() -> None:
    with pytest.raises(ImportError):
        from strands_robots.simulation.isaac._deprecated_api import World  # noqa: F401


def test_an_unknown_name_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError):
        _deprecated_api.NotAnIsaacName  # noqa: B018


def test_articulation_prefers_the_modules_6x_resolves(monkeypatch) -> None:
    prims = types.ModuleType("isaacsim.core.prims")
    prims.SingleArticulation = "single"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaacsim.core.api.articulations", types.ModuleType("x"))
    monkeypatch.setitem(sys.modules, "isaacsim.core.prims", prims)

    assert _deprecated_api.articulation_cls() == "single"
