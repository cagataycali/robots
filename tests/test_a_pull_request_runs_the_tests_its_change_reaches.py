"""A pull request's required check runs the tests its change reaches, and a push runs them all.

``hatch run test`` goes through ``scripts/select_tests.py``. These pins build a
small repository in ``tmp_path`` - a package, a test helper, a few test files -
and ask the selector what a change to each file runs, so every rule it applies
(an import, a lazy root name, a patched string, a helper, a subclass, a
conftest, a docs path, a file it cannot scope) is one row of one table.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "select_tests.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("select_tests", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()

_TREE = {
    "strands_robots/__init__.py": (
        "from .registry import list_robots\n"
        "_LAZY = {'Engine': ('strands_robots.engine', 'Engine')}\n"
        "def __getattr__(name):\n    return _LAZY[name]\n"
    ),
    "strands_robots/registry.py": "def list_robots():\n    return []\n",
    "strands_robots/base.py": "class Base:\n    def step(self):\n        return 1\n",
    "strands_robots/engine.py": "from strands_robots.base import Base\n\nclass Engine(Base):\n    pass\n",
    "strands_robots/tools.py": "from strands_robots.engine import Engine\n\ndef tool():\n    return Engine()\n",
    "strands_robots/data/robots.json": "{}\n",
    "strands_robots/data/loader.py": "def load():\n    return {}\n",
    "tests/__init__.py": "",
    "tests/conftest.py": "",
    "tests/_helper.py": "import strands_robots.registry\n",
    "tests/test_lazy_root_name.py": "from strands_robots import Engine\n",
    "tests/test_eager_root_name.py": "from strands_robots import list_robots\n",
    "tests/test_patched_string.py": "def test_x(monkeypatch):\n    monkeypatch.setattr('strands_robots.tools.tool', None)\n",
    "tests/test_through_a_helper.py": "from tests._helper import strands_robots\n",
    "tests/test_loader.py": "from strands_robots.data import loader\n",
    "tests/test_reads_docs.py": "PAGE = 'docs/index.md'\n",
    "tests/sim/__init__.py": "",
    "tests/sim/conftest.py": "",
    "tests/sim/test_attribute_chain.py": "import strands_robots.tools as t\n\ndef test_x():\n    t.tool()\n",
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for rel, text in _TREE.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        # A lazy root name is the module its table names; the subclass in it runs the base.
        ("strands_robots/engine.py", {"test_lazy_root_name.py"}),
        ("strands_robots/base.py", {"test_lazy_root_name.py"}),
        # An eager re-export is the module it comes from; a helper hands it on.
        ("strands_robots/registry.py", {"test_eager_root_name.py", "test_through_a_helper.py"}),
        # A dotted string and an attribute chain off an alias both name the module.
        ("strands_robots/tools.py", {"test_patched_string.py", "sim/test_attribute_chain.py"}),
        # A data file is read by the Python beside it.
        ("strands_robots/data/robots.json", {"test_loader.py"}),
        ("tests/_helper.py", {"test_through_a_helper.py"}),
        ("tests/sim/conftest.py", {"sim/test_attribute_chain.py"}),
        ("docs/index.md", {"test_reads_docs.py"}),
        ("changelog.d/0000-x.md", set()),
        # A file the selector cannot scope runs everything.
        ("pyproject.toml", None),
        ("scripts/select_tests.py", None),
        ("tests/conftest.py", None),
    ],
)
def test_a_change_runs_the_tests_that_name_it(repo: Path, changed: str, expected: set[str] | None) -> None:
    selection = mod.select([changed], repo)
    if expected is None:
        assert selection is None
    else:
        assert selection == sorted(f"tests/{name}" for name in expected)


def test_a_module_reached_only_through_another_module_is_not_selected(repo: Path) -> None:
    """``tools`` uses ``engine``, but no test of ``tools`` names ``engine``: the scope stops there."""
    assert "tests/test_patched_string.py" not in mod.select(["strands_robots/engine.py"], repo)


def test_a_deleted_module_selects_what_still_spells_it(repo: Path) -> None:
    (repo / "strands_robots/tools.py").unlink()
    assert mod.select(["strands_robots/tools.py"], repo) == [
        "tests/sim/test_attribute_chain.py",
        "tests/test_patched_string.py",
    ]


@pytest.mark.parametrize(
    ("environ", "argv", "scoped"),
    [
        ({"GITHUB_EVENT_NAME": "pull_request", "GITHUB_BASE_REF": "main"}, ["-x"], True),
        ({"GITHUB_EVENT_NAME": "push", "GITHUB_BASE_REF": ""}, ["-x"], False),
        ({}, ["-x"], False),
        # A caller that names a path asked for exactly that path.
        ({"GITHUB_EVENT_NAME": "pull_request", "GITHUB_BASE_REF": "main"}, ["tests/sim"], False),
    ],
)
def test_only_a_pull_request_is_scoped(
    repo: Path, monkeypatch: pytest.MonkeyPatch, environ: dict[str, str], argv: list[str], scoped: bool
) -> None:
    monkeypatch.setattr(mod, "changed_paths", lambda base, root: ["strands_robots/tools.py"])
    command = mod.pytest_command(argv, environ, repo)
    assert command[: 3 + len(argv)] == [sys.executable, "-m", "pytest", *argv]
    if scoped:
        assert command[3 + len(argv) :] == [
            "--no-cov",
            "tests/sim/test_attribute_chain.py",
            "tests/test_patched_string.py",
        ]
    else:
        assert len(command) == 3 + len(argv)


def test_a_base_the_checkout_lacks_runs_the_whole_suite(repo: Path) -> None:
    """``repo`` is no git checkout, so there is no diff to scope by: the whole suite runs."""
    environ = {"GITHUB_EVENT_NAME": "pull_request", "GITHUB_BASE_REF": "main"}
    assert mod.pytest_command(["-x"], environ, repo) == [sys.executable, "-m", "pytest", "-x"]
