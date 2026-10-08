#!/usr/bin/env python3
"""Run the unit suite, scoped on a pull request to the tests its change reaches.

``hatch run test`` runs this script. On a push, a schedule or a developer's
machine it is ``python -m pytest`` with the arguments it was given. On a
``pull_request`` event, when the caller named no test path, it runs:

- every test file that names a changed module. A file names a module when it
  imports it at any depth, spells it in a string
  (``monkeypatch.setattr("strands_robots.x.y", ...)``) or reaches it by an
  attribute chain off an imported module; a name a package re-exports, eagerly
  or through a lazy ``__getattr__`` table, counts as naming the module that
  defines it. A test helper under ``tests/`` passes on what it names to the
  tests that import it, and a module whose class inherits from a changed
  module counts as changed too: its callers run the inherited methods;
- the whole-tree graders :mod:`check_whole_tree_graders` derives, whose input
  is the rest of the repository rather than the file under change;
- for a changed ``conftest.py`` below ``tests/``, every test file under it;
  for a change under ``docs/``, ``examples/``, ``changelog.d/``,
  ``tests_integ/``, ``README.md`` or ``mkdocs.yml``, every test file that
  spells a path into it.

A test that reaches a changed module only through another package module is
not selected. Anything else that changed - ``pyproject.toml``, ``uv.lock``,
``scripts/``, ``.github/``, ``tests/conftest.py`` - runs the whole suite, and
so does every push to ``main``, which is where a test the scoping left out is
caught. A scoped run passes ``--no-cov``: the coverage floor is a floor over
the whole suite.

``python scripts/select_tests.py --list [--base REF]`` prints what a pull
request from the working tree would run against ``REF`` (default
``origin/main``).
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

from check_import_layers import PACKAGE, _import_targets  # noqa: E402
from check_whole_tree_graders import roster  # noqa: E402

ROOT = _SCRIPTS.parent

#: Top-level paths whose change selects the test files that spell a path into them.
NAMED_AREAS = ("docs", "examples", "changelog.d", "tests_integ", "README.md", "mkdocs.yml")

_DOTTED = re.compile(rf"\b(?:{PACKAGE}|tests)(?:\.\w+)+")


def changed_paths(base: str, root: Path = ROOT) -> list[str]:
    """Return the paths that differ between the working tree and its merge base with ``base``."""
    merge_base = subprocess.run(
        ["git", "merge-base", "HEAD", base], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", merge_base], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    return sorted(set(diff.split()))


def _dotted(path: Path) -> str:
    """The dotted module name of a repository-relative ``.py`` path."""
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_test_file(path: Path) -> bool:
    return path.parts[0] == "tests" and path.name.startswith("test_") and path.suffix == ".py"


def _strings(tree: ast.AST) -> Iterable[str]:
    return (node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str))


class Graph:
    """Which package and test modules each Python file under ``root`` names, read from source."""

    def __init__(self, root: Path = ROOT) -> None:
        paths = sorted(
            path.relative_to(root)
            for area in (PACKAGE, "tests")
            for path in (root / area).rglob("*.py")
            if "__pycache__" not in path.parts
        )
        self.files: dict[str, Path] = {_dotted(path): path for path in paths}
        self.trees = {name: ast.parse((root / path).read_text(encoding="utf-8")) for name, path in self.files.items()}
        self.served = {name: self._lazy_table(name) | self._exports(name) for name in self.files}
        self.bases: dict[str, set[str]] = {}
        self.edges = {name: self._references(name) for name in self.files}

    def owner(self, dotted: str) -> str | None:
        """The module a dotted reference lands on, following a package's re-exports."""
        parts = dotted.split(".")
        for end in range(len(parts), 0, -1):
            candidate = ".".join(parts[:end])
            if candidate in self.files:
                served = self.served[candidate].get(parts[end]) if end < len(parts) else None
                if served is not None and served != candidate:
                    return self.owner(".".join([served, *parts[end + 1 :]])) or served
                return candidate
        return None

    def names_area(self, name: str, area: str) -> bool:
        """Whether a module spells a path into the top-level ``area`` in a string."""
        return any(s == area or s.startswith(f"{area}/") or s.endswith(f"/{area}") for s in _strings(self.trees[name]))

    def _exports(self, name: str) -> dict[str, str]:
        """``{attribute: dotted}`` for the names a package ``__init__`` imports at module scope."""
        if self.files[name].name != "__init__.py":
            return {}
        table: dict[str, str] = {}
        for node in self.trees[name].body:
            if isinstance(node, ast.ImportFrom):
                for alias, target in zip(node.names, _import_targets(name, node, is_package=True), strict=False):
                    table[alias.asname or alias.name] = target
        return table

    def _lazy_table(self, name: str) -> dict[str, str]:
        """``{attribute: module}`` for the names a module serves through a module-level ``__getattr__``."""
        tree = self.trees[name]
        if not any(isinstance(node, ast.FunctionDef) and node.name == "__getattr__" for node in tree.body):
            return {}
        package = name if self.files[name].name == "__init__.py" else name.rpartition(".")[0]
        table: dict[str, str] = {}
        for node in ast.walk(tree):
            for key, value in zip(node.keys, node.values, strict=True) if isinstance(node, ast.Dict) else ():
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    for text in _strings(value):
                        target = (f"{package}{text}" if text.startswith(".") else text).split(":")[0]
                        if target in self.files:
                            table[key.value] = target
        return table

    def _references(self, name: str) -> set[str]:
        """Every module ``name`` imports, names in a string, or reaches by an attribute chain."""
        tree = self.trees[name]
        named: set[str] = set()
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    named.add(alias.name)
                    aliases[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and (node.module or "").split(".")[0] == "tests":
                    targets = [f"{node.module}.{alias.name}" for alias in node.names]
                else:
                    targets = _import_targets(name, node, is_package=self.files[name].name == "__init__.py")
                named.update(targets)
                aliases.update(
                    (alias.asname or alias.name, target) for alias, target in zip(node.names, targets, strict=False)
                )
        if self.files[name].parts[0] == "tests":
            # A test names a module in a string to patch or import it; the
            # package names one in a message, which imports nothing.
            named.update(match for text in _strings(tree) for match in _DOTTED.findall(text))

        def dotted(node: ast.expr) -> str | None:
            chain: list[str] = []
            while isinstance(node, ast.Attribute):
                chain.append(node.attr)
                node = node.value
            return (
                ".".join([aliases[node.id], *reversed(chain)])
                if isinstance(node, ast.Name) and node.id in aliases
                else None
            )

        bases: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and (reference := dotted(node)):
                named.add(reference)
            elif isinstance(node, ast.ClassDef):
                bases.update(reference for base in node.bases if (reference := dotted(base)))
        self.bases[name] = {owner for owner in map(self.owner, bases) if owner is not None and owner != name}
        return {owner for owner in map(self.owner, named) if owner is not None and owner != name}

    def _carries(self, user: str, module: str) -> bool:
        """Whether ``user`` naming ``module`` makes it as changed as ``module`` for selection."""
        file = self.files[user]
        if file.name == "conftest.py":
            # Every selected test loads its conftest, so what a conftest
            # imports is exercised by whichever tests run.
            return False
        return file.parts[0] == "tests" or module in self.bases[user]

    def reaching(self, changed: Iterable[str]) -> set[str]:
        """``changed`` plus every module that names one of them and carries the change on."""
        users: dict[str, set[str]] = defaultdict(set)
        for source, targets in self.edges.items():
            for target in targets:
                users[target].add(source)
        reached = set(changed)
        frontier = list(reached)
        while frontier:
            module = frontier.pop()
            for user in users[module] - reached:
                if self._carries(user, module):
                    reached.add(user)
                    if not _is_test_file(self.files[user]):
                        frontier.append(user)
        return reached


def select(changed: Iterable[str], root: Path = ROOT, graph: Graph | None = None) -> list[str] | None:
    """Return the test files a change reaches, or ``None`` when it runs the whole suite."""
    paths = [Path(rel) for rel in changed]
    if any(
        path.parts[0] not in (PACKAGE, "tests", *NAMED_AREAS) or path == Path("tests/conftest.py") for path in paths
    ):
        return None
    graph = graph or Graph(root)
    tests = {str(path) for path in graph.files.values() if _is_test_file(path)}
    selected = set(roster(root))
    seeds: set[str] = set()
    areas: set[str] = set()
    for path in paths:
        if path.parts[0] not in (PACKAGE, "tests"):
            areas.add(path.parts[0])
        elif path.name == "conftest.py":
            selected.update(test for test in tests if Path(test).is_relative_to(path.parent))
        elif path.suffix == ".py":
            seeds.add(_dotted(path))
        else:
            # A data file is read by the Python beside it, or above it, and by
            # the tests that spell its name.
            folder = path.parent
            while folder != Path(".") and not any(file.parent == folder for file in graph.files.values()):
                folder = folder.parent
            seeds.update(name for name, file in graph.files.items() if file.parent == folder)
            selected.update(t for t in tests if any(path.name in s for s in _strings(graph.trees[_dotted(Path(t))])))
    reached = graph.reaching(seeds & graph.files.keys())
    # A deleted module is no node of the tree: what still spells it is selected.
    gone = seeds - graph.files.keys()
    for name, file in graph.files.items():
        if not _is_test_file(file):
            continue
        spelled = gone and any(
            ref == module or ref.startswith(f"{module}.")
            for ref in _DOTTED.findall((root / file).read_text(encoding="utf-8"))
            for module in gone
        )
        if name in reached or spelled or any(graph.names_area(name, area) for area in areas):
            selected.add(str(file))
    return sorted(selected & tests)


def pytest_command(argv: list[str], environ: Mapping[str, str], root: Path = ROOT) -> list[str]:
    """The pytest invocation for ``argv``, scoped when ``environ`` is a pull request's."""
    command = [sys.executable, "-m", "pytest", *argv]
    base_ref = environ.get("GITHUB_BASE_REF", "")
    named_paths = any(not arg.startswith("-") and (root / arg.split("::")[0]).exists() for arg in argv)
    if environ.get("GITHUB_EVENT_NAME") != "pull_request" or not base_ref or named_paths:
        return command
    try:
        changed = changed_paths(f"origin/{base_ref}", root)
    except subprocess.CalledProcessError as exc:
        print(
            f"select_tests: no diff against origin/{base_ref} ({exc.stderr.strip()}); running the whole suite",
            file=sys.stderr,
        )
        return command
    selection = select(changed, root)
    if selection is None:
        print(f"select_tests: {len(changed)} changed path(s) run the whole suite", file=sys.stderr)
        return command
    total = sum(1 for _ in (root / "tests").rglob("test_*.py"))
    print(f"select_tests: {len(changed)} changed path(s) reach {len(selection)} of {total} test files", file=sys.stderr)
    return [*command, "--no-cov", *selection]


def main(argv: list[str]) -> int:
    """Print the selection (``--list``) or run pytest over it."""
    if argv[:1] == ["--list"]:
        base = argv[argv.index("--base") + 1] if "--base" in argv else "origin/main"
        selection = select(changed_paths(base))
        print("<the whole suite>" if selection is None else "\n".join(selection))
        return 0
    command = pytest_command(argv, os.environ)
    sys.stderr.flush()
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
