"""The package's runtime import graph is an acyclic, downward-only DAG.

Two properties of ``strands_robots``, both read from the source by
``scripts/check_import_layers.py`` and both pinned here:

* the **runtime** module-scope import graph has no cycle - a cycle there is what
  makes an import order load-bearing and an interpreter deadlock possible;
* every runtime edge that points at a higher layer is written down in
  ``KNOWN_UPWARD_EDGES``. The pin is an equality, so an inversion added to the
  package fails until someone declares it, and an inversion removed from the
  package fails until someone deletes its line. The roster is a ratchet, not a
  suppression list.

Typing-only imports (``if TYPE_CHECKING:``) and late imports (inside a function)
are reported by the script and deliberately excluded from the acyclicity
requirement: they cost nothing at import time and are the two sanctioned ways to
break a cycle, which is exactly what ``simulation.base`` and
``simulation.policy_runner`` use them for.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "check_import_layers.py"
_PACKAGE_ROOT = _REPO_ROOT / "strands_robots"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_import_layers", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The script is annotated and mypy-clean on its own
# (mypy scripts/check_import_layers.py); it is reached through importlib here
# because scripts/ is not an importable package, so its members are module
# attributes at runtime rather than names mypy can resolve to types.
mod = _load()


@pytest.fixture(scope="module")
def graph() -> Any:
    """The real package graph, built once for every pin that reads the tree."""
    return mod.build_graph(_PACKAGE_ROOT)


class TestTheGraphItIsBuiltFrom:
    """A grader that parses nothing passes every pin below, so measure it first."""

    def test_the_package_is_read_whole(self, graph: Any) -> None:
        assert len(graph.modules) > 250, f"only {len(graph.modules)} modules parsed"
        assert graph.edge_count("runtime") > 500, f"only {graph.edge_count('runtime')} runtime edges"
        assert graph.edge_count("typing_only") > 50
        assert graph.edge_count("late") > 100

    def test_the_layers_are_the_declared_order(self) -> None:
        assert mod.LAYER_NAMES == (
            "core",
            "registry",
            "drivers|mesh",
            "sim|policies",
            "app",
            "tools",
            "dashboard",
        )

    def test_every_top_level_member_has_a_layer(self, graph: Any) -> None:
        assert mod.unassigned_members(graph) == ()

    def test_a_member_missing_from_the_map_is_named(self, graph: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without this the completeness pin above passes by answering nothing."""
        monkeypatch.setattr(mod, "LAYER_OF_MEMBER", {k: v for k, v in mod.LAYER_OF_MEMBER.items() if k != "drivers"})
        assert mod.unassigned_members(graph) == ("drivers",)


class TestTheParserBehindIt:
    """The three import kinds, and submodule-versus-attribute, on a fake tree.

    Built on disk rather than asserted against ``strands_robots`` so each cell
    isolates one rule: a resolver that answered the parent package for every
    ``from X import y`` would make a leaf's read of a core module look like a
    cycle through ``X/__init__.py``, and nothing in the package's own graph
    distinguishes that from the truth.
    """

    @staticmethod
    def _write(root: Path) -> None:
        (root / "leaf").mkdir(parents=True)
        (root / "__init__.py").write_text("ROOT_ATTR = 1\n", encoding="utf-8")
        (root / "core.py").write_text("VALUE = 2\n", encoding="utf-8")
        (root / "leaf" / "__init__.py").write_text("", encoding="utf-8")
        # Reads a submodule and nothing else, so an edge to the parent package
        # would be visible rather than masked by a second import that earns one.
        (root / "leaf" / "reader.py").write_text(
            "from typing import TYPE_CHECKING\n\n"
            f"from {root.name} import core\n"
            "\n"
            "if TYPE_CHECKING:\n"
            f"    from {root.name}.leaf import writer\n"
            "\n"
            "\n"
            "def late() -> None:\n"
            f"    from {root.name}.leaf import writer as _w\n",
            encoding="utf-8",
        )
        # Reads an attribute of the package root, which is the only way to earn
        # an edge to the package itself.
        (root / "leaf" / "attrs.py").write_text(f"from {root.name} import ROOT_ATTR\n", encoding="utf-8")
        (root / "leaf" / "writer.py").write_text("", encoding="utf-8")

    @pytest.fixture
    def fake(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
        root = tmp_path / "fakepkg"
        self._write(root)
        monkeypatch.setattr(mod, "PACKAGE", root.name)
        return mod.build_graph(root)

    @pytest.mark.parametrize(
        ("module", "kind", "expected"),
        [
            ("fakepkg.leaf.reader", "runtime", {"fakepkg.core"}),
            ("fakepkg.leaf.reader", "typing_only", {"fakepkg.leaf.writer"}),
            ("fakepkg.leaf.reader", "late", {"fakepkg.leaf.writer"}),
            ("fakepkg.leaf.attrs", "runtime", {"fakepkg"}),
            ("fakepkg.leaf.attrs", "typing_only", set()),
            ("fakepkg.leaf.attrs", "late", set()),
        ],
    )
    def test_each_import_kind_lands_in_its_own_graph(
        self, fake: Any, module: str, kind: str, expected: set[str]
    ) -> None:
        assert set(getattr(fake, kind).get(module, frozenset())) == expected

    def test_a_two_module_cycle_is_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root = tmp_path / "cyclic"
        root.mkdir()
        (root / "__init__.py").write_text("", encoding="utf-8")
        (root / "a.py").write_text("from cyclic import b\n", encoding="utf-8")
        (root / "b.py").write_text("from cyclic import a\n", encoding="utf-8")
        monkeypatch.setattr(mod, "PACKAGE", "cyclic")
        cyclic = mod.build_graph(root)
        assert mod.cycles(cyclic.runtime, frozenset(cyclic.modules)) == [["cyclic.a", "cyclic.b"]]


class TestTheContract:
    """The two properties the roadmap's layered-DAG milestone is measured by."""

    def test_the_runtime_graph_has_no_cycle(self, graph: Any) -> None:
        found = mod.cycles(graph.runtime, frozenset(graph.modules))
        assert found == [], f"runtime import cycles: {found}"

    def test_the_upward_edges_are_exactly_the_declared_ones(self, graph: Any) -> None:
        found = set(mod.upward_edges(graph))
        declared = set(mod.KNOWN_UPWARD_EDGES)
        assert sorted(found - declared) == [], "undeclared inversion; fix it or declare it"
        assert sorted(declared - found) == [], "declared inversion is gone; delete its line"

    def test_no_driver_imports_a_policy(self, graph: Any) -> None:
        """The cut this contract was first used to make, named on its own.

        A driver that reads a constant out of a policy takes its wire roster from
        that policy's tensor ordering; the roster belongs to the robot.
        """
        drivers_mesh = mod.LAYER_NAMES.index("drivers|mesh")
        sim_policies = mod.LAYER_NAMES.index("sim|policies")
        offenders = [
            edge
            for edge in mod.upward_edges(graph)
            if mod.layer_of(edge[0]) == drivers_mesh and mod.layer_of(edge[1]) == sim_policies
        ]
        assert offenders == []

    def test_the_script_reports_the_tree_as_conforming(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert mod.main([]) == 0
        assert "OK: no runtime cycle, no undeclared inversion" in capsys.readouterr().out
