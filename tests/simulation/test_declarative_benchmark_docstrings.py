"""The declarative-benchmark eval surface must document every public member.

:class:`~strands_robots.simulation.benchmark.BenchmarkProtocol` defines the
contract the ``PolicyRunner`` eval loop drives (``supported_robots`` /
``default_robot`` / ``instruction`` / ``on_episode_start`` / ``on_step`` /
``augment_observation`` / ``is_success`` / ``is_failure``) with rich
docstrings. :class:`~strands_robots.simulation.benchmark_spec.DeclarativeBenchmark`
is the YAML/JSON-authored implementation agents reach through the
``register_benchmark_from_file`` tool - it overrides those members, and each
override needs its own docstring rather than silently leaning on the inherited
protocol text: an accessor like ``supported_robots`` returns a defensive copy
and ``is_success`` states its side-effect-free contract, both worth stating on
the concrete class a spec author actually reads.

This guard walks the two modules by AST (no import, so it never needs an
optional sim backend installed) and fails if any public method or property of
a benchmark class defines no docstring.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.simulation as simulation_pkg

_PACKAGE_DIR = Path(simulation_pkg.__file__).parent

# The declarative-benchmark surface: the protocol ABC + its structured helpers
# (benchmark.py) and the DSL-backed implementation (benchmark_spec.py). Both are
# dependency-free pure-Python modules, so the AST walk needs no optional extra.
_MODULES = ("benchmark.py", "benchmark_spec.py")

_EXPECTED_CLASSES = {
    "benchmark.py::StepInfo",
    "benchmark.py::BenchmarkProtocol",
    "benchmark.py::BenchmarkCompatibilityError",
    "benchmark_spec.py::DeclarativeBenchmark",
}


def _public_members_without_docstring(class_node: ast.ClassDef) -> list[str]:
    """Return names of public methods/properties in the class body lacking a docstring.

    Dunder methods (``__init__`` and friends) are out of scope - their contract
    is documented on the class itself.
    """
    offenders: list[str] = []
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name.startswith("_"):
            continue
        if ast.get_docstring(node) is None:
            offenders.append(node.name)
    return offenders


def _benchmark_classes() -> dict[str, ast.ClassDef]:
    """Map ``module.py::ClassName`` -> ClassDef for every public class in the modules."""
    classes: dict[str, ast.ClassDef] = {}
    for module in _MODULES:
        source_file = _PACKAGE_DIR / module
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                classes[f"{module}::{node.name}"] = node
    return classes


def test_modules_define_expected_benchmark_classes() -> None:
    """Guard: the scan actually found the benchmark classes it protects."""
    assert set(_benchmark_classes()) == _EXPECTED_CLASSES, set(_benchmark_classes())


def test_benchmark_public_members_have_docstrings() -> None:
    offenders = {
        qualname: missing
        for qualname, node in _benchmark_classes().items()
        if (missing := _public_members_without_docstring(node))
    }
    assert not offenders, (
        "Every public method/property of a benchmark class (BenchmarkProtocol "
        "and the DeclarativeBenchmark DSL implementation) must have a docstring "
        "describing its behavior. Undocumented members: " + repr(offenders)
    )
