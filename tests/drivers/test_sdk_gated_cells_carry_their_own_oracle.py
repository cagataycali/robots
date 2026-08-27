"""Guard: a cell gated on the vendor SDK has to say why it needs the real one.

``unitree-sdk2`` is not a declared dependency of this project - it is not on
PyPI under a name ``call-test-lint`` can install, and the hatch test env is
built from ``features = ["all"]``, which does not reach it.  So a cell behind
``skipif(not _HAS_SDK)`` is graded by nothing on the runner that decides
whether a pull request may merge.  It runs here, on a box where the SDK
happens to be installed, and nowhere else.

That is only acceptable when the cell needs the *real* SDK: one that recomputes
an SDK-computed value as an independent oracle cannot use a stub, because a stub
would compare a constant against itself.  A cell that needs a ``LowCmd_``-shaped
object to write into needs a *shape*, not the SDK, and can take one from the
stub :mod:`tests.drivers.test_g1_control_loop` installs - which is what puts it
in front of CI.

The distinction is decidable from the cell: an independent-oracle cell has to
import the SDK to obtain the reference value, so the rule below is that a gated
node imports the SDK inside itself.  A gated node that imports nothing is
asserting a contract CI never reads, and the remedy is the stub rather than the
marker.

Scope: the rule reads gating, not skipping.  A cell that skips on an optional
dependency this project *does* declare is outside it, because CI installs that
dependency and grades the cell.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_SDK_ROOT = "unitree_sdk2py"
_TESTS = pathlib.Path(__file__).resolve().parent.parent
_GATE_NAME = "_HAS_SDK"

# The population is small and lives in one directory today, but the scan walks
# every suite: the next gated cell is graded the hour it lands, wherever it
# lands.
_MINIMUM_GATED_NODES = 2


def _imports_the_sdk(node: ast.AST) -> bool:
    """Whether ``node``'s own subtree imports the vendor SDK.

    An independent-oracle cell reaches for the SDK to obtain the reference
    value it grades against, so the import is the evidence that the marker is
    load-bearing rather than inherited.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.ImportFrom) and (child.module or "").startswith(_SDK_ROOT):
            return True
        if isinstance(child, ast.Import) and any(alias.name.startswith(_SDK_ROOT) for alias in child.names):
            return True
    return False


def _gated_nodes(tree: ast.AST, path: str) -> list[tuple[str, int, bool]]:
    """Every node gated on the SDK probe, with whether it imports the SDK."""
    found: list[tuple[str, int, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        gated = any(_GATE_NAME in ast.unparse(decorator) for decorator in node.decorator_list)
        if gated:
            found.append((f"{path}::{node.name}", node.lineno, _imports_the_sdk(node)))
    return found


def _scan_the_suites() -> list[tuple[str, int, bool]]:
    """Walk every test module and collect the gated nodes."""
    rows: list[tuple[str, int, bool]] = []
    for module in sorted(_TESTS.rglob("test_*.py")):
        source = module.read_text(encoding="utf-8")
        if _GATE_NAME not in source:
            continue
        rows.extend(_gated_nodes(ast.parse(source), str(module.relative_to(_TESTS.parent))))
    return rows


class TestEveryGatedCellNeedsTheRealSdk:
    """A cell CI cannot run has to be one a stub could not serve."""

    def test_no_gated_node_is_missing_its_oracle(self) -> None:
        """Each gated node imports the SDK, so its marker is load-bearing."""
        inherited = [f"{name} (line {line})" for name, line, imports in _scan_the_suites() if not imports]
        assert not inherited, (
            "these nodes are gated on the vendor SDK but import nothing from it, so the "
            "contract they assert is graded by nothing in CI: "
            f"{inherited}.  Either drive them through the stub "
            "tests.drivers.test_g1_control_loop installs and drop the marker, or import "
            "the SDK value the cell grades against so the marker is load-bearing."
        )

    def test_the_scan_finds_the_gated_nodes_it_grades(self) -> None:
        """Non-vacuity: an empty scan would pass the rule above for free."""
        rows = _scan_the_suites()
        assert len(rows) >= _MINIMUM_GATED_NODES, (
            f"expected at least {_MINIMUM_GATED_NODES} SDK-gated nodes to grade, found {rows}"
        )


class TestTheRuleReadsTheCellRatherThanItsName:
    """Constructed exemplars, because the shipped suites satisfy the rule.

    With no violator left in the tree the scan cannot exercise its own failing
    branch, so the two outcomes are graded on source built here.
    """

    _WITH_ORACLE = """
@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_recomputes_the_vendor_value() -> None:
    from unitree_sdk2py.utils.crc import CRC

    assert CRC().Crc(object()) is not None
"""

    _WITHOUT_ORACLE = """
@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_reads_back_a_field_it_wrote() -> None:
    cmd, err = _build_lowcmd_from_action({"left_knee": 0.2}, mode_machine=9)
    assert cmd.motor_cmd[3].kp == 100.0
"""

    _UNGATED = """
def test_needs_no_sdk() -> None:
    from unitree_sdk2py.utils.crc import CRC

    assert CRC is not None
"""

    @pytest.mark.parametrize(
        ("label", "source", "expected"),
        [
            pytest.param("gated-and-imports-the-sdk", _WITH_ORACLE, [True], id="gated-with-oracle"),
            pytest.param("gated-and-imports-nothing", _WITHOUT_ORACLE, [False], id="gated-without-oracle"),
            pytest.param("not-gated-at-all", _UNGATED, [], id="not-gated"),
        ],
    )
    def test_the_rule_classifies_a_constructed_cell(self, label: str, source: str, expected: list[bool]) -> None:
        """The scan reads the cell's imports, not its name or its reason."""
        rows = _gated_nodes(ast.parse(source), "exemplar.py")
        assert [imports for _name, _line, imports in rows] == expected, label

    def test_both_outcomes_are_reachable(self) -> None:
        """Meta: the exemplars really do land on both sides of the rule."""
        outcomes: set[bool] = set()
        for source in (self._WITH_ORACLE, self._WITHOUT_ORACLE):
            outcomes.update(imports for _n, _l, imports in _gated_nodes(ast.parse(source), "exemplar.py"))
        assert outcomes == {True, False}
