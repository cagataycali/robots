"""Regression: a G1 module-private helper may not document itself as uncalled
while the module calls it.

``_build_zero_torque_lowcmd`` builds the soft-stop frame the control loop
publishes on its way out. Its docstring said the opposite:

    This helper is defined but not yet wired: ``G1Driver.stop`` and
    ``stop_task`` currently return refusal envelopes rather than publishing
    a frame, and no other call site exists.

Every clause of that was false once the 500 Hz loop landed. ``_ControlLoop``
calls the helper from its ``finally``, and ``G1Driver.stop_task``'s own
docstring - 370 lines above in the same module - already said so ("the loop
publishes :func:`_build_zero_torque_lowcmd` on the way out"). Two docstrings in
one module directly contradicting each other is worse than either being vague:
a reader who finds the helper first is told the shutdown path does not exist,
and the natural conclusion is that a soft stop still has to be built.

The scan is deliberately scoped to this one module and to its *module-private*
functions, because that is the scope in which matching a call by name is sound:
a leading-underscore module-level function is called by its bare name inside its
own module. The same rule package-wide is not sound and is not shipped - a
public method name such as ``send_action`` is defined on several unrelated
classes, so counting calls by name would attribute every ``robot.send_action``
in the tree to whichever driver happened to declare it, and a driver that
legitimately documents its own bus as unwired would be reported as drifting.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots.drivers.g1 as g1_module

# Prose that claims the function the reader is looking at has no caller. Each
# alternative is a claim about *wiring*, which the module's own call graph
# either bears out or contradicts.
_UNCALLED_CLAIM = re.compile(r"not yet wired|no other call site|no call sites?\b", re.IGNORECASE)

_G1_SOURCE = Path(g1_module.__file__).resolve()


def _module_tree() -> ast.Module:
    return ast.parse(_G1_SOURCE.read_text(encoding="utf-8"))


def _module_private_functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Module-level functions whose name marks them private to this module."""
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("_")
    }


def _bare_name_call_lines(tree: ast.Module, name: str) -> list[int]:
    """Lines in the module that call ``name`` by its bare name."""
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    )


def _claims_it_is_uncalled(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether the function's docstring tells the reader nothing calls it."""
    doc = ast.get_docstring(node) or ""
    return bool(_UNCALLED_CLAIM.search(" ".join(doc.split())))


def test_the_soft_stop_helper_has_a_production_call_site() -> None:
    """Premise: the wiring this test grades exists, so it cannot pass vacuously."""
    tree = _module_tree()
    assert "_build_zero_torque_lowcmd" in _module_private_functions(tree)
    call_lines = _bare_name_call_lines(tree, "_build_zero_torque_lowcmd")
    assert call_lines, "the soft-stop helper is no longer called; this test's premise is gone"


def test_no_called_module_private_helper_documents_itself_as_uncalled() -> None:
    """A helper the module calls must not tell the reader it has no caller."""
    tree = _module_tree()
    offenders: list[str] = []
    for name, node in _module_private_functions(tree).items():
        call_lines = _bare_name_call_lines(tree, name)
        if call_lines and _claims_it_is_uncalled(node):
            offenders.append(f"{name}() at line {node.lineno} claims it is unwired, but is called at {call_lines}")
    assert not offenders, (
        "A module-private helper documents itself as unwired while this module "
        "calls it. Describe the call site instead - a reader who believes the "
        "helper is dead will build a second one:\n" + "\n".join(offenders)
    )


def test_the_derivation_reaches_the_helpers_it_grades() -> None:
    """Non-vacuity: the scan found a real population, not an empty one."""
    tree = _module_tree()
    private = _module_private_functions(tree)
    assert len(private) > 3, f"only {len(private)} module-private functions found"
    called = [name for name in private if _bare_name_call_lines(tree, name)]
    assert "_build_zero_torque_lowcmd" in called


def test_the_claim_predicate_separates_the_two_shapes() -> None:
    """The predicate answers both ways, so the rule is not trivially satisfied."""
    module = ast.parse(
        "def _drifted():\n"
        '    """Defined but not yet wired: no other call site exists."""\n'
        "\n"
        "def _accurate():\n"
        '    """The control loop publishes this frame from its finally."""\n'
    )
    functions = _module_private_functions(module)
    assert _claims_it_is_uncalled(functions["_drifted"]) is True
    assert _claims_it_is_uncalled(functions["_accurate"]) is False
