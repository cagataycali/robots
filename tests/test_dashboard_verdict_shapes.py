"""A verdict producer must not emit two shapes of the same list.

Born from a real defect (bcb9cd50 fixing c46dcf2d): ``policy_fit`` appended its existing problems as
``{kind, detail}`` and a NEW problem as ``{field, text, remedy}``. RunForm renders ``p.detail`` keyed
by ``p.kind``, so that problem BLOCKED the run while rendering a blank line - refused, unexplained,
worse than the mismatch it was reporting. Every python test passed, because they asserted the shape
the author had just invented.

The frontend cannot be blamed for this and cannot defend against it either: a payload crossing the
python -> TypeScript boundary has its type hand-declared on the far side, so the compiler never sees
the disagreement. What IS checkable, mechanically and cheaply, is INTERNAL CONSISTENCY: if one list
carries two different key sets, at most one of them can match what the screen reads.

Deliberately a shape check, not a naming rule - this codebase has ``{kind, detail}``, ``{text}`` and
``{reason, remedy}`` payloads that are all correct for their own screens. The rule is only that a
single producer speaks one language.
"""
from __future__ import annotations

import ast
import collections
import pathlib

DASHBOARD = pathlib.Path(__file__).resolve().parents[1] / "strands_robots" / "dashboard"


def shapes_by_producer(source: str, filename: str = "<memory>") -> dict[tuple[str, str, str], list[tuple[int, tuple[str, ...]]]]:
    """Key sets appended as dict literals to a local list, grouped by (file, function, list name).

    Only dict literals with constant string keys are read: a dict built dynamically has no shape
    visible to a static reader, and inventing one would produce false alarms.
    """
    rows: dict[tuple[str, str, str], list[tuple[int, tuple[str, ...]]]] = collections.defaultdict(list)
    for fn in (n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "append" and node.args
                    and isinstance(node.func.value, ast.Name)):
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Dict) and arg.keys and all(
                isinstance(k, ast.Constant) and isinstance(k.value, str) for k in arg.keys
            ):
                rows[(filename, fn.name, node.func.value.id)].append(
                    (node.lineno, tuple(sorted(str(k.value) for k in arg.keys)))  # type: ignore[union-attr]
                )
    return rows


def _all_producers() -> dict[tuple[str, str, str], list[tuple[int, tuple[str, ...]]]]:
    found: dict[tuple[str, str, str], list[tuple[int, tuple[str, ...]]]] = {}
    for f in sorted(DASHBOARD.rglob("*.py")):
        found.update(shapes_by_producer(f.read_text(), str(f.relative_to(DASHBOARD.parent.parent))))
    return found


def test_no_producer_appends_two_different_shapes() -> None:
    offenders = []
    for (f, fn, target), items in sorted(_all_producers().items()):
        shapes = {keys for _, keys in items}
        if len(shapes) > 1:
            detail = "; ".join(f"line {ln}: {list(k)}" for ln, k in items)
            offenders.append(f"{f}:{fn} -> {target} speaks {len(shapes)} shapes ({detail})")
    assert not offenders, (
        "a screen reads ONE set of keys per list, so a second shape renders as a blank:\n  "
        + "\n  ".join(offenders)
    )


def test_the_scan_actually_reaches_the_producers() -> None:
    """A scanner that quietly matches nothing would pass this file forever."""
    producers = _all_producers()
    assert len(producers) >= 10, f"only {len(producers)} producers found - the scan stopped working"
    assert any(fn == "policy_fit" for _, fn, _ in producers), (
        "policy_fit is the module this test was born from; if it is no longer seen, the scan drifted"
    )


def test_the_scanner_catches_the_HISTORICAL_defect() -> None:
    """The exact c46dcf2d mistake, in miniature - the guard must fail on it, not shrug."""
    source = '''
def policy_fit():
    problems = []
    problems.append({"kind": "state_dim", "detail": "6 joints, 5-dim state"})
    problems.append({"field": "norm_tag", "text": "undeclared", "remedy": "pick another"})
    return problems
'''
    shapes = {keys for _, keys in shapes_by_producer(source)[("<memory>", "policy_fit", "problems")]}
    assert len(shapes) == 2, "the scanner must SEE both shapes, or it protects nothing"


def test_a_dynamically_built_dict_is_not_guessed_at() -> None:
    """No shape is visible here, so the honest answer is silence rather than a false alarm."""
    source = '''
def build(rows):
    problems = []
    for r in rows:
        problems.append(dict(r))
    return problems
'''
    assert shapes_by_producer(source) == {}
