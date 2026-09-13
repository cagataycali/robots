"""The API reference's parameter names are names the callables accept.

``docs/api-reference.md`` is the table a caller reads before their first call,
and every row leads with a signature-shaped code span. A parameter name in that
span the callable does not accept is not a typo a reader routes around: it is a
``TypeError`` at the call site. Worse, when the wrong name is a real concept
somewhere else in the package -- ``list_robots(category=...)`` against a
registry that really does group robots by category -- the row sends the reader
after a filter that does not exist and describes what it would return.

So the spans are graded against :func:`inspect.signature` rather than reviewed.
A row whose callable takes ``**kwargs`` is not graded on absent names: such a
callable accepts any spelling, so the doc cannot be wrong about one.

*Every* code span on a table line is graded, not just the leading one. A cell
that pairs two related callables -- ``list_backends()`` beside
``register_backend(name, loader)`` -- puts the second signature where a
leading-span reader never looks, and the reader who copies it is no better off
for the row having started with something correct.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

DOC = Path(__file__).resolve().parents[1] / "docs" / "api-reference.md"

_HEADING = re.compile(r"^#+ `([A-Za-z_][\w.]*)`\s*$")
_ROW_SPANS = re.compile(r"`([^`]+)`")
_CALL = re.compile(r"^([A-Za-z_][\w.]*)\((.*)\)$")
_PARAM = re.compile(r"^([A-Za-z_]\w*)\s*(?:=|$)")


def _sections() -> list[tuple[str, list[str], list[str]]]:
    """Split the reference into ``(module, fence_names, row_spans)`` sections."""
    out: list[tuple[str, list[str], list[str]]] = []
    module: str | None = None
    fence: list[str] = []
    spans: list[str] = []
    in_fence = False
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if heading := _HEADING.match(line):
            if module:
                out.append((module, fence, spans))
            module, fence, spans, in_fence = heading.group(1), [], [], False
            continue
        if module is None:
            continue
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            fence += re.findall(r"[A-Za-z_]\w*", line)
        elif line.lstrip().startswith("|"):
            spans += _ROW_SPANS.findall(line)
    if module:
        out.append((module, fence, spans))
    return out


def _owners(module_name: str, fence: list[str]) -> list[Any]:
    """The module plus the classes its import fence names, as lookup roots."""
    module = importlib.import_module(module_name)
    roots: list[Any] = [module]
    for name in dict.fromkeys(fence):
        candidate = getattr(module, name, None)
        if inspect.isclass(candidate):
            roots.append(candidate)
    return roots


def _resolve(roots: list[Any], dotted: str) -> Any | None:
    """Resolve a row's callable, ignoring a receiver segment like ``recorder.``."""
    parts = dotted.split(".")
    for start in range(len(parts)):
        for root in roots:
            found: Any = root
            for attr in parts[start:]:
                found = getattr(found, attr, None)
                if found is None:
                    break
            if callable(found):
                return found
    return None


def _documented_params(arglist: str) -> list[str]:
    """Parameter names a row's argument list spells, skipping ``*``/``...``."""
    names = []
    for part in arglist.replace("\u2026", "").replace("...", "").split(","):
        part = part.strip()
        if part and not part.startswith("*") and (m := _PARAM.match(part)):
            names.append(m.group(1))
    return names


def _graded_rows() -> list[tuple[str, str, list[str], inspect.Signature]]:
    rows = []
    for module_name, fence, spans in _sections():
        roots = _owners(module_name, fence)
        for span in spans:
            call = _CALL.match(span.strip())
            if not call:
                continue
            params = _documented_params(call.group(2))
            target = _resolve(roots, call.group(1))
            if not params or target is None:
                continue
            try:
                signature = inspect.signature(target)
            except (TypeError, ValueError):
                continue
            rows.append((module_name, span, params, signature))
    return rows


def test_every_documented_parameter_is_a_name_its_callable_accepts() -> None:
    """No row spells a parameter its callable would reject."""
    drifted = []
    for module_name, span, params, signature in _graded_rows():
        accepted = set(signature.parameters) - {"self", "cls"}
        if any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
            continue
        if absent := [p for p in params if p not in accepted]:
            drifted.append(f"{module_name}: `{span}` names {absent}, accepted: {sorted(accepted)}")
    assert not drifted, "docs/api-reference.md documents parameters that do not exist:\n" + "\n".join(drifted)


def test_the_reference_grades_the_rows_it_is_written_for() -> None:
    """The parser reaches the table; a grader that resolves nothing passes vacuously."""
    rows = _graded_rows()
    assert len(rows) >= 30, f"only {len(rows)} rows resolved -- the parser or the reference moved"
    graded = {row[1].split("(")[0] for row in rows}
    expected_names = (
        "list_robots",
        "register_robot",
        "run_policy",
        "start_task",
        "recorder.add_frame",
        # Second span in its cell: reached only because whole lines are read.
        "register_backend",
    )
    for expected in expected_names:
        assert expected in graded, f"{expected} is documented with parameters but was not graded"


@pytest.mark.parametrize(
    ("mode", "documented"),
    [("all", True), ("sim", True), ("real", True), ("both", True), ("arm", False)],
)
def test_list_robots_mode_is_a_backend_filter_not_a_category(mode: str, documented: bool) -> None:
    """The reference's ``mode`` values are the ones the registry honours."""
    from strands_robots.registry import list_robots

    if documented:
        assert isinstance(list_robots(mode), list)
    else:
        with pytest.raises(ValueError, match="Unknown list_robots mode"):
            list_robots(mode)
