"""The API reference's directives resolve to the objects they document.

``docs/reference/api/*.md`` is what a caller reads before their first call. The
pages no longer type signatures by hand: each ``::: dotted.path`` directive asks
mkdocstrings to render the object from the source, and an optional ``members:``
list picks the names shown. That removes the old drift class (a hand-typed
parameter the callable rejects, which is a ``TypeError`` at the call site) and
replaces it with two the build only catches late: a directive whose dotted path
imports nothing, and a ``members:`` name the object does not have. mkdocstrings
turns both into a warning that ``mkdocs build --strict`` fails on, but only in
the docs venv, well after the test suite has said green.

So the directives are graded against ``importlib`` and ``getattr`` here, with
the full package installed. The little prose that still spells a call, such as
``Robot(name, mode="real")``, is graded the old way against
:func:`inspect.signature`: every code span on every line is read, and a callable
taking ``**kwargs`` is not graded on absent names because it accepts any
spelling.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

API_DIR = Path(__file__).resolve().parents[1] / "docs" / "reference" / "api"

_DIRECTIVE = re.compile(r"^::: ([A-Za-z_][\w.]*)\s*$")
_MEMBER = re.compile(r"^\s+- ([A-Za-z_]\w*)\s*$")
_SPANS = re.compile(r"`([^`]+)`")
_CALL = re.compile(r"^([A-Za-z_][\w.]*)\((.*)\)$")
_PARAM = re.compile(r"^([A-Za-z_]\w*)\s*(?:=|$)")


def _pages() -> list[Path]:
    pages = sorted(p for p in API_DIR.glob("*.md") if p.name != "index.md")
    assert pages, f"{API_DIR} holds no API pages"
    return pages


def _directives() -> list[tuple[str, str, list[str]]]:
    """Every ``(page, dotted_path, members)`` directive across the API pages."""
    out: list[tuple[str, str, list[str]]] = []
    for page in _pages():
        current: tuple[str, str, list[str]] | None = None
        in_members = False
        for line in page.read_text(encoding="utf-8").splitlines():
            if directive := _DIRECTIVE.match(line):
                current = (page.name, directive.group(1), [])
                out.append(current)
                in_members = False
                continue
            if current is None:
                continue
            if line.strip() == "members:":
                in_members = True
                continue
            if in_members and (member := _MEMBER.match(line)):
                current[2].append(member.group(1))
                continue
            if line and not line.startswith(" "):
                current, in_members = None, False
            elif in_members and line.strip():
                in_members = False
    return out


def _import_dotted(dotted: str) -> Any:
    """Import the longest module prefix of ``dotted`` and walk the remaining attributes."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            found: Any = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:]:
            found = getattr(found, attr, None)
            if found is None:
                return None
        return found
    return None


@pytest.mark.parametrize(("page", "dotted", "members"), _directives(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_directive_resolves_and_every_member_exists(page: str, dotted: str, members: list[str]) -> None:
    """A ``::: path`` names an importable object and each ``members:`` name is one of its attributes."""
    target = _import_dotted(dotted)
    assert target is not None, f"{page}: `::: {dotted}` resolves to nothing importable"
    absent = [m for m in members if not hasattr(target, m)]
    assert not absent, f"{page}: `::: {dotted}` lists members the object lacks: {absent}"


def _documented_params(arglist: str) -> list[str]:
    """Parameter names a span's argument list spells, skipping ``*``/``...``."""
    names = []
    for part in arglist.replace("\u2026", "").replace("...", "").split(","):
        part = part.strip()
        if part and not part.startswith("*") and (m := _PARAM.match(part)):
            names.append(m.group(1))
    return names


def _resolve(roots: list[Any], dotted: str) -> Any | None:
    """Resolve a span's callable, ignoring a receiver segment like ``recorder.``."""
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


def _prose_calls() -> list[tuple[str, str, list[str], inspect.Signature]]:
    """Code spans shaped like a call in the prose of the API pages, resolved against the page's directives."""
    rows = []
    roots_by_page: dict[str, list[Any]] = {}
    for page, dotted, _ in _directives():
        target = _import_dotted(dotted)
        if target is not None:
            roots_by_page.setdefault(page, []).append(target)
    for path in _pages():
        roots = [importlib.import_module("strands_robots"), *roots_by_page.get(path.name, [])]
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(":::") or line.startswith(" "):
                continue
            for span in _SPANS.findall(line):
                call = _CALL.match(span.strip())
                if not call:
                    continue
                params = _documented_params(call.group(2))
                target = _resolve(roots, call.group(1))
                if not params or target is None:
                    continue
                try:
                    rows.append((path.name, span, params, inspect.signature(target)))
                except (TypeError, ValueError):
                    continue
    return rows


def test_every_documented_parameter_is_a_name_its_callable_accepts() -> None:
    """No prose call spells a parameter its callable would reject."""
    drifted = []
    for page, span, params, signature in _prose_calls():
        accepted = set(signature.parameters) - {"self", "cls"}
        if any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
            continue
        if absent := [p for p in params if p not in accepted]:
            drifted.append(f"{page}: `{span}` names {absent}, accepted: {sorted(accepted)}")
    assert not drifted, "docs/reference/api documents parameters that do not exist:\n" + "\n".join(drifted)


def test_the_reference_grades_the_directives_it_is_written_for() -> None:
    """The parser reaches the directives; a grader that resolves nothing passes vacuously."""
    directives = _directives()
    assert len(directives) >= 30, f"only {len(directives)} directives found: the parser or the reference moved"
    documented = {d.split(".")[-1] for _, d, _ in directives} | {m for _, _, members in directives for m in members}
    for expected in (
        "Robot",
        "list_robots",
        "register_robot",
        "create_simulation",
        "register_backend",
        "create_policy",
        "Mesh",
        "run_policy",
    ):
        assert expected in documented, f"{expected} is no longer documented by any API page"


def test_simengine_directive_exposes_run_policy() -> None:
    """The engine contract is rendered by filter, so the method a caller wants most must be public on it."""
    from strands_robots.simulation.base import SimEngine

    assert any(d == "strands_robots.simulation.base.SimEngine" for _, d, _ in _directives())
    assert callable(getattr(SimEngine, "run_policy", None))


@pytest.mark.parametrize(
    ("mode", "documented"),
    [("all", True), ("sim", True), ("real", True), ("both", True), ("arm", False)],
)
def test_list_robots_mode_is_a_backend_filter_not_a_category(mode: str, documented: bool) -> None:
    """The ``mode`` values the rendered docstring lists are the ones the registry honours."""
    from strands_robots.registry import list_robots

    if documented:
        assert f'"{mode}"' in (list_robots.__doc__ or "")
        assert isinstance(list_robots(mode), list)
    else:
        with pytest.raises(ValueError, match="Unknown list_robots mode"):
            list_robots(mode)
