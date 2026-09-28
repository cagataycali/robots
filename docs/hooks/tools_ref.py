"""mkdocs hook: the agent tool catalog, generated from every ``@tool`` in the package.

``{{tools_ref}}`` on a page becomes one table per area (core tools, Unitree G1,
Reachy Mini, mesh bridges, dashboard console). Each row is one ``@tool``: its
name, module, first docstring line, and the action values it dispatches on.

Action values come from the source in this order: a ``Literal[...]`` annotation
on the ``action`` parameter, a module constant named ``_ACTIONS`` (tuple, set or
dict of string keys) when the body keys it by ``action``, then every
``action == "x"`` / ``action in {...}`` comparison in the function body. A tool without an ``action`` parameter shows
its parameter names instead.

Filesystem only: ``ast`` over the source, no ``strands_robots`` import, so the
hook runs in the docs venv. ``python3 docs/hooks/tools_ref.py`` prints the
markdown for a look.
"""

from __future__ import annotations

import ast
import html
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.tools_ref")

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "strands_robots"
_TOKEN = re.compile(r"^\{\{\s*tools_ref\s*\}\}\s*$", re.M)

# (path prefix relative to strands_robots/, heading, blurb). First match wins.
_AREAS: tuple[tuple[str, str, str], ...] = (
    ("tools/g1/", "Unitree G1", "The humanoid's DDS surface: locomotion, arm gestures, tasks and sensor reads."),
    ("tools/reachy/", "Reachy Mini", "Head, antennas, body turn, home and stop on the Mini daemon."),
    (
        "tools/",
        "Core",
        "Cameras, teleoperation, training, rollouts, poses, the serial bus, the mesh and ROS transports.",
    ),
    ("mesh/", "Mesh bridges", "Tools a bridged ROS 2, rosbridge or mobile-base robot publishes on the mesh."),
    ("dashboard/", "Dashboard console", "Tools the dashboard's agent console hands its model."),
)

_CONTEXT_ANNOTATIONS = frozenset({"ToolContext", "ToolContext | None", "Optional[ToolContext]"})
_ROLE = re.compile(r":[a-z:]+:`~?([^`]+)`")
_RST_LITERAL = re.compile(r"``([^`]+)``")


@dataclass(frozen=True)
class Tool:
    """One statically named ``@tool``."""

    name: str
    module: str
    summary: str
    actions: tuple[str, ...]
    params: tuple[str, ...]
    nested: bool


def _is_tool_decorator(node: ast.expr) -> bool:
    func = node.func if isinstance(node, ast.Call) else node
    return (isinstance(func, ast.Name) and func.id == "tool") or (
        isinstance(func, ast.Attribute) and func.attr == "tool"
    )


def _template(node: ast.expr) -> str | None:
    """A string constant, or an f-string with each ``{expr}`` shown as ``<expr>``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                parts.append(f"<{ast.unparse(value.value)}>")
        return "".join(parts)
    return None


def _tool_name(node: ast.FunctionDef | ast.AsyncFunctionDef, decorator: ast.expr) -> str | None:
    """The published tool name; an f-string name keeps its ``<placeholder>``."""
    if isinstance(decorator, ast.Call):
        for keyword in decorator.keywords:
            if keyword.arg == "name":
                return _template(keyword.value)
    return node.name


def _decorator_description(decorator: ast.expr) -> str | None:
    """``@tool(description=...)`` when it is a literal, else None."""
    if isinstance(decorator, ast.Call):
        for keyword in decorator.keywords:
            if keyword.arg == "description":
                return _template(keyword.value)
    return None


def _strings(node: ast.expr) -> tuple[str, ...]:
    """String constants inside a tuple/list/set/dict/frozenset(...) literal."""
    if isinstance(node, ast.Call) and node.args:  # frozenset({...}) / tuple((...))
        return _strings(node.args[0])
    if isinstance(node, ast.Dict):
        return tuple(k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str))
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        out: list[str] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
            elif isinstance(elt, ast.Starred):
                out.extend(_strings(elt.value))
        return tuple(out)
    return ()


def _module_constants(tree: ast.Module) -> dict[str, ast.expr]:
    out: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            out[node.target.id] = node.value
    return out


def _resolve(node: ast.expr, constants: dict[str, ast.expr], depth: int = 0) -> tuple[str, ...]:
    """Strings behind a literal or a module-level name (one level of starred expansion)."""
    if depth > 3:
        return ()
    if isinstance(node, ast.Name) and node.id in constants:
        return _resolve(constants[node.id], constants, depth + 1)
    if isinstance(node, ast.Tuple | ast.List | ast.Set | ast.Dict | ast.Call):
        out: list[str] = []
        elts = node.elts if isinstance(node, ast.Tuple | ast.List | ast.Set) else []
        if isinstance(node, ast.Call) and node.args:
            return _resolve(node.args[0], constants, depth + 1)
        if isinstance(node, ast.Dict):
            return _strings(node)
        for elt in elts:
            if isinstance(elt, ast.Starred):
                out.extend(_resolve(elt.value, constants, depth + 1))
            elif isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
        return tuple(out)
    return ()


def _literal_values(annotation: ast.expr | None) -> tuple[str, ...]:
    """``Literal["a", "b"]`` values on an annotation, or ()."""
    if annotation is None:
        return ()
    for node in ast.walk(annotation):
        if isinstance(node, ast.Subscript):
            base = node.value
            name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
            if name == "Literal":
                inner = node.slice
                elts = inner.elts if isinstance(inner, ast.Tuple) else [inner]
                return tuple(e.value for e in elts if isinstance(e, ast.Constant) and isinstance(e.value, str))
    return ()


def _body_actions(fn: ast.FunctionDef | ast.AsyncFunctionDef, constants: dict[str, ast.expr]) -> tuple[str, ...]:
    """Every value ``action`` is compared against in the body, in source order."""
    out: list[str] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Name) or node.left.id != "action":
            continue
        for op, right in zip(node.ops, node.comparators, strict=True):
            if isinstance(op, ast.Eq | ast.NotEq) and isinstance(right, ast.Constant) and isinstance(right.value, str):
                out.append(right.value)
            elif isinstance(op, ast.In | ast.NotIn):
                out.extend(_resolve(right, constants))
    seen: dict[str, None] = {}
    for value in out:
        seen.setdefault(value, None)
    return tuple(seen)


def _actions(fn: ast.FunctionDef | ast.AsyncFunctionDef, constants: dict[str, ast.expr]) -> tuple[str, ...]:
    args = [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]
    action = next((a for a in args if a.arg == "action"), None)
    if action is None:
        return ()
    values = _literal_values(action.annotation)
    if values:
        return values
    if "_ACTIONS" in constants and _body_keys_by(fn, "action", "_ACTIONS"):
        values = _resolve(constants["_ACTIONS"], constants)
        if values:
            return values
    return _body_actions(fn, constants)


def _body_keys_by(fn: ast.FunctionDef | ast.AsyncFunctionDef, param: str, constant: str) -> bool:
    """Whether the body relates ``param`` to ``constant``: ``param in constant``, ``constant[param]``, ``constant.get(param)``.

    A module constant named ``_ACTIONS`` is only the action vocabulary when the
    function keys it by ``action``; a verb table keyed by tool name (g1_actions.py)
    shares the name and must not be published as the values ``action`` accepts.
    """
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == param:
            if any(isinstance(c, ast.Name) and c.id == constant for c in node.comparators):
                return True
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == constant:
            if isinstance(node.slice, ast.Name) and node.slice.id == param:
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            receiver = node.func.value
            if isinstance(receiver, ast.Name) and receiver.id == constant and node.args:
                if isinstance(node.args[0], ast.Name) and node.args[0].id == param:
                    return True
    return False


def _params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    out: list[str] = []
    for arg in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]:
        annotation = ast.unparse(arg.annotation) if arg.annotation else ""
        if annotation in _CONTEXT_ANNOTATIONS or arg.arg in {"self", "cls"}:
            continue
        out.append(arg.arg)
    return tuple(out)


def _summary(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    doc = ast.get_docstring(fn, clean=True) or ""
    first = doc.split("\n\n")[0].strip()
    first = _ROLE.sub(lambda m: f"`{m.group(1)}`", first)
    first = _RST_LITERAL.sub(lambda m: f"`{m.group(1)}`", first)
    return " ".join(first.split())


def _read(path: Path) -> list[Tool]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants = _module_constants(tree)
    module = ".".join(path.relative_to(_REPO).with_suffix("").parts)
    top_level = {id(n) for n in tree.body}
    out: list[Tool] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        decorators = [d for d in node.decorator_list if _is_tool_decorator(d)]
        if not decorators:
            continue
        name = _tool_name(node, decorators[0])
        if name is None:
            continue
        summary = _decorator_description(decorators[0]) or _summary(node)
        summary = " ".join(summary.split()).split(". ")[0].rstrip(".")
        summary = f"{summary}." if summary else "described at runtime"
        out.append(
            Tool(
                name=name,
                module=module,
                summary=summary,
                actions=_actions(node, constants),
                params=_params(node),
                nested=id(node) not in top_level,
            )
        )
    return out


def _area(path: Path) -> tuple[str, str]:
    rel = path.relative_to(_PKG).as_posix()
    for prefix, heading, blurb in _AREAS:
        if rel.startswith(prefix):
            return heading, blurb
    return "Other", "Tools outside the areas above."


@lru_cache(maxsize=1)
def tools() -> dict[tuple[str, str], tuple[Tool, ...]]:
    """Every statically named ``@tool`` under the package, grouped by area in page order."""
    grouped: dict[tuple[str, str], list[Tool]] = {}
    for path in sorted(_PKG.rglob("*.py")):
        found = _read(path)
        if found:
            grouped.setdefault(_area(path), []).extend(found)
    order = {heading: index for index, (_, heading, _) in enumerate(_AREAS)}
    return {
        key: tuple(sorted(value, key=lambda t: t.name))
        for key, value in sorted(grouped.items(), key=lambda kv: order.get(kv[0][0], 99))
    }


def _code(text: str) -> str:
    return "<code>" + html.escape(text, quote=False).replace("|", "&#124;") + "</code>"


def _cell(text: str) -> str:
    parts = html.escape(text, quote=False).split("`")
    if len(parts) % 2 == 0:
        return "`".join(p.replace("|", r"\|") for p in parts)
    return "".join(_code(p) if i % 2 else p.replace("|", r"\|") for i, p in enumerate(parts))


def render() -> str:
    """The whole catalog as markdown."""
    groups = tools()
    total = sum(len(v) for v in groups.values())
    lines = [f"{total} tools in {len(groups)} areas, read from the `@tool` decorators at this commit.", ""]
    for (heading, blurb), items in groups.items():
        lines += [f"## {heading}", "", f"{blurb} {len(items)} tools.", ""]
        lines += ["| tool | module | does | actions or parameters |", "|---|---|---|---|"]
        for t in items:
            if t.actions:
                verbs = ", ".join(_code(a) for a in t.actions)
            elif t.params:
                verbs = "params: " + ", ".join(_code(p) for p in t.params)
            else:
                verbs = "no parameters"
            name = _code(t.name) + (" (built at runtime)" if t.nested else "")
            lines.append(
                f"| {name} | {_code(t.module.removeprefix('strands_robots.'))} | {_cell(t.summary)} | {verbs} |"
            )
        lines.append("")
    return "\n".join(lines)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001
    """mkdocs hook entry point: expand ``{{tools_ref}}``."""
    if not _TOKEN.search(markdown):
        return markdown
    return _TOKEN.sub(lambda _: render(), markdown)


if __name__ == "__main__":
    print(render())
