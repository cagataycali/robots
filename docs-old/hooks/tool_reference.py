"""mkdocs hook: the agent tool reference, generated from the tool functions.

``{{tool_reference}}`` in a page becomes one section per agent-callable
``@tool`` under ``strands_robots/tools/``: the tool's one-line summary, the
operator gate when it has one, what its ``Returns:`` section promises (which is
where each tool says what it refuses), and a table of every parameter with its
annotation, default and the description the agent is shown.

The facts are read out of the source with :mod:`ast`, so the page cannot
disagree with the schema an agent receives -- ``tests/test_docs_tool_reference_hook.py``
compares every field on this page against the live ``tool_spec`` that
``strands.tool`` builds from the same function. A tool added to the package
appears at the next build, and a renamed parameter cannot linger.

Filesystem only (no ``strands_robots`` import), so the hook runs in the docs
venv without the package's optional extras.
"""

from __future__ import annotations

import ast
import html
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.tool_reference")

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "strands_robots" / "tools"
_TOKEN = re.compile(r"^\{\{\s*tool_reference\s*\}\}\s*$", re.M)

# The tool families, in page order: (directory, heading, blurb).
_FAMILIES: tuple[tuple[str, str, str], ...] = (
    (".", "Core", "Cameras, teleoperation, training, rollout, poses, the serial bus, the mesh and the ROS transports."),
    (
        "g1",
        "Unitree G1",
        "The humanoid's DDS surface: locomotion FSM, arm actions, task control and every sensor read.",
    ),
    ("reachy", "Reachy Mini", "The Mini's daemon surface: head and antenna gestures, emotions, sound and camera."),
)

# A parameter carrying the tool context is supplied by the agent runtime, not by
# the caller, and ``strands.tool`` keeps it out of the schema it publishes.
_CONTEXT_ANNOTATIONS = frozenset({"ToolContext", "ToolContext | None", "Optional[ToolContext]"})

_ROLE = re.compile(r":[a-z:]+:`~?([^`]+)`")
_RST_LITERAL = re.compile(r"``([^`]+)``")


@dataclass(frozen=True)
class Param:
    """One parameter of a tool, as the agent's schema declares it."""

    name: str
    annotation: str
    default: str | None
    description: str

    @property
    def required(self) -> bool:
        """True when the caller must supply the parameter (no default)."""
        return self.default is None


@dataclass(frozen=True)
class Tool:
    """One agent-callable ``@tool``, read out of the source."""

    name: str
    module: str
    summary: str
    returns: str
    takes_context: bool
    params: tuple[Param, ...] = field(default_factory=tuple)


def _text(node: ast.expr) -> str:
    """Source text of an annotation or default expression."""
    return ast.unparse(node)


def _collapse(text: str) -> str:
    """One line of whitespace-collapsed prose."""
    return " ".join(text.split())


def _sections(doc: str) -> dict[str, str]:
    """Split a Google-style docstring into its named sections.

    The unnamed summary and prose above the first section land under ``""``.
    """
    out: dict[str, list[str]] = {"": []}
    current = ""
    for line in doc.splitlines():
        header = re.match(r"^([A-Z][A-Za-z ]*):\s*$", line)
        if header:
            current = header.group(1)
            out.setdefault(current, [])
            continue
        out[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in out.items()}


def _arg_descriptions(args_section: str) -> dict[str, str]:
    """Parameter name -> its description, from a docstring ``Args:`` section."""
    out: dict[str, str] = {}
    name = ""
    for line in args_section.splitlines():
        entry = re.match(r"^(\*{0,2}[A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?:\s*(.*)$", line.strip())
        if entry and not line.startswith(("     ", "\t")):
            name = entry.group(1).lstrip("*")
            out[name] = entry.group(2)
        elif name and line.strip():
            out[name] = f"{out[name]} {line.strip()}".strip()
    return {k: _collapse(v) for k, v in out.items()}


def _markdown(text: str) -> str:
    """Docstring prose as markdown: RST roles and literals become code spans."""
    text = _ROLE.sub(lambda m: f"`{m.group(1)}`", text)
    text = _RST_LITERAL.sub(lambda m: f"`{m.group(1)}`", text)
    return text


def _code(text: str) -> str:
    """Inline code as HTML, so a ``|`` inside it cannot split a table row."""
    return "<code>" + html.escape(text, quote=False).replace("|", "&#124;") + "</code>"


def _cell(text: str) -> str:
    """Prose fit for a markdown table cell: code spans become HTML, pipes escape."""
    parts = _markdown(_collapse(text)).split("`")
    if len(parts) % 2 == 0:  # unbalanced backticks - nothing here is a code span
        return "`".join(part.replace("|", r"\|") for part in parts)
    return "".join(_code(part) if index % 2 else part.replace("|", r"\|") for index, part in enumerate(parts))


def _is_tool_decorator(node: ast.expr) -> bool:
    """True for ``@tool`` and ``@tool(...)``, however ``tool`` was imported."""
    func = node.func if isinstance(node, ast.Call) else node
    return (isinstance(func, ast.Name) and func.id == "tool") or (
        isinstance(func, ast.Attribute) and func.attr == "tool"
    )


def _tool_name(node: ast.FunctionDef | ast.AsyncFunctionDef, decorator: ast.expr) -> str | None:
    """The published tool name, or None when the decorator computes it."""
    if isinstance(decorator, ast.Call):
        for keyword in decorator.keywords:
            if keyword.arg == "name":
                return keyword.value.value if isinstance(keyword.value, ast.Constant) else None
    return node.name


def _constants(tree: ast.Module) -> dict[str, str]:
    """Module-scope literal constants, so a default named by one shows its value."""
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign | ast.AnnAssign) and isinstance(node.value, ast.Constant):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    out[target.id] = _text(node.value)
    return out


def _default(node: ast.expr, constants: dict[str, str]) -> str:
    """A default as the reader should see it: the value, not the name that holds it."""
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    return _text(node)


def _params(
    node: ast.FunctionDef | ast.AsyncFunctionDef, described: dict[str, str], constants: dict[str, str]
) -> tuple[Param, ...]:
    """Every parameter the agent may supply, in declaration order."""
    spec = node.args
    positional = [*spec.posonlyargs, *spec.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(spec.defaults)) + list(spec.defaults)
    out: list[Param] = []
    for arg, default in [*zip(positional, defaults, strict=True), *zip(spec.kwonlyargs, spec.kw_defaults, strict=True)]:
        annotation = _text(arg.annotation) if arg.annotation else ""
        if annotation in _CONTEXT_ANNOTATIONS:
            continue
        out.append(
            Param(
                name=arg.arg,
                annotation=annotation,
                default=_default(default, constants) if default is not None else None,
                description=described.get(arg.arg, ""),
            )
        )
    return tuple(out)


def _module(path: Path) -> str:
    """Dotted module path of a file inside the package."""
    return ".".join(path.relative_to(_REPO).with_suffix("").parts)


def _read(path: Path) -> list[Tool]:
    """Every statically named ``@tool`` in one source file."""
    out: list[Tool] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants = _constants(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        decorators = [d for d in node.decorator_list if _is_tool_decorator(d)]
        if not decorators:
            continue
        name = _tool_name(node, decorators[0])
        if name is None:  # a per-instance tool whose name is built at runtime
            continue
        sections = _sections(ast.get_docstring(node, clean=True) or "")
        described = _arg_descriptions(sections.get("Args", ""))
        params = _params(node, described, constants)
        out.append(
            Tool(
                name=name,
                module=_module(path),
                summary=_collapse(sections[""].split("\n\n")[0]),
                returns=_collapse(sections.get("Returns", "")),
                takes_context=len(params) < len([*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]),
                params=params,
            )
        )
    return out


@lru_cache(maxsize=1)
def tools() -> dict[str, tuple[Tool, ...]]:
    """Every agent-callable tool, grouped by family in page order."""
    out: dict[str, tuple[Tool, ...]] = {}
    for directory, heading, _ in _FAMILIES:
        root = _TOOLS if directory == "." else _TOOLS / directory
        found: list[Tool] = []
        for path in sorted(root.glob("*.py")):
            found.extend(_read(path))
        out[heading] = tuple(sorted(found, key=lambda t: t.name))
    return out


def _table(params: tuple[Param, ...]) -> str:
    """The parameter table for one tool."""
    if not params:
        return "_No parameters._\n"
    rows = [
        "| Parameter | Type | Default | Description |",
        "|-----------|------|---------|-------------|",
    ]
    for param in params:
        default = "**required**" if param.required else _code(param.default or "")
        rows.append(f"| {_code(param.name)} | {_code(param.annotation)} | {default} | {_cell(param.description)} |")
    return "\n".join(rows) + "\n"


def section(item: Tool) -> str:
    """One tool as a markdown section."""
    lines = [f"### `{item.name}`", "", _markdown(item.summary), ""]
    if item.takes_context:
        lines += [
            "Takes the agent's tool context (`@tool(context=True)`) - the seam it prompts an operator through.",
            "",
        ]
    lines += [f"`{item.module}`", "", _table(item.params)]
    if item.returns:
        lines += ["", f"**Returns.** {_markdown(item.returns)}", ""]
    return "\n".join(lines)


def render() -> str:
    """The whole reference: every family, every tool."""
    grouped = tools()
    total = sum(len(family) for family in grouped.values())
    blurbs = {heading: blurb for _, heading, blurb in _FAMILIES}
    out = [f"{total} tools, generated from the source at build time.", ""]
    for heading, family in grouped.items():
        out += [f"## {heading}", "", blurbs[heading], ""]
        out += [", ".join(f"[`{t.name}`](#{t.name})" for t in family), ""]
        out += [section(item) for item in family]
    return "\n".join(out)


def substitute(markdown: str) -> str:
    """Replace a ``{{tool_reference}}`` token with the rendered reference."""
    return _TOKEN.sub(lambda _: render(), markdown)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001 - mkdocs signature
    """mkdocs hook: expand the token before the page is rendered."""
    if _TOKEN.search(markdown) and not tools():
        log.warning("tool_reference found no tools under %s", _TOOLS)
    return substitute(markdown)
