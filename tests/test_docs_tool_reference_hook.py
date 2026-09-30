"""``docs/hooks/tools_ref.py`` publishes the catalog an agent is handed.

The tool reference is generated at build time from the source, so the page cannot go stale the
way a hand-written table does. What it *can* do is disagree with ``strands.tool``: the hook reads
each signature and docstring with :mod:`ast`, because the docs environment installs mkdocs alone
and cannot import the package, while an agent is handed a schema that ``docstring_parser`` and
pydantic build from the same function. This grader drives both and compares them tool by tool:
names, parameters in order, the first sentence, and the action values a dispatching tool accepts.

That comparison is not decorative. The predecessor of this hook caught two tools whose parameter
description no agent ever received (``g1_decode_error_code`` and ``g1_get_state`` began a
docstring line with an RST role, so the Google ``Args:`` block was never read). The action column
is the new place a reader can be misled: the hook falls back to a module constant named
``_ACTIONS``, and a module that uses that name for something else (a verb table keyed by tool
name) would publish tool names as the values of a gesture parameter. So the action column is
graded against the function itself: a ``Literal`` enum when the schema has one, and nothing at all
when the body never compares ``action`` to a value.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
from pathlib import Path

import pytest

import strands_robots
from tests._docs_hooks import docs_hook
from tests._package_ast import parse_file

_REPO = Path(strands_robots.__file__).resolve().parents[1]
_PKG = _REPO / "strands_robots"
_HOOK = _REPO / "docs" / "hooks" / "tools_ref.py"
_PAGE = _REPO / "docs" / "reference" / "tools.md"

# A hook that found nothing must not read as a clean sweep.
_MINIMUM_TOOLS = 50


def _hook():  # noqa: ANN202 - the loaded hook module
    """The hook module, loaded from the docs tree the build loads it from."""
    return docs_hook("tools_ref")


def _published() -> tuple:
    """Every tool the hook puts on the page, flattened."""
    return tuple(item for family in _hook().tools().values() for item in family)


def _importable() -> tuple:
    """The published tools a caller can import by name: top level, literal name."""
    return tuple(item for item in _published() if not item.nested and "<" not in item.name)


def _importable_tools(tree: ast.Module):  # noqa: ANN202
    """Every ``def`` a caller can reach as an attribute: module level, or on a class there.

    A ``@tool`` written inside another function is not one of those. The dashboard's agent
    console builds its tools per conversation, closed over that conversation's safety object,
    so nothing can import them; the hook lists them as "built at runtime" and this grader holds
    them to nothing more than their existence.
    """
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield node
        elif isinstance(node, ast.ClassDef):
            yield from (item for item in node.body if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef))


def _statically_named_tools() -> set[tuple[str, str]]:
    """(module, name) for every importable ``@tool`` in the package whose name is a literal.

    Derived here rather than through the hook, so a hook that stops looking in a directory
    fails this grader instead of quietly shortening the page. A tool whose name the decorator
    computes per instance (the mesh robots build one per robot) names nothing a page could
    spell exactly and is out of scope here.
    """
    found: set[tuple[str, str]] = set()
    for path in sorted(_PKG.rglob("*.py")):
        tree = parse_file(path)
        for node in _importable_tools(tree):
            for decorator in node.decorator_list:
                call = decorator.func if isinstance(decorator, ast.Call) else decorator
                if not (isinstance(call, ast.Name) and call.id == "tool"):
                    continue
                name: str | None = node.name
                if isinstance(decorator, ast.Call):
                    for keyword in decorator.keywords:
                        if keyword.arg == "name":
                            value = keyword.value
                            name = str(value.value) if isinstance(value, ast.Constant) else None
                if name is not None:
                    module = ".".join(path.relative_to(_REPO).with_suffix("").parts)
                    found.add((module, name))
    return found


def _function_node(item) -> ast.FunctionDef | ast.AsyncFunctionDef:  # noqa: ANN001 - the hook's Tool
    path = _REPO / (item.module.replace(".", "/") + ".py")
    tree = parse_file(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == item.name:
            return node
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and any(
                    k.arg == "name" and isinstance(k.value, ast.Constant) and k.value.value == item.name
                    for k in decorator.keywords
                ):
                    return node
    raise AssertionError(f"{item.module}.{item.name} not found in the source")


def _plain(text: str) -> str:
    """The schema text with RST markup reduced the way the page reduces it: roles and ``x`` to `x`."""
    text = re.sub(r":[a-z:]+:`~?([^`]+)`", lambda m: f"`{m.group(1)}`", text)
    return re.sub(r"``([^`]+)``", lambda m: f"`{m.group(1)}`", text)


def _body_compares_action(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "action"
        for node in ast.walk(fn)
    )


def test_the_page_names_every_statically_named_tool_in_the_package() -> None:
    published = {(item.module, item.name) for item in _importable()}
    assert len(published) >= _MINIMUM_TOOLS
    assert published == _statically_named_tools()


def test_every_importable_tool_gets_exactly_one_row() -> None:
    page = _hook().render()
    for item in _importable():
        rows = [line for line in page.splitlines() if line.startswith(f"| <code>{item.name}</code> |")]
        assert len(rows) == 1, (item.name, len(rows))


@pytest.mark.parametrize("item", _importable(), ids=lambda item: item.name)
def test_the_page_publishes_the_surface_the_agent_receives(item) -> None:  # noqa: ANN001 - the hook's Tool
    schema = getattr(importlib.import_module(item.module), item.name).tool_spec
    properties = schema["inputSchema"]["json"]["properties"]

    assert list(item.params) == list(properties), f"{item.name}: page {item.params}, schema {list(properties)}"
    description = _plain(" ".join(schema["description"].split()))
    assert description.startswith(item.summary.rstrip(".")), f"{item.name}: {item.summary!r} vs {description[:120]!r}"


@pytest.mark.parametrize("item", [i for i in _importable() if "action" in i.params], ids=lambda item: item.name)
def test_the_action_column_is_what_the_function_dispatches_on(item) -> None:  # noqa: ANN001 - the hook's Tool
    """The column comes from a ``Literal``, or from values the body compares ``action`` against, or is empty."""
    schema = getattr(importlib.import_module(item.module), item.name).tool_spec
    enum = schema["inputSchema"]["json"]["properties"]["action"].get("enum")
    if enum:
        assert list(item.actions) == list(enum), f"{item.name}: page {item.actions}, schema enum {enum}"
        return
    if not _body_compares_action(_function_node(item)):
        assert not item.actions, (
            f"{item.name} never compares its `action` parameter to a value, yet the page lists "
            f"{item.actions} as its actions. The hook's `_ACTIONS` fallback picked up a module "
            "constant that is not this tool's action set."
        )
        return
    source = ast.unparse(_function_node(item))
    stray = [a for a in item.actions if f"'{a}'" not in source and f'"{a}"' not in source]
    assert stray == [] or all(
        f'"{a}"' in (_REPO / (item.module.replace(".", "/") + ".py")).read_text() for a in stray
    ), f"{item.name}: the page lists {stray} as actions; neither the body nor the module spells them"


def test_a_default_or_type_carrying_a_pipe_cannot_split_a_table_row() -> None:
    """``str | None`` is the commonest annotation here, and a raw pipe would forge a column."""
    rows = [line for line in _hook().render().splitlines() if line.startswith("| ") and "---" not in line]
    assert rows
    for row in rows:
        assert len(re.split(r"(?<!\\)\|", row.strip("|"))) == 4, row


def test_the_hook_reads_the_tree_without_importing_the_package() -> None:
    """The docs environment installs mkdocs only, so an import of the package fails the build."""
    tree = parse_file(_HOOK)
    imported = {
        alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    } | {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert "strands_robots" not in imported
    assert "strands" not in imported


def test_the_page_carries_the_token_and_no_hand_written_table() -> None:
    """The catalog is the token; the prose above it may not restate a row by hand."""
    source = _PAGE.read_text(encoding="utf-8")
    assert source.count("{{tools_ref}}") == 1
    assert not [line for line in source.splitlines() if line.startswith("| ")], (
        "a hand-written table row on the tools page"
    )
    rendered = _hook().on_page_markdown(source, None, None, None)
    assert "{{tools_ref}}" not in rendered
    assert "| tool | module | does | actions or parameters |" in rendered
