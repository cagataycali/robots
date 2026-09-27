"""mkdocs hook: the layered module map with line counts, generated from the tree.

``{{module_map}}`` on a page becomes one table row per layer of the import DAG
that ``scripts/check_import_layers.py`` grades: the layer name, its top-level
members (module or package under ``strands_robots/``) each with its line count,
and the layer total. ``{{module_map_total}}`` becomes the package total.

The layer list is read from ``LAYERS`` in that script with ``ast``, so the docs
cannot disagree with the grader about where a module sits. Lines are counted
over every ``*.py`` under the member (all lines, the same number ``wc -l``
prints). Filesystem only, no package import.

``python3 docs/hooks/module_map.py`` prints the markdown.
"""

from __future__ import annotations

import ast
import logging
import re
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.module_map")

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "strands_robots"
_GRADER = _REPO / "scripts" / "check_import_layers.py"
_TOKEN = re.compile(r"\{\{\s*module_map\s*\}\}")
_TOTAL = re.compile(r"\{\{\s*module_map_total\s*\}\}")


def _layers() -> list[tuple[str, tuple[str, ...]]]:
    """``LAYERS`` from the grader, as (name, members) pairs."""
    tree = ast.parse(_GRADER.read_text(encoding="utf-8"))
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == "LAYERS" and node.value is not None:
            value = ast.literal_eval(node.value)
            return [(str(name), tuple(str(m) for m in members)) for name, members in value]
    raise RuntimeError(f"LAYERS not found in {_GRADER}")


def _loc(member: str) -> int:
    path = _PKG / member
    if path.is_dir():
        return sum(len(p.read_text(encoding="utf-8").splitlines()) for p in path.rglob("*.py"))
    file = path.with_suffix(".py")
    return len(file.read_text(encoding="utf-8").splitlines()) if file.exists() else 0


@lru_cache(maxsize=1)
def rows() -> list[tuple[str, list[tuple[str, int]], int]]:
    """(layer, [(member, loc)], layer total) in DAG order, lowest layer first."""
    out = []
    for name, members in _layers():
        counted = sorted(((m, _loc(m)) for m in members), key=lambda mc: -mc[1])
        out.append((name, counted, sum(c for _, c in counted)))
    return out


def total() -> int:
    """Lines across every ``*.py`` under the package."""
    return sum(len(p.read_text(encoding="utf-8").splitlines()) for p in _PKG.rglob("*.py"))


def render() -> str:
    """The table as markdown."""
    lines = ["| layer | members (lines) | lines |", "|---|---|---|"]
    for index, (name, members, subtotal) in enumerate(rows()):
        shown = ", ".join(f"`{m}` {c:,}" for m, c in members if not m.startswith("_") or c > 0)
        lines.append(f"| {index} `{name}` | {shown} | {subtotal:,} |")
    lines.append(f"| | **package** | **{total():,}** |")
    return "\n".join(lines)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001
    """mkdocs hook entry point: expand ``{{module_map}}`` and ``{{module_map_total}}``."""
    if _TOKEN.search(markdown):
        markdown = _TOKEN.sub(lambda _: render(), markdown)
    if _TOTAL.search(markdown):
        markdown = _TOTAL.sub(lambda _: f"{total():,}", markdown)
    return markdown


if __name__ == "__main__":
    print(render())
