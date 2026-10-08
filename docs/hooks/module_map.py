"""mkdocs hook: the layered module map, generated from the tree.

``{{module_map}}`` on a page becomes one table row per layer of the import DAG
that ``scripts/check_import_layers.py`` grades: the layer's name, what the layer
is for, and its top-level members (module or package under ``strands_robots/``),
public names first and the ``_private`` helpers folded into one chip. No line
counts: a reader sizing the package wants ``{{module_map_total}}`` (the package
total) or ``{{module_map:lines}}`` (the table with a count per member), the
architecture page wants to know where a module sits.

The layer list is read from ``LAYERS`` in that script with ``ast``, so the docs
cannot disagree with the grader about where a module sits. Lines are counted
over every ``*.py`` under the member (all lines, the same number ``wc -l``
prints). Filesystem only, no package import.

``python3 docs/hooks/module_map.py [--lines]`` prints the markdown.
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
_TOKEN_LINES = re.compile(r"\{\{\s*module_map:lines\s*\}\}")
_TOTAL = re.compile(r"\{\{\s*module_map_total\s*\}\}")

#: Layer name -> what the layer is for, one clause. A layer the grader adds and
#: this dict does not know renders with an empty cell rather than failing the build.
ROLES: dict[str, str] = {
    "core": "leaves everything imports: gates, audit, dataset formats, refusal codes, rendering helpers",
    "registry": "the robot and policy rows, and the asset paths they declare",
    "drivers|mesh": "talk to hardware (native drivers, ROS, teleop) and to other peers (Zenoh mesh)",
    "sim|policies": "the simulation backends, the policy providers, training and remote inference",
    "app": "what `Robot(...)` returns: the factory, the hardware lane, doctor and dataset verification",
    "tools": "the `@tool` surface an agent calls",
    "dashboard": "the operator UI, a mesh gateway over everything below",
}


def _layers() -> list[tuple[str, tuple[str, ...]]]:
    """``LAYERS`` from the grader, as (name, members) pairs."""
    tree = ast.parse(_GRADER.read_text(encoding="utf-8"))
    for node in tree.body:
        target = (
            node.target
            if isinstance(node, ast.AnnAssign)
            else (node.targets[0] if isinstance(node, ast.Assign) else None)
        )
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


def render(lines: bool = False) -> str:
    """The table as markdown: layer, role and members; ``lines=True`` adds a count per member."""
    if lines:
        out = ["| layer | members (lines) | lines |", "|---|---|---|"]
        for index, (name, members, subtotal) in enumerate(rows()):
            shown = ", ".join(f"`{m}` {c:,}" for m, c in members if not m.startswith("_") or c > 0)
            out.append(f"| {index} `{name}` | {shown} | {subtotal:,} |")
        out.append(f"| | **package** | **{total():,}** |")
        return "\n".join(out)
    out = ["| layer | what it is for | members |", "|---|---|---|"]
    for index, (name, members, _subtotal) in enumerate(rows()):
        # ``__main__`` is the ``python -m`` entry point, a public door; ``_x`` is a private helper.
        public = [m for m, _ in sorted(members) if not m.startswith("_") or m.startswith("__")]
        private = [m for m, c in sorted(members) if m.startswith("_") and not m.startswith("__") and c > 0]
        shown = ", ".join(f"`{m}`" for m in public)
        if private:
            noun = "private helper" if len(private) == 1 else "private helpers"
            shown += f" and {len(private)} {noun} (" + ", ".join(f"`{m}`" for m in private) + ")"
        out.append(f"| {index} `{name}` | {ROLES.get(name, '')} | {shown} |")
    return "\n".join(out)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001
    """mkdocs hook entry point: expand ``{{module_map}}`` and ``{{module_map_total}}``."""
    if _TOKEN_LINES.search(markdown):
        markdown = _TOKEN_LINES.sub(lambda _: render(lines=True), markdown)
    if _TOKEN.search(markdown):
        markdown = _TOKEN.sub(lambda _: render(), markdown)
    if _TOTAL.search(markdown):
        markdown = _TOTAL.sub(lambda _: f"{total():,}", markdown)
    return markdown


if __name__ == "__main__":
    import sys

    print(render(lines="--lines" in sys.argv[1:]))
