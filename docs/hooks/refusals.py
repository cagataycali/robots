"""mkdocs hook: the refusal-code table, generated from ``strands_robots/refusal_codes.py``.

``{{refusals}}`` on a page becomes one table row per code in ``REFUSAL_CODES``:
the code, its meaning (the ``#:`` comment block above the constant, with the
``Subject:`` and ``Grant:`` sentences kept), the operator grant from
``REFUSAL_GRANTS``, and every raise site in the package (``code=refusal_codes.X``
or ``code=X`` after an import), as ``module:line``. A second table lists the
exception classes in ``recording_errors.py`` the same way.

Filesystem only, ``ast`` plus a line scan. ``python3 docs/hooks/refusals.py``
prints the markdown.
"""

from __future__ import annotations

import ast
import html
import logging
import re
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.refusals")

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "strands_robots"
_CODES = _PKG / "refusal_codes.py"
_RECORDING = _PKG / "recording_errors.py"
_TOKEN = re.compile(r"^\{\{\s*refusals\s*\}\}\s*$", re.M)
_ROLE = re.compile(r":[a-z:]+:`~?([^`]+)`")
_RST = re.compile(r"``([^`]+)``")


def _prose(text: str) -> str:
    text = _ROLE.sub(lambda m: f"`{m.group(1).split('.')[-1]}`", text)
    text = _RST.sub(lambda m: f"`{m.group(1)}`", text)
    return " ".join(text.replace(" -- ", ", ").split())


def _code(text: str) -> str:
    return "<code>" + html.escape(text, quote=False).replace("|", "&#124;") + "</code>"


def _cell(text: str) -> str:
    parts = html.escape(text, quote=False).split("`")
    if len(parts) % 2 == 0:
        return "`".join(p.replace("|", r"\|") for p in parts)
    return "".join(_code(p) if i % 2 else p.replace("|", r"\|") for i, p in enumerate(parts))


def _doc_comments(source: str) -> dict[str, str]:
    """Constant name -> the ``#:`` block right above its assignment."""
    out: dict[str, str] = {}
    block: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#:"):
            block.append(stripped[2:].strip())
            continue
        match = re.match(r"^([A-Z][A-Z0-9_]+)\s*(?::[^=]+)?=", line)
        if match and block:
            out[match.group(1)] = _prose(" ".join(block))
        if not stripped.startswith("#"):
            block = []
    return out


def _string_tuple(tree: ast.Module, name: str, constants: dict[str, str]) -> tuple[str, ...]:
    for node in tree.body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(t, ast.Name) and t.id == name for t in targets) and isinstance(node.value, ast.Tuple):
            return tuple(
                constants.get(e.id, e.id) if isinstance(e, ast.Name) else str(e.value) for e in node.value.elts
            )
    return ()


def _string_dict(tree: ast.Module, name: str, constants: dict[str, str]) -> dict[str, str]:
    for node in tree.body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(t, ast.Name) and t.id == name for t in targets) and isinstance(node.value, ast.Dict):
            out: dict[str, str] = {}
            for k, v in zip(node.value.keys, node.value.values, strict=True):
                key = constants.get(k.id, k.id) if isinstance(k, ast.Name) else str(getattr(k, "value", k))
                out[key] = str(v.value) if isinstance(v, ast.Constant) else ast.unparse(v)
            return out
    return {}


def _module(path: Path) -> str:
    return ".".join(path.relative_to(_REPO).with_suffix("").parts).removeprefix("strands_robots.")


@lru_cache(maxsize=1)
def raise_sites() -> dict[str, list[str]]:
    """Code -> ``module:line`` for every ``code=`` keyword naming it (outside refusal_codes.py)."""
    out: dict[str, list[str]] = {}
    for path in sorted(_PKG.rglob("*.py")):
        if path == _CODES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "code":
                    continue
                value = kw.value
                name = value.attr if isinstance(value, ast.Attribute) else getattr(value, "id", None)
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    name = value.value
                if name and re.match(r"^[A-Z][A-Z0-9_]+$", name):
                    out.setdefault(name, []).append(f"{_module(path)}:{node.lineno}")
    return out


def render() -> str:
    """Both tables as markdown."""
    source = _CODES.read_text(encoding="utf-8")
    tree = ast.parse(source)
    constants = {
        t.id: n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    comments = _doc_comments(source)
    codes = _string_tuple(tree, "REFUSAL_CODES", constants)
    grants = _string_dict(tree, "REFUSAL_GRANTS", constants)
    sites = raise_sites()
    lines = [
        f"{len(codes)} codes in `REFUSAL_CODES`. A refusal carries `exc.code` and `exc.subject`; the grant column is the "
        "environment variable in `REFUSAL_GRANTS` that lifts it.",
        "",
        "| code | meaning | grant | raised in |",
        "|---|---|---|---|",
    ]
    for code in codes:
        where = ", ".join(_code(s) for s in sites.get(code, [])) or "no raise site found"
        lines.append(f"| {_code(code)} | {_cell(comments.get(code, ''))} | {_code(grants.get(code, ''))} | {where} |")
    lines += ["", "## Recording errors", ""]
    rec_tree = ast.parse(_RECORDING.read_text(encoding="utf-8"))
    lines += ["| class | base | meaning |", "|---|---|---|"]
    for node in rec_tree.body:
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            doc = (ast.get_docstring(node, clean=True) or "").split("\n\n")
            meaning = _prose(" ".join(doc[:2]))
            lines.append(f"| {_code(node.name)} | {_code(bases)} | {_cell(meaning)} |")
    lines.append("")
    return "\n".join(lines)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001
    """mkdocs hook entry point: expand ``{{refusals}}``."""
    if not _TOKEN.search(markdown):
        return markdown
    return _TOKEN.sub(lambda _: render(), markdown)


if __name__ == "__main__":
    print(render())
