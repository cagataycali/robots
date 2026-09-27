"""mkdocs hook: every environment variable the package reads, as a generated table.

``{{env_vars}}`` on a page becomes one table per prefix (STRANDS_MESH_, STRANDS_,
MUJOCO_, HF_, ..., then everything else) with: variable, the module that reads
it, the default the code passes when it is a literal, and a one-line meaning
taken from the nearest docstring or comment that names the variable. When no
prose names it the meaning column says "see module".

How reads are found (``ast`` over the source, no import):

1. ``os.environ.get("X")``, ``os.getenv("X")``, ``os.environ["X"]``,
   ``os.environ.setdefault("X", ...)`` and ``os.environ.pop("X")`` with a string
   literal, or with a module-level constant that holds one.
2. Helper wrappers: a function whose body performs one of those reads on one of
   its own parameters (``def _env_flag(name): ... os.getenv(name)``). Every call
   to that helper with a string literal counts as a read of that variable.

``python3 docs/hooks/env_vars.py`` prints the markdown.
"""

from __future__ import annotations

import ast
import html
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.env_vars")

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "strands_robots"
_TOKEN = re.compile(r"^\{\{\s*env_vars\s*\}\}\s*$", re.M)
_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")
_PREFIXES = ("STRANDS_MESH_", "STRANDS_", "DEVICE_CONNECT_", "MUJOCO_", "HF_", "ZENOH_", "ROS_", "AWS_")
_READERS = {"get", "getenv", "setdefault", "pop", "__getitem__"}


@dataclass
class Read:
    """One environment variable and where the package reads it."""

    name: str
    modules: set[str]
    defaults: set[str]
    meaning: str = ""


def _is_environ(node: ast.expr) -> bool:
    """``os.environ`` or a bare ``environ`` name."""
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (isinstance(node, ast.Name) and node.id == "environ")


def _read_target(node: ast.AST) -> tuple[ast.expr, ast.expr | None] | None:
    """(key expr, default expr) when ``node`` is an environment read, else None."""
    if isinstance(node, ast.Subscript) and _is_environ(node.value):
        return node.slice, None
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        if func.attr == "getenv" and isinstance(func.value, ast.Name) and func.value.id == "os" and node.args:
            return node.args[0], node.args[1] if len(node.args) > 1 else None
        if func.attr in _READERS and _is_environ(func.value) and node.args:
            return node.args[0], node.args[1] if len(node.args) > 1 else None
    if isinstance(func, ast.Name) and func.id == "getenv" and node.args:
        return node.args[0], node.args[1] if len(node.args) > 1 else None
    return None


def _constants(tree: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in tree.body:
        value = getattr(node, "value", None)
        if isinstance(node, ast.Assign | ast.AnnAssign) and isinstance(value, ast.Constant) and isinstance(value.value, str):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    out[target.id] = value.value
    return out


def _literal(node: ast.expr | None, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    return None


def _default_text(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return repr(node.value) if node.value is not None else None
    return None


def _helpers(tree: ast.Module) -> set[str]:
    """Names of functions that read the environment through one of their parameters."""
    out: set[str] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        params = {a.arg for a in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]}
        for node in ast.walk(fn):
            target = _read_target(node)
            if target and isinstance(target[0], ast.Name) and target[0].id in params:
                out.add(fn.name)
                break
    return out


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(_REPO).with_suffix("").parts)


def _prose_blocks(text: str, tree: ast.Module) -> list[str]:
    """Docstrings and comment blocks of one module, each as one paragraph."""
    blocks: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            doc = ast.get_docstring(node, clean=True)
            if doc:
                blocks.extend(" ".join(part.split()) for part in doc.split("\n\n"))
    comment: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            comment.append(stripped.lstrip("#").strip())
        elif comment:
            blocks.append(" ".join(comment))
            comment = []
    if comment:
        blocks.append(" ".join(comment))
    return blocks


def _meaning(text: str, tree: ast.Module, name: str) -> str:
    """The sentence in a docstring or comment that names the variable, best match first."""
    best = ""
    for block in _prose_blocks(text, tree):
        if name not in block:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z`(])", block):
            if name not in sentence:
                continue
            prose = re.sub(r"``([^`]+)``", r"`\1`", sentence).strip()
            prose = re.sub(r":[a-z:]+:`~?([^`]+)`", r"`\1`", prose)
            if len(prose) > 170:
                prose = prose[:167].rstrip(",;: ") + "..."
            if prose.lstrip("`").startswith(name):
                return prose
            best = best or prose
    return best


@lru_cache(maxsize=1)
def reads() -> dict[str, Read]:
    """Every environment variable read, keyed by name."""
    trees = {path: ast.parse(path.read_text(encoding="utf-8")) for path in sorted(_PKG.rglob("*.py"))}
    helper_names: set[str] = set()
    for tree in trees.values():
        helper_names |= _helpers(tree)
    out: dict[str, Read] = {}

    def record(name: str, module: str, default: str | None) -> None:
        if not _NAME.match(name):
            return
        entry = out.setdefault(name, Read(name=name, modules=set(), defaults=set()))
        entry.modules.add(module)
        if default:
            entry.defaults.add(default)

    for path, tree in trees.items():
        constants = _constants(tree)
        module = _module_name(path)
        for node in ast.walk(tree):
            target = _read_target(node)
            if target:
                name = _literal(target[0], constants)
                if name:
                    record(name, module, _default_text(target[1]))
                continue
            if isinstance(node, ast.Call):
                func = node.func
                fname = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if fname in helper_names and node.args:
                    name = _literal(node.args[0], constants)
                    if name:
                        record(name, module, _default_text(node.args[1] if len(node.args) > 1 else None))
    by_module = {_module_name(p): (p.read_text(encoding="utf-8"), tree) for p, tree in trees.items()}
    for entry in out.values():
        candidates = [_meaning(*by_module[m], entry.name) for m in sorted(entry.modules)]
        candidates = [c for c in candidates if c]
        entry.meaning = next((c for c in candidates if c.lstrip("`").startswith(entry.name)), candidates[0] if candidates else "")
    return out


def _group(name: str) -> str:
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            return prefix
    return "other"


def _code(text: str) -> str:
    return "<code>" + html.escape(text, quote=False).replace("|", "&#124;") + "</code>"


def _cell(text: str) -> str:
    parts = html.escape(text, quote=False).split("`")
    if len(parts) % 2 == 0:
        return "`".join(p.replace("|", r"\|") for p in parts)
    return "".join(_code(p) if i % 2 else p.replace("|", r"\|") for i, p in enumerate(parts))


def render() -> str:
    """The whole table set as markdown."""
    entries = reads()
    groups: dict[str, list[Read]] = defaultdict(list)
    for entry in entries.values():
        groups[_group(entry.name)].append(entry)
    order = [*_PREFIXES, "other"]
    lines = [f"{len(entries)} variables read by the package at this commit.", ""]
    for key in order:
        items = sorted(groups.get(key, []), key=lambda e: e.name)
        if not items:
            continue
        title = f"`{key}*`" if key != "other" else "Everything else"
        lines += [f"## {title}", "", "| variable | read in | default | meaning |", "|---|---|---|---|"]
        for e in items:
            modules = ", ".join(_code(m.removeprefix("strands_robots.")) for m in sorted(e.modules)[:3])
            if len(e.modules) > 3:
                modules += f" and {len(e.modules) - 3} more"
            default = ", ".join(_code(d) for d in sorted(e.defaults)) if e.defaults else "unset"
            meaning = _cell(e.meaning) if e.meaning else f"see {_code(sorted(e.modules)[0].removeprefix('strands_robots.'))}"
            lines.append(f"| {_code(e.name)} | {modules} | {default} | {meaning} |")
        lines.append("")
    return "\n".join(lines)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001
    """mkdocs hook entry point: expand ``{{env_vars}}``."""
    if not _TOKEN.search(markdown):
        return markdown
    return _TOKEN.sub(lambda _: render(), markdown)


if __name__ == "__main__":
    print(render())
