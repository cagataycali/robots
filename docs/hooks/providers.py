"""mkdocs hook: policy provider tables the docs never type by hand.

Tokens replaced at build time (same shape as ``facts.py``):

* ``{{providers:table}}``           one row per provider in
                                    ``strands_robots/registry/policies.json``:
                                    name, class, install extra, spellings,
                                    what it drives, whether a trainer exists
* ``{{providers:kwargs:<name>}}``   the constructor keywords of that provider's
                                    policy class, read from the source with
                                    ``ast`` (name, annotation, default)
* ``{{providers:cameras}}``         the lerobot_local embodiment camera table
                                    from ``policies/lerobot_local/embodiments.json``
                                    (``obs_rename``: runtime camera key to the
                                    model's image feature)
* ``{{providers:embodiments}}``     every lerobot_local embodiment name plus its
                                    aliases, one comma separated line

Reads files only. Never imports ``strands_robots``: the docs build runs in a
venv without the package's optional extras. An unknown token warns, which
``mkdocs build --strict`` turns into a failed build.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import tomllib
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.providers")

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "strands_robots"
_TOKEN = re.compile(r"\{\{\s*providers:([a-z_]+)(?::([a-z0-9_]+))?\s*\}\}")

# Which pyproject extra installs a provider's client side. policies.json names
# an ``extra`` only for lerobot_local; the rest are matched here by name and
# every value is checked against ``[project.optional-dependencies]`` at build.
_EXTRA_FOR: dict[str, str] = {
    "groot": "groot-service",
    "cosmos3": "cosmos3-service",
    "moveit2": "moveit2",
    "curobo": "curobo",
    "wbc": "wbc",
    "wbc_gait": "wbc",
    "wbc_latent": "wbc",
    "kimodo": "kimodo",
    "protomotions": "protomotions",
    "microduck": "microduck",
    "rl": "rl",
    "remote": "inference",
}

_PAGE_FOR: dict[str, str] = {
    "lerobot_local": "lerobot-local.md",
    "wbc_gait": "wbc.md",
    "wbc_latent": "wbc-latent.md",
}


@lru_cache(maxsize=1)
def _providers() -> dict[str, dict]:
    return json.loads((_PKG / "registry" / "policies.json").read_text(encoding="utf-8"))["providers"]


@lru_cache(maxsize=1)
def _extras() -> dict[str, list[str]]:
    with (_REPO / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["optional-dependencies"]


def _extra_cell(name: str, spec: dict) -> str:
    extra = spec.get("extra") or _EXTRA_FOR.get(name)
    if extra is None:
        return "none"
    if extra not in _extras():
        log.warning("providers hook: extra [%s] for %s is not in pyproject.toml", extra, name)
    if not _extras()[extra]:
        return f"`[{extra}]` (empty: install cuRobo yourself)" if name == "curobo" else f"`[{extra}]`"
    return f"`[{extra}]`"


def _spellings(spec: dict) -> str:
    names: list[str] = []
    for key in ("shorthands", "aliases"):
        for alias in spec.get(key, ()):
            if alias not in names:
                names.append(alias)
    return ", ".join(f"`{n}`" for n in names) or "none"


def table() -> str:
    """The provider matrix as markdown, one row per registry provider."""
    rows = [
        "| provider | class | install extra | also spelled | what it drives | trainer |",
        "|---|---|---|---|---|---|",
    ]
    for name, spec in _providers().items():
        page = _PAGE_FOR.get(name, f"{name}.md")
        link = f"[`{name}`]({page})" if (_REPO / "docs" / "learn" / "policies" / page).exists() else f"`{name}`"
        trainer = "yes" if spec.get("trainer") else "no"
        desc = spec["description"].replace("\u2014", ",").replace("|", "/")
        rows.append(
            f"| {link} | `{spec['class']}` | {_extra_cell(name, spec)} | {_spellings(spec)} | {desc} | {trainer} |"
        )
    return "\n".join(rows)


def _module_constants(tree: ast.Module, depth: int = 0) -> dict[str, ast.expr]:
    """Module-level ``NAME = <literal>`` bindings, following ``from x import NAME`` one hop into the package."""
    out: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign | ast.AnnAssign) and isinstance(node.value, ast.Constant):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            out.update({t.id: node.value for t in targets if isinstance(t, ast.Name)})
        elif isinstance(node, ast.ImportFrom) and node.module and depth == 0 and node.level == 0:
            path = _PKG.parent / Path(*node.module.split("."))
            path = path.with_suffix(".py") if path.with_suffix(".py").exists() else path / "__init__.py"
            if not path.exists():
                continue
            imported = _module_constants(ast.parse(path.read_text(encoding="utf-8")), depth + 1)
            for alias in node.names:
                if alias.name in imported:
                    out[alias.asname or alias.name] = imported[alias.name]
    return out


def _class_init(module: str, cls: str) -> tuple[ast.FunctionDef | None, list[str]]:
    """Locate ``cls.__init__`` under ``module`` (a package dir or a .py file).

    A keyword default that is a bare name (``timeout=DEFAULT_TIMEOUT``) is replaced by
    the literal the module binds that name to, so the table states the value, not
    the constant's name.
    """
    root = _PKG.parent / Path(*module.split("."))
    files = [root.with_suffix(".py")] if root.with_suffix(".py").exists() else sorted(root.rglob("*.py"))
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                bases = [ast.unparse(b) for b in node.bases]
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        constants = _module_constants(tree)
                        args = item.args
                        args.defaults = [_resolved(d, constants) for d in args.defaults]
                        args.kw_defaults = [None if d is None else _resolved(d, constants) for d in args.kw_defaults]
                        return item, bases
                return None, bases
    return None, []


def _resolved(default: ast.expr, constants: dict[str, ast.expr]) -> ast.expr:
    return constants.get(default.id, default) if isinstance(default, ast.Name) else default


def kwargs_table(name: str) -> str:
    """The constructor keywords of one provider as a markdown table."""
    spec = _providers().get(name)
    if spec is None:
        log.warning("providers hook: unknown provider %s", name)
        return f"{{{{providers:kwargs:{name}}}}}"
    init, _bases = _class_init(spec["module"], spec["class"])
    if init is None:
        log.warning("providers hook: no __init__ found for %s", name)
        return ""
    rows = ["| keyword | type | default |", "|---|---|---|"]
    args = init.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(positional, defaults, strict=True):
        if arg.arg == "self":
            continue
        rows.append(_row(arg, default, required=default is None))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        rows.append(_row(arg, default, required=default is None))
    if args.kwarg is not None:
        rows.append(f"| `**{args.kwarg.arg}` | | unknown keywords are ignored |")
    else:
        rows.append("| | | no `**kwargs`: an unknown keyword is a `TypeError` |")
    return "\n".join(rows)


def _row(arg: ast.arg, default: ast.expr | None, *, required: bool) -> str:
    annotation = ast.unparse(arg.annotation).replace("|", "\\|") if arg.annotation is not None else ""
    if required:
        value = "required"
    else:
        value = f"`{ast.unparse(default)}`".replace("|", "\\|")
    return f"| `{arg.arg}` | `{annotation}` | {value} |" if annotation else f"| `{arg.arg}` | | {value} |"


@lru_cache(maxsize=1)
def _embodiments() -> dict:
    return json.loads((_PKG / "policies" / "lerobot_local" / "embodiments.json").read_text(encoding="utf-8"))


def cameras_table() -> str:
    """The camera-key table for lerobot policy types as markdown."""
    data = _embodiments()
    aliases: dict[str, list[str]] = {}
    for alias, target in data["aliases"].items():
        aliases.setdefault(target, []).append(alias)
    rows = ["| embodiment | camera key you attach | model image feature | aliases |", "|---|---|---|---|"]
    for name, cfg in data["configs"].items():
        rename = cfg.get("obs_rename")
        if rename is None and "_extends" in cfg:
            rename = data["configs"][cfg["_extends"]].get("obs_rename")
        if not rename:
            continue
        keys = "<br>".join(f"`{k}`" for k in rename)
        feats = "<br>".join(f"`{v}`" for v in rename.values())
        alias_cell = ", ".join(f"`{a}`" for a in aliases.get(name, ())) or ""
        rows.append(f"| `{name}` | {keys} | {feats} | {alias_cell} |")
    return "\n".join(rows)


def embodiments_line() -> str:
    """The one-line list of GR00T embodiment tags."""
    data = _embodiments()
    names = list(data["configs"]) + list(data["aliases"])
    return ", ".join(f"`{n}`" for n in names)


def substitute(markdown: str, page_path: str = "<string>") -> str:
    """Expand every ``{{providers:...}}`` token in ``markdown``."""

    def _one(match: re.Match[str]) -> str:
        kind, arg = match.group(1), match.group(2)
        if kind == "table":
            return table()
        if kind == "kwargs" and arg:
            return kwargs_table(arg)
        if kind == "cameras":
            return cameras_table()
        if kind == "embodiments":
            return embodiments_line()
        log.warning("%s: unknown providers token %s", page_path, match.group(0))
        return match.group(0)

    return _TOKEN.sub(_one, markdown)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001 - mkdocs signature
    """mkdocs hook entry point: expand the providers tokens."""
    return substitute(markdown, page.file.src_path)


if __name__ == "__main__":
    print(table())
    print()
    print(kwargs_table("wbc"))
    print()
    print(cameras_table())
