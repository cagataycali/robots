"""mkdocs hook: the driver coverage matrix, generated from the two registries.

``{{coverage_matrix}}`` in a page becomes one row per registered robot: its
category, the ``driver=`` values that can build it for real, and the asset
directory its simulation scene loads. The join is the one
:func:`~strands_robots.drivers.list_driver_coverage` derives - a robot is
reachable through lerobot when the package registry declares a
``hardware.lerobot_type``, through a native driver when one is registered for
it, through both, or through neither - and the last group is the answer no
single registry holds, because it is defined by two absences.

Both halves are read out of the source so the page cannot drift from the code:
the registry from ``strands_robots/registry/robots.json``, the native drivers
from the ``_SHIPPED_DRIVERS`` table in ``strands_robots/drivers/__init__.py``
and the ``SUPPORTED_ROBOTS`` tuple each driver module declares.
``tests/test_docs_coverage_matrix_hook.py`` compares every published row against
the live join, so a driver registered tomorrow widens both at once.

Filesystem only (no ``strands_robots`` import), so the hook runs in the docs
venv without the package's optional extras.
"""

from __future__ import annotations

import ast
import collections
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.coverage_matrix")

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "strands_robots"
_TOKEN = re.compile(r"^\{\{\s*coverage_matrix\s*\}\}\s*$", re.M)

#: The table the drivers package registers from, and the attribute each entry
#: may name instead of listing its robots inline.
_DRIVER_TABLE = "_SHIPPED_DRIVERS"

#: An empty cell: this registry has nothing for that robot.
_NONE = "-"


@dataclass(frozen=True)
class Row:
    """One registered robot and what can build it."""

    name: str
    category: str
    lerobot_type: str | None
    native_driver: str | None
    asset_dir: str | None

    @property
    def drivers(self) -> tuple[str, ...]:
        """The ``driver=`` values that can build this robot, sorted as the join sorts them."""
        out: list[str] = []
        if self.lerobot_type is not None:
            out.append("lerobot")
        if self.native_driver is not None:
            out.append("strands")
        return tuple(out)


def _module_path(dotted: str) -> Path:
    """File holding a dotted module path, module or package."""
    base = _REPO.joinpath(*dotted.split("."))
    module = base.with_suffix(".py")
    return module if module.is_file() else base / "__init__.py"


def _literal(path: Path, name: str) -> object:
    """Value of a module-scope literal assignment, read with :mod:`ast`."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name and node.value is not None:
                return ast.literal_eval(node.value)
    raise ValueError(f"{path.relative_to(_REPO)} declares no literal {name}")


@lru_cache(maxsize=1)
def native_drivers() -> dict[str, str]:
    """Robot name -> the driver class registered for it by this package.

    Names are as the driver table spells them, so an alias still has to be
    resolved against the registry before the row is built.
    """
    table = _literal(_module_path("strands_robots.drivers"), _DRIVER_TABLE)
    out: dict[str, str] = {}
    for module, class_name, names in table:  # type: ignore[union-attr]
        robots = _literal(_module_path(module), names) if isinstance(names, str) else names
        for robot in robots:  # type: ignore[union-attr]
            out[str(robot)] = str(class_name)
    return out


@lru_cache(maxsize=1)
def rows() -> tuple[Row, ...]:
    """Every registered robot, grouped by category then name."""
    registry = json.loads((_PKG / "registry" / "robots.json").read_text(encoding="utf-8"))["robots"]
    canonical = {alias: name for name, spec in registry.items() for alias in spec.get("aliases", ())}
    native: dict[str, str] = {}
    for robot, class_name in native_drivers().items():
        native[canonical.get(robot, robot)] = class_name
    out = [
        Row(
            name=name,
            category=str(spec.get("category", "")),
            lerobot_type=spec.get("hardware", {}).get("lerobot_type"),
            native_driver=native.get(name),
            asset_dir=spec.get("asset", {}).get("dir"),
        )
        for name, spec in registry.items()
    ]
    return tuple(sorted(out, key=lambda row: (row.category, row.name)))


def _summary() -> str:
    """Per-category counts of what can be driven, and what cannot."""
    per_category: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in rows():
        counter = per_category[row.category]
        counter["robots"] += 1
        counter["lerobot"] += row.lerobot_type is not None
        counter["native"] += row.native_driver is not None
        counter["neither"] += not row.drivers
    lines = [
        "| Category | Robots | lerobot | Native driver | Neither |",
        "|----------|-------:|--------:|--------------:|--------:|",
    ]
    total: collections.Counter[str] = collections.Counter()
    for category, counter in sorted(per_category.items()):
        total.update(counter)
        lines.append(
            f"| {category} | {counter['robots']} | {counter['lerobot']} | {counter['native']} | {counter['neither']} |"
        )
    lines.append(
        f"| **Total** | **{total['robots']}** | **{total['lerobot']}** | "
        f"**{total['native']}** | **{total['neither']}** |"
    )
    return "\n".join(lines)


def _matrix() -> str:
    """One row per registered robot."""
    lines = [
        '| Robot | Category | `driver="lerobot"` | `driver="strands"` | Sim asset |',
        "|-------|----------|--------------------|--------------------|-----------|",
    ]
    for row in rows():
        lerobot = f"`{row.lerobot_type}`" if row.lerobot_type else _NONE
        native = f"`{row.native_driver}`" if row.native_driver else _NONE
        asset = f"`{row.asset_dir}`" if row.asset_dir else _NONE
        lines.append(f"| `{row.name}` | {row.category} | {lerobot} | {native} | {asset} |")
    return "\n".join(lines)


def render() -> str:
    """The whole generated block: the per-category summary, then every robot."""
    return f"{_summary()}\n\n{_matrix()}\n"


def substitute(markdown: str, page_path: str = "<string>") -> str:
    """Replace a ``{{coverage_matrix}}`` line with the generated tables."""
    if not _TOKEN.search(markdown):
        return markdown
    log.debug("%s: rendering the coverage matrix for %d robots", page_path, len(rows()))
    return _TOKEN.sub(lambda _: render(), markdown)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001 - mkdocs signature
    """Render the matrix into one page, the hook entry point mkdocs calls."""
    return substitute(markdown, page.file.src_path)
