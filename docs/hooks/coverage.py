"""mkdocs hook: the robot coverage matrix, generated from the source.

``{{coverage_matrix}}`` in a page becomes two tables: a per-family summary and
one row per registered robot with the ``driver=`` values that build it for
real, the asset directory its simulation loads, and the policy providers
written for that body.

Every cell is witnessed by a file at this commit, never typed:

* the robot registry, ``strands_robots/registry/robots.json`` and the
  ``robot_descriptions`` URDF tail in ``urdf_robots.json`` (see ``registry_view.py``);
* the native drivers, from the ``_SHIPPED_DRIVERS`` table in
  ``strands_robots/drivers/__init__.py`` and the ``SUPPORTED_ROBOTS`` tuple each
  driver module declares (the same join ``list_driver_coverage`` makes);
* the policy providers, ``strands_robots/registry/policies.json``;
* the body a provider is bound to, from a literal in that provider's own
  module (see :data:`EMBODIMENT_WITNESSES`). A witness the source no longer
  contains is logged as a warning, which ``mkdocs build --strict`` turns into a
  failed build, so a provider that widens or moves cannot leave a stale cell.

Filesystem only: this module never imports ``strands_robots``, so the docs
build runs without the package or its optional extras.
"""

from __future__ import annotations

import ast
import collections
import importlib.util
import json
import logging
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.coverage")

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "strands_robots"
_TOKEN = re.compile(r"^\{\{\s*coverage_matrix\s*\}\}\s*$", re.M)

_DRIVER_TABLE = "_SHIPPED_DRIVERS"
_NONE = "-"

#: Providers written for one body, and the line in their own module that says
#: so. ``(provider, robot names, module path, literal that must be present)``.
#: The literal is the provider's own refusal or joint table naming the robot;
#: :func:`check_witnesses` greps for it at build time.
EMBODIMENT_WITNESSES: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    ("wbc", ("unitree_g1",), "policies/wbc/policy.py", "WBC_G1_ALL_JOINTS"),
    ("wbc_gait", ("unitree_g1",), "policies/wbc/gait.py", "single_obs_dim=95 for the G1"),
    ("holosoma", ("unitree_g1",), "policies/holosoma/policy.py", "load unitree_g1"),
    ("kimodo", ("unitree_g1",), "policies/kimodo/policy.py", "load the full unitree_g1 model"),
    ("protomotions", ("unitree_g1",), "policies/protomotions/policy.py", "load unitree_g1"),
    ("microduck", ("microduck",), "policies/microduck/policy.py", 'return "microduck"'),
)


@dataclass(frozen=True)
class Row:
    """One registered robot and what can build or drive it."""

    name: str
    category: str
    lerobot_type: str | None
    native_driver: str | None
    default_driver: str
    asset_dir: str | None
    policies: tuple[str, ...]

    @property
    def drivers(self) -> tuple[str, ...]:
        """The ``driver=`` values that can build this robot for real."""
        out: list[str] = []
        if self.lerobot_type is not None:
            out.append("lerobot")
        if self.native_driver is not None:
            out.append("strands")
        return tuple(out)

    @property
    def real(self) -> bool:
        """Whether any driver reaches hardware for this robot."""
        return bool(self.drivers)


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
def registry() -> dict[str, dict]:
    """The robot registry, read once per build: ``robots.json`` plus the URDF tail."""
    return dict(_registry_view().merged())


@lru_cache(maxsize=1)
def providers() -> dict[str, dict]:
    """The policy provider registry, read once per build."""
    return json.loads((_PKG / "registry" / "policies.json").read_text(encoding="utf-8"))["providers"]


@lru_cache(maxsize=1)
def native_drivers() -> dict[str, str]:
    """Canonical robot name -> the native driver class registered for it."""
    table = _literal(_module_path("strands_robots.drivers"), _DRIVER_TABLE)
    canonical = {alias: name for name, spec in registry().items() for alias in spec.get("aliases", ())}
    out: dict[str, str] = {}
    for module, class_name, names in table:  # type: ignore[union-attr]
        robots = _literal(_module_path(module), names) if isinstance(names, str) else names
        for robot in robots:  # type: ignore[union-attr]
            out[canonical.get(str(robot), str(robot))] = str(class_name)
    return out


@lru_cache(maxsize=1)
def driver_modules() -> dict[str, str]:
    """Native driver class name -> the dotted module that defines it."""
    table = _literal(_module_path("strands_robots.drivers"), _DRIVER_TABLE)
    return {str(class_name): str(module) for module, class_name, _names in table}  # type: ignore[union-attr]


def check_witnesses() -> list[str]:
    """Every embodiment witness that the source no longer contains."""
    missing: list[str] = []
    known = providers()
    for provider, robots, rel, literal in EMBODIMENT_WITNESSES:
        path = _PKG / rel
        if provider not in known:
            missing.append(f"{provider}: not in registry/policies.json")
        elif not path.is_file() or literal not in path.read_text(encoding="utf-8"):
            missing.append(f"{provider}: {rel} no longer contains {literal!r}")
        for robot in robots:
            if robot not in registry():
                missing.append(f"{provider}: robot {robot!r} not in registry/robots.json")
    return missing


@lru_cache(maxsize=1)
def embodiment_policies() -> dict[str, tuple[str, ...]]:
    """Canonical robot name -> providers written for that body, in registry order."""
    order = list(providers())
    out: dict[str, list[str]] = collections.defaultdict(list)
    for provider, robots, _rel, _literal in EMBODIMENT_WITNESSES:
        for robot in robots:
            out[robot].append(provider)
    return {robot: tuple(sorted(set(items), key=order.index)) for robot, items in out.items()}


@lru_cache(maxsize=1)
def rows() -> tuple[Row, ...]:
    """Every registered robot, grouped by family then name."""
    native = native_drivers()
    bound = embodiment_policies()
    out = []
    for name, spec in registry().items():
        hardware = spec.get("hardware") or {}
        # The precedence of strands_robots.drivers.resolve_driver with no driver=:
        # a declared driver, else the native one when registered, else lerobot.
        declared = hardware.get("driver")
        default = declared if declared not in (None, "auto") else ("strands" if name in native else "lerobot")
        out.append(
            Row(
                name=name,
                category=str(spec.get("category", "")),
                lerobot_type=hardware.get("lerobot_type"),
                native_driver=native.get(name),
                default_driver=default,
                asset_dir=(spec.get("asset") or {}).get("dir"),
                policies=bound.get(name, ()),
            )
        )
    return tuple(sorted(out, key=lambda row: (row.category, row.name)))


def row(name: str) -> Row:
    """The coverage row for one canonical robot name."""
    for item in rows():
        if item.name == name:
            return item
    raise KeyError(name)


def _summary() -> str:
    per_category: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for item in rows():
        counter = per_category[item.category]
        counter["robots"] += 1
        counter["sim"] += item.asset_dir is not None
        counter["lerobot"] += item.lerobot_type is not None
        counter["native"] += item.native_driver is not None
        counter["neither"] += not item.drivers
    lines = [
        "| Family | Robots | Sim | lerobot | Native driver | No driver |",
        "|--------|-------:|----:|--------:|--------------:|----------:|",
    ]
    total: collections.Counter[str] = collections.Counter()
    for category, counter in sorted(per_category.items()):
        total.update(counter)
        lines.append(
            f"| `{category}` | {counter['robots']} | {counter['sim']} | {counter['lerobot']} | "
            f"{counter['native']} | {counter['neither']} |"
        )
    lines.append(
        f"| **Total** | **{total['robots']}** | **{total['sim']}** | **{total['lerobot']}** | "
        f"**{total['native']}** | **{total['neither']}** |"
    )
    return "\n".join(lines)


def _matrix() -> str:
    lines = [
        '| Robot | Family | `driver="lerobot"` | `driver="strands"` | Sim asset | Body-bound policies |',
        "|-------|--------|--------------------|--------------------|-----------|---------------------|",
    ]
    for item in rows():
        lerobot = f"`{item.lerobot_type}`" if item.lerobot_type else _NONE
        native = f"`{item.native_driver}`" if item.native_driver else _NONE
        asset = f"`{item.asset_dir}`" if item.asset_dir else _NONE
        policies = ", ".join(f"`{p}`" for p in item.policies) or _NONE
        lines.append(
            f"| [`{item.name}`]({item.name}.md) | `{item.category}` | {lerobot} | {native} | {asset} | {policies} |"
        )
    return "\n".join(lines)


def render() -> str:
    """The whole generated block: the per-family summary, then every robot."""
    return f"{_summary()}\n\n{_matrix()}\n"


def substitute(markdown: str, page_path: str = "<string>") -> str:
    """Replace a ``{{coverage_matrix}}`` line with the generated tables."""
    if not _TOKEN.search(markdown):
        return markdown
    for problem in check_witnesses():
        log.warning("coverage: stale embodiment witness, %s", problem)
    return _TOKEN.sub(lambda _: render(), markdown)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001 - mkdocs signature
    """Render the matrix into one page, the hook entry point mkdocs calls."""
    return substitute(markdown, page.file.src_path)


if __name__ == "__main__":
    problems = check_witnesses()
    print(render())
    print("\n".join(f"STALE {p}" for p in problems) or "witnesses ok")


def _registry_view():  # noqa: ANN202 - a sibling hook module, loaded by path like the others
    """``docs/hooks/registry_view.py``: robots.json merged with the URDF long tail."""
    name = "docs_hooks_registry_view"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / "registry_view.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
