"""A recipe a page hands an agent-writer imports under that page's install, and builds what it says.

``docs/learn/agents.md`` teaches the two things a reader cannot get from the tool spec:
which tools to hand the ``Agent``, and what happens when the model asks a real robot to
move. ``docs/learn/simulation/worlds-and-objects.md`` teaches how a plain-English
dimension becomes an ``add_object`` call. The old versions of both pages were
satisfiable only on some other page's terms:

* The agents page's snippet imports ``pose_tool``, whose module body requires
  ``pyserial``. No extra of this project declares that (it arrives inside
  ``lerobot[feetech]``), and the Start pages install ``strands-robots[sim-mujoco]``,
  so following the site top to bottom ends in ``ImportError``. The page that
  imports a guarded tool has to name the install line itself.
* A "Common patterns" table once mapped "Add a 5cm red cube" to ``size=[0.025]*3``.
  ``size`` is the FULL extent, so that built a 2.5 cm cube, the half-extents MuJoCo
  stores for a 5 cm one. The new page states the convention as a table, one row per
  shape, and that table is graded by building each shape and measuring the geom.

The import rule is graded over every shipped page and example, with the guarded
tools and their install tokens harvested from the ``require_optional`` calls in
``strands_robots/tools/``, so an import-time dependency that lands later is
covered without editing a list. The size rule builds each documented shape and
measures the compiled geom, so it grades the recipe and not the string.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest

from tests._package_ast import parse_file

REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECTS_PAGE = REPO_ROOT / "docs" / "learn" / "simulation" / "worlds-and-objects.md"
TOOLS_DIR = REPO_ROOT / "strands_robots" / "tools"


def _require_optional_call(node: ast.stmt) -> ast.Call | None:
    """The module-level ``require_optional(...)`` call ``node`` is, if it is one.

    Covers the three spellings the tools use: a bare expression, a plain
    assignment, and the annotated ``serial: Any = require_optional(...)``.
    """
    value = (
        node.value
        if isinstance(node, ast.Expr | ast.Assign | ast.AnnAssign) and isinstance(node.value, ast.Call)
        else None
    )
    if value is None or getattr(value.func, "id", "") != "require_optional":
        return None
    return value


def _import_time_guarded_tools() -> dict[str, str]:
    """Map each tool that needs a dependency to import to its ``pip install`` token."""
    guarded: dict[str, str] = {}
    for path in sorted(TOOLS_DIR.glob("*.py")):
        for node in parse_file(path).body:
            call = _require_optional_call(node)
            if call is None:
                continue
            for keyword in call.keywords:
                if keyword.arg == "pip_install" and isinstance(keyword.value, ast.Constant):
                    guarded[path.stem] = str(keyword.value.value)
    return guarded


GUARDED = _import_time_guarded_tools()


def _shipped_prose() -> list[Path]:
    """Every page and example a reader copies from; generated robot pages carry no imports."""
    docs = REPO_ROOT / "docs"
    return sorted(
        {
            *(p for p in docs.rglob("*.md") if "robots" not in p.relative_to(docs).parts),
            *(REPO_ROOT / "examples").rglob("*.py"),
            *(REPO_ROOT / "examples").rglob("*.md"),
            REPO_ROOT / "README.md",
        }
    )


def _guarded_imports(text: str) -> set[str]:
    """The guarded tools ``text`` tells a reader to import from the package."""
    imported: set[str] = set()
    for line in re.findall(r"from strands_robots(?:\.tools[.\w]*)? import [^\n]*", text):
        imported |= {name for name in GUARDED if re.search(rf"\b{name}\b", line)}
    return imported


def _pages_importing_a_guarded_tool() -> list[tuple[Path, frozenset[str]]]:
    found = []
    for path in _shipped_prose():
        names = _guarded_imports(path.read_text(encoding="utf-8"))
        if names:
            found.append((path, frozenset(names)))
    return found


IMPORTERS = _pages_importing_a_guarded_tool()


def test_the_guarded_roster_and_its_readers_are_populated():
    """Neither harvest may go quietly empty, which would pass every cell below."""
    assert GUARDED, f"no tool under {TOOLS_DIR} requires a dependency to import"
    assert IMPORTERS, "no shipped page imports a guarded tool; the rule below grades nothing"


@pytest.mark.parametrize(
    ("path", "names"),
    IMPORTERS,
    ids=[str(path.relative_to(REPO_ROOT)) for path, _ in IMPORTERS],
)
def test_a_page_importing_a_guarded_tool_names_the_install_it_needs(path: Path, names: frozenset[str]):
    """A snippet that cannot import on the page's own terms must say what it needs."""
    text = path.read_text(encoding="utf-8")
    missing = sorted(name for name in names if GUARDED[name] not in text)
    assert not missing, (
        f"{path.relative_to(REPO_ROOT)} imports {missing} without naming "
        f"{sorted({GUARDED[name] for name in missing})}, which no extra of this project declares"
    )


# --- the size convention ---------------------------------------------------------

#: How the page's ``size`` row for each shape maps onto the geom MuJoCo stores, given
#: one probe size. ``expected`` is what ``model.geom_size`` must read for that probe.
_PROBES: dict[str, tuple[list[float], list[float]]] = {
    "box": ([0.05, 0.06, 0.07], [0.025, 0.03, 0.035]),  # full edge lengths, halved
    "ellipsoid": ([0.05, 0.06, 0.07], [0.025, 0.03, 0.035]),
    "sphere": ([0.04], [0.02]),  # diameter to radius
    "cylinder": ([0.04, 0.0, 0.2], [0.02, 0.1]),  # diameter, unused, full height
    "capsule": ([0.04, 0.0, 0.2], [0.02, 0.1]),  # diameter, unused, cylinder length
}

_CONVENTION_WORDS: dict[str, str] = {
    "box": "full edge lengths",
    "ellipsoid": "full edge lengths",
    "sphere": "diameter",
    "cylinder": "full height",
    "capsule": "cylinder length",
}


def _size_rows() -> dict[str, str]:
    """``shape -> size cell`` from the page's size-convention table."""
    rows: dict[str, str] = {}
    for line in OBJECTS_PAGE.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\|\s*((?:`\w+`(?:,\s*)?)+)\s*\|(.+)\|\s*$", line)
        if not match:
            continue
        for shape in re.findall(r"`(\w+)`", match.group(1)):
            rows[shape] = match.group(2).strip()
    return rows


SIZE_ROWS = _size_rows()


def test_the_page_still_documents_the_size_convention():
    """A table that lost its rows would silently stop being graded."""
    assert set(_PROBES) <= set(SIZE_ROWS), (
        f"{OBJECTS_PAGE.relative_to(REPO_ROOT)} lacks size rows for {sorted(set(_PROBES) - set(SIZE_ROWS))}"
    )
    text = OBJECTS_PAGE.read_text(encoding="utf-8")
    assert "full extent" in text, "the page must say size is the full extent, the fact the old table got wrong"


def test_the_documented_signature_names_only_real_parameters():
    from strands_robots.simulation.mujoco.simulation import Simulation

    text = OBJECTS_PAGE.read_text(encoding="utf-8")
    shown = re.search(r"`add_object\(([^)]*)\)`", text)
    assert shown is not None, "the page no longer states the add_object signature"
    listed = [p.split("=")[0].strip() for p in shown.group(1).split(",") if p.strip()]
    real = [p for p in inspect.signature(Simulation.add_object).parameters if p != "self"]
    assert listed == real, f"the page shows add_object({', '.join(listed)}); the method takes ({', '.join(real)})"


@pytest.mark.parametrize("shape", sorted(_PROBES))
def test_each_documented_size_row_builds_the_extent_it_states(shape: str):
    """``size`` is the full extent, so the compiled geom must measure what the row says."""
    mj = pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    assert _CONVENTION_WORDS[shape] in SIZE_ROWS.get(shape, ""), (
        f"the {shape} row reads {SIZE_ROWS.get(shape)!r}; expected it to say {_CONVENTION_WORDS[shape]!r}"
    )
    size, expected = _PROBES[shape]
    sim = Simulation(tool_name="test_docs_agent_recipe_sim", mesh=False)
    try:
        sim.create_world()
        assert sim.add_object("documented", shape=shape, size=size)["status"] == "success"
        model = sim._world._model if sim._world else None
        assert model is not None, "the documented recipe compiled no world"
        geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "documented_geom")
        assert geom >= 0, "the documented recipe compiled no geom"
        stored = [float(x) for x in model.geom_size[geom][: len(expected)]]
    finally:
        sim.cleanup()
    assert stored == pytest.approx(expected), (
        f"{shape}: size={size} compiles geom_size={stored}; the row says {SIZE_ROWS[shape]!r}, "
        "which predicts {expected}"
    )


AGENTS_PAGE = REPO_ROOT / "docs" / "learn" / "agents.md"


def _tool_table_rows() -> list[tuple[str, str]]:
    """``(tool, module)`` per name in the agents page tool table; a row without ``from`` means the parent."""
    section = AGENTS_PAGE.read_text(encoding="utf-8").split("## The tools around the robot", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        cell = line.split("|")[1]
        names, _, module = cell.partition(" from ")
        target = module.strip(" `") or "strands_robots.tools"
        rows += [(name, target) for name in re.findall(r"`([a-z0-9_]+\*?)`", names)]
    return rows


@pytest.mark.parametrize(("tool", "module"), _tool_table_rows())
def test_each_tool_in_the_agents_table_imports_from_the_module_its_row_names(tool: str, module: str):
    """``from <module> import <tool>`` is what a reader types after reading the row; ``g1_*`` needs one match."""
    lazy = importlib.import_module(module)._LAZY_IMPORTS
    found = any(name.startswith(tool[:-1]) for name in lazy) if tool.endswith("*") else tool in lazy
    assert found, f"agents.md says `{tool}` imports from {module}; it does not"
