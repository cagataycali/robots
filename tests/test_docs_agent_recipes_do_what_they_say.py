"""A recipe a page hands an agent-writer imports under that page's install, and builds what it says.

``docs/agents.md`` teaches the two things a reader cannot get from the tool spec:
which tools to hand the ``Agent``, and how a plain-English dimension becomes an
``add_object`` call. Both of its recipes were satisfiable only on some other
page's terms.

* The "Add more tools" snippet imports ``pose_tool``, whose module body requires
  ``pyserial``. No extra of this project declares that (it arrives inside
  ``lerobot[feetech]``), and the page's own install fence is
  ``strands-agents "strands-robots[sim-mujoco]"`` -- so following the page top to
  bottom ends in ``ImportError: cannot import name 'pose_tool'``.
  ``docs/hardware/tools.md`` states the dependency; this page did not.
* The "Common patterns" table mapped "Add a 5cm red cube" to ``size=[0.025]*3``.
  ``size`` is the FULL extent, so that builds a 2.5 cm cube -- half what the
  instruction asks for, and the half-extents MuJoCo stores for a 5 cm one.
  ``add_object``'s own docstring and ``docs/simulation/objects.md`` spell 5 cm as
  ``[0.05, 0.05, 0.05]``, and ``add_object("default_box")`` compiles exactly
  those extents.

The import rule is graded over every shipped page and example, with the guarded
tools and their install tokens harvested from the ``require_optional`` calls in
``strands_robots/tools/``, so an import-time dependency that lands later is
covered without editing a list. The size rule builds each documented row and
measures the compiled geom, so it grades the recipe and not the string.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_PAGE = REPO_ROOT / "docs" / "agents.md"
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
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            call = _require_optional_call(node)
            if call is None:
                continue
            for keyword in call.keywords:
                if keyword.arg == "pip_install" and isinstance(keyword.value, ast.Constant):
                    guarded[path.stem] = str(keyword.value.value)
    return guarded


GUARDED = _import_time_guarded_tools()


def _shipped_prose() -> list[Path]:
    """Every page and example a reader copies from."""
    return sorted(
        {
            *(REPO_ROOT / "docs").rglob("*.md"),
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


def _literal(node: ast.expr) -> object:
    """Evaluate a documented argument: a literal, or the ``[x] * n`` a table uses."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left, right = _literal(node.left), _literal(node.right)
        if isinstance(left, list) and isinstance(right, int):
            return left * right
        raise ValueError(f"unsupported product in a documented recipe: {ast.unparse(node)}")
    return ast.literal_eval(node)


def _sized_add_object_rows(page: str) -> list[tuple[str, float, str, list[float]]]:
    """Table rows that state a size in cm and answer with an ``add_object`` call.

    Returns the instruction, the size it states, and the shape and extents its
    recipe passes - explicit columns, because a ``**kwargs`` dict harvested from
    prose has no type the signature accepts.
    """
    rows = []
    for instruction, chain in re.findall(r"^\|\s*\"([^\"]+)\"\s*\|(.+?)\|\s*$", page, re.MULTILINE):
        centimetres = re.search(r"(\d+(?:\.\d+)?)\s*cm", instruction, re.IGNORECASE)
        call = re.search(r"add_object\((.*?)\)`", chain)
        if not centimetres or not call:
            continue
        parsed = ast.parse(f"add_object({call.group(1)})", mode="eval").body
        assert isinstance(parsed, ast.Call)
        kwargs = {kw.arg: _literal(kw.value) for kw in parsed.keywords if kw.arg}
        size, shape = kwargs.get("size"), kwargs.get("shape", "box")
        if isinstance(size, list) and isinstance(shape, str):
            rows.append((instruction, float(centimetres.group(1)), shape, [float(x) for x in size]))
    return rows


SIZED_ROWS = _sized_add_object_rows(AGENTS_PAGE.read_text(encoding="utf-8"))


def test_the_page_still_documents_a_sized_recipe():
    """A table that lost its sized row would silently stop being graded."""
    assert SIZED_ROWS, f"{AGENTS_PAGE.relative_to(REPO_ROOT)} documents no add_object size recipe"


@pytest.mark.parametrize(
    ("instruction", "centimetres", "shape", "size"),
    SIZED_ROWS,
    ids=[instruction for instruction, _, _, _ in SIZED_ROWS],
)
def test_a_documented_size_recipe_builds_the_size_it_states(
    instruction: str, centimetres: float, shape: str, size: list[float]
):
    """``size`` is the full extent, so the compiled geom must measure what the row says."""
    mj = pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    sim = Simulation(tool_name="test_docs_agent_recipe_sim", mesh=False)
    try:
        sim.create_world()
        assert sim.add_object("documented", shape=shape, size=size)["status"] == "success"
        model = sim._world._model if sim._world else None
        assert model is not None, "the documented recipe compiled no world"
        geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "documented_geom")
        assert geom >= 0, "the documented recipe compiled no geom"
        edges = [2 * half for half in model.geom_size[geom]]
    finally:
        sim.cleanup()
    assert max(edges) == pytest.approx(centimetres / 100.0), (
        f"{instruction!r} maps to size={size}, which builds "
        f"{max(edges) * 100:.2f} cm; size is the full extent, not the half-extents MuJoCo stores"
    )
