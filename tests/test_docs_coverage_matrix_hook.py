"""The published coverage matrix is the live driver-coverage join.

``docs/hooks/coverage_matrix.py`` renders the catalog's "Drivable for real"
table at build time, so the page cannot go stale the way a hand-maintained
coverage list does: a robot that gains a native driver still reads as a gap
until someone re-runs the join by hand. What the hook *can* do is disagree with
the package, because it reads both registries out of the source with
:mod:`ast` and :mod:`json` - the docs environment installs mkdocs alone and
cannot import ``strands_robots`` - while a caller is answered by
:func:`~strands_robots.drivers.list_driver_coverage`.

Every cell here drives the hook and compares the rendered table with the live
registries rather than with a second reading of the same source, so a driver
registered tomorrow widens both sides at once and a table that drifts from
either registry fails at the row that drifted.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

from strands_robots.drivers import list_driver_coverage, list_native_drivers

_REPO = Path(__file__).resolve().parents[1]
_PAGE = _REPO / "docs" / "robots" / "index.md"
_TOKEN = "{{coverage_matrix}}"

#: A hook that published nothing must not read as a clean sweep.
_MINIMUM_ROBOTS = 70

#: One rendered row: the robot, its category, then the three coverage cells.
_ROW = re.compile(r"^\| `(?P<robot>[a-z0-9_]+)` \| (?P<category>\w+) \| (?P<cells>.*) \|$", re.M)
_TOTAL = re.compile(r"^\| \*\*Total\*\* \| \*\*(\d+)\*\* \| \*\*(\d+)\*\* \| \*\*(\d+)\*\* \| \*\*(\d+)\*\* \|$", re.M)
_CODE = re.compile(r"`([^`]+)`")


def _hook():
    """The hook module, loaded from the docs tree the build loads it from."""
    name = "docs_coverage_matrix_hook"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _REPO / "docs/hooks/coverage_matrix.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # the hook's dataclass resolves its module by name
    spec.loader.exec_module(module)
    return module


def _span(cell: str) -> str | None:
    """The code span a cell holds, or ``None`` for the empty cell."""
    match = _CODE.match(cell)
    return match.group(1) if match else None


def _published() -> dict[str, dict[str, str | None]]:
    """The rendered matrix, as robot name -> its lerobot, native and asset cells."""
    out: dict[str, dict[str, str | None]] = {}
    for match in _ROW.finditer(_hook().render()):
        cells = [cell.strip() for cell in match.group("cells").split("|")]
        assert len(cells) == 3, f"{match.group('robot')} row has {len(cells)} coverage cells, expected 3"
        lerobot, native, asset = (_span(cell) for cell in cells)
        out[match.group("robot")] = {"lerobot": lerobot, "native": native, "asset": asset}
    return out


def _registry() -> dict[str, dict]:
    """The built-in robot registry, read the way the hook reads it."""
    path = _REPO / "strands_robots" / "registry" / "robots.json"
    return json.loads(path.read_text(encoding="utf-8"))["robots"]


def test_every_registered_robot_has_exactly_one_row() -> None:
    published, registry = _published(), _registry()
    assert len(registry) >= _MINIMUM_ROBOTS, f"the registry holds {len(registry)} robots - the floor is stale"
    assert set(published) == set(registry), (
        f"rows the registry does not hold: {sorted(set(published) - set(registry))}; "
        f"registered robots with no row: {sorted(set(registry) - set(published))}"
    )


def test_the_published_join_is_the_live_join() -> None:
    """Each row's two driver cells spell the drivers that can build that robot."""
    published = {
        robot: tuple(
            name for name, cell in (("lerobot", cells["lerobot"]), ("strands", cells["native"])) if cell is not None
        )
        for robot, cells in _published().items()
    }
    assert published == list_driver_coverage()


def test_every_native_cell_names_the_class_registered_for_that_robot() -> None:
    published = {robot: cells["native"] for robot, cells in _published().items() if cells["native"]}
    assert published == list_native_drivers()


def test_every_lerobot_cell_is_the_type_the_registry_declares() -> None:
    registry = _registry()
    published = {robot: cells["lerobot"] for robot, cells in _published().items()}
    declared = {name: spec.get("hardware", {}).get("lerobot_type") for name, spec in registry.items()}
    assert published == declared


def test_the_summary_totals_count_the_same_join() -> None:
    coverage = list_driver_coverage()
    total = _TOTAL.search(_hook().render())
    assert total is not None, "the generated block carries no Total row"
    robots, lerobot, native, neither = (int(group) for group in total.groups())
    assert (robots, lerobot, native, neither) == (
        len(coverage),
        sum("lerobot" in drivers for drivers in coverage.values()),
        sum("strands" in drivers for drivers in coverage.values()),
        sum(not drivers for drivers in coverage.values()),
    )


def test_the_catalog_page_carries_the_token_and_names_the_generator() -> None:
    page = _PAGE.read_text(encoding="utf-8")
    assert _TOKEN in page, f"{_PAGE.name} does not spell {_TOKEN} - the hook renders nothing"
    assert "docs/hooks/coverage_matrix.py" in page, (
        f"{_PAGE.name} carries a generated table without saying which hook writes it"
    )
    rendered = _hook().substitute(page, str(_PAGE))
    assert _TOKEN not in rendered and "| **Total** |" in rendered


def test_a_page_without_the_token_is_returned_unchanged() -> None:
    assert _hook().substitute("nothing to render here\n") == "nothing to render here\n"
