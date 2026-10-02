"""The published coverage matrix is the live driver-coverage join.

``docs/hooks/coverage.py`` renders the catalog's "Coverage" block (a per-family
summary, then one row per robot) at build time, so the page cannot go stale the
way a hand-maintained coverage list does: a robot that gains a native driver
still reads as a gap until someone re-runs the join by hand. What the hook *can* do is disagree with
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

import json
import re
from pathlib import Path

from strands_robots.drivers import list_driver_coverage, list_native_drivers, resolve_driver
from tests._docs_hooks import docs_hook

_REPO = Path(__file__).resolve().parents[1]
_PAGE = _REPO / "docs" / "robots" / "index.md"
_TOKEN = "{{coverage_matrix}}"

#: A hook that published nothing must not read as a clean sweep.
_MINIMUM_ROBOTS = 70

#: One rendered row: the robot (linked to its page), its family, then the
#: four coverage cells (lerobot type, native driver, sim asset dir, policies).
_ROW = re.compile(
    r"^\| \[`(?P<robot>[a-z0-9_]+)`\]\((?P=robot)\.md\) \| `(?P<category>\w+)` \| (?P<cells>.*) \|$", re.M
)
#: The summary's Total row: robots, sim, lerobot, native, no driver.
_TOTAL = re.compile(
    r"^\| \*\*Total\*\* \| \*\*(\d+)\*\* \| \*\*(\d+)\*\* \| \*\*(\d+)\*\* \| \*\*(\d+)\*\* \| \*\*(\d+)\*\* \|$", re.M
)
_CODE = re.compile(r"`([^`]+)`")


def _hook():
    """The hook module, loaded from the docs tree the build loads it from."""
    return docs_hook("coverage")


def _span(cell: str) -> str | None:
    """The code span a cell holds, or ``None`` for the empty cell."""
    match = _CODE.match(cell)
    return match.group(1) if match else None


def _published() -> dict[str, dict[str, str | None]]:
    """The rendered matrix, as robot name -> its lerobot, native, asset and policies cells."""
    out: dict[str, dict[str, str | None]] = {}
    for match in _ROW.finditer(_hook().render()):
        cells = [cell.strip() for cell in match.group("cells").split("|")]
        assert len(cells) == 4, f"{match.group('robot')} row has {len(cells)} coverage cells, expected 4"
        lerobot, native, asset = (_span(cell) for cell in cells[:3])
        out[match.group("robot")] = {"lerobot": lerobot, "native": native, "asset": asset, "policies": cells[3]}
    return out


def _registry() -> dict[str, dict]:
    """The built-in robot registry, read the way the hook reads it.

    ``robots.json`` plus ``urdf_robots.json``, the ``robot_descriptions`` URDF
    robots the MuJoCo backend compiles on first use: both ship in the package
    and both are rows of ``list_robots()``, so both are rows of the matrix.
    """
    return dict(docs_hook("registry_view").merged())


def _live_coverage() -> dict[str, tuple[str, ...]]:
    """The live join over the built-in registry only.

    ``list_driver_coverage`` also reports robots the caller registered in
    ``~/.strands_robots/user_robots.json``; the published matrix documents the
    package, so those are the caller's rows, not the docs'.
    """
    shipped = _registry()
    return {name: drivers for name, drivers in list_driver_coverage().items() if name in shipped}


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
    assert published == _live_coverage()


def test_every_native_cell_names_the_class_registered_for_that_robot() -> None:
    published = {robot: cells["native"] for robot, cells in _published().items() if cells["native"]}
    assert published == {name: cls for name, cls in list_native_drivers().items() if name in _registry()}


def test_every_default_driver_is_the_one_robot_builds_without_driver() -> None:
    """The robot pages say which driver ``Robot(name, mode="real")`` builds; it must be the factory's."""
    published = {row.name: row.default_driver for row in _hook().rows()}
    assert published == {name: resolve_driver(name) for name in published}


def test_every_lerobot_cell_is_the_type_the_registry_declares() -> None:
    registry = _registry()
    published = {robot: cells["lerobot"] for robot, cells in _published().items()}
    declared = {name: spec.get("hardware", {}).get("lerobot_type") for name, spec in registry.items()}
    assert published == declared


def test_the_summary_totals_count_the_same_join() -> None:
    coverage = _live_coverage()
    total = _TOTAL.search(_hook().render())
    assert total is not None, "the generated block carries no Total row"
    robots, sim, lerobot, native, neither = (int(group) for group in total.groups())
    assert (robots, sim, lerobot, native, neither) == (
        len(coverage),
        sum(1 for spec in _registry().values() if spec.get("asset")),
        sum("lerobot" in drivers for drivers in coverage.values()),
        sum("strands" in drivers for drivers in coverage.values()),
        sum(not drivers for drivers in coverage.values()),
    )


def test_every_sim_asset_cell_is_the_directory_the_registry_declares() -> None:
    registry = _registry()
    published = {robot: cells["asset"] for robot, cells in _published().items()}
    declared = {name: (spec.get("asset") or {}).get("dir") for name, spec in registry.items()}
    assert published == declared


def test_every_body_bound_policy_is_a_registered_provider_with_a_live_witness() -> None:
    """The policies column names providers from policies.json whose witness literal is still in the source."""
    hook = _hook()
    assert hook.check_witnesses() == [], "coverage.py carries a stale embodiment witness"
    providers = set(
        json.loads((_REPO / "strands_robots/registry/policies.json").read_text(encoding="utf-8"))["providers"]
    )
    for robot, cells in _published().items():
        named = _CODE.findall(cells["policies"] or "")
        assert set(named) <= providers, f"{robot} row names providers the registry lacks: {set(named) - providers}"
    bound = {robot for robot, cells in _published().items() if _CODE.findall(cells["policies"] or "")}
    assert bound == {robot for _p, robots, _m, _l in hook.EMBODIMENT_WITNESSES for robot in robots}


def test_the_catalog_page_carries_the_token_and_names_the_generator() -> None:
    page = _PAGE.read_text(encoding="utf-8")
    assert _TOKEN in page, f"{_PAGE.name} does not spell {_TOKEN} - the hook renders nothing"
    assert "docs/hooks/coverage.py" in page, (
        f"{_PAGE.name} carries a generated table without saying which hook writes it"
    )
    rendered = _hook().substitute(page, str(_PAGE))
    assert _TOKEN not in rendered and "| **Total** |" in rendered


def test_a_page_without_the_token_is_returned_unchanged() -> None:
    assert _hook().substitute("nothing to render here\n") == "nothing to render here\n"
