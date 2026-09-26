"""What a page or an example publishes as drivable is the driver-coverage join.

``list_driver_coverage()`` is the only complete answer to "can this robot be
driven for real": it joins the lerobot robot types the registry declares with
the native drivers this package registers, and the second half needs no registry
declaration at all. Two shipped surfaces published a narrower answer as if it
were that one, both measured on ``982013eeb``:

1. ``examples/registry/lerobot_hardware_catalog.py`` walked
   ``list_robots(mode="real")``, which reads the registry's ``hardware`` block.
   It printed **24** robots under the header "with LeRobot hardware support"
   while **35** can be built, leaving out every robot a shipped native driver
   reaches without a declaration -- ``panda``, ``fr3``, ``ur5e``, ``vx300s``,
   ``aloha``, ``wx250s``, ``trossen_wxai``, ``dynamixel_2r``, ``open_duck_mini``,
   ``ur10e``, ``fr3_v2`` -- and printing ``?`` as the type of the 8 rows it did
   list that have no lerobot type either. A reader could not tell the ``?`` rows
   from a registry defect, nor find the 11 missing arms at all.

2. ``docs/getting-started/robot-factory.md`` taught the join with a worked
   result, ``coverage["panda"]`` as the empty tuple that is the driver gap. The
   Franka driver shipped in 0.5.2 and ``coverage["panda"]`` reads
   ``('strands',)``, so the page's own example of a sim-only robot is one of the
   robots its native driver can build.

Both cells read the surface and compare it with the live join, so the numbers
stay derived: a driver registered tomorrow widens both sides at once.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import re
from pathlib import Path

import pytest

from strands_robots.drivers import list_driver_coverage

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLE = _REPO_ROOT / "examples" / "registry" / "lerobot_hardware_catalog.py"
_PAGE = _REPO_ROOT / "docs" / "getting-started" / "robot-factory.md"

#: The worked result the page states: the expression line, then its output as a
#: comment. Read as one claim because neither half means anything alone.
_DOCUMENTED_CLAIM = re.compile(r"^(?P<expression>coverage\[.*\])\n#\s*(?P<result>\(.*\))\s*$", re.M)
_CLAIMED_ROBOT = re.compile(r'coverage\["([a-z0-9_]+)"\]')

#: Floors, so a reflow or a rename reports a shrunken sweep rather than a pass.
_MINIMUM_CATALOGUED = 30
_MINIMUM_CLAIMED = 3


def _drivable() -> dict[str, tuple[str, ...]]:
    """Every robot a driver can build, and the drivers that can."""
    return {name: drivers for name, drivers in list_driver_coverage().items() if drivers}


def catalogue_rows(text: str) -> dict[str, set[str]]:
    """Parse a printed catalog into robot name -> the driver names its row spells.

    The table is the block between the header's rule and the blank line after
    it, whatever columns the header declares, so the parse grades the published
    rows rather than one layout of them.
    """
    rows: dict[str, set[str]] = {}
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if set(line.strip()) == {"-"}]
    assert starts, "the catalog printed no table"
    for line in lines[starts[0] + 1 :]:
        if not line.strip():
            break
        columns = line.split()
        rows[columns[0]] = {column for column in columns[1:] if column in {"lerobot", "strands"}}
    return rows


def _printed_catalogue() -> dict[str, set[str]]:
    """Run the example's catalog and read what it published."""
    spec = importlib.util.spec_from_file_location("lerobot_hardware_catalog", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert module.show_catalog() == 0
    return catalogue_rows(buffer.getvalue())


def documented_claims(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """The ``coverage[robot]`` results ``text`` states, as (robot, tuple) pairs."""
    match = _DOCUMENTED_CLAIM.search(text)
    assert match is not None, "the page states no worked coverage result"
    robots = _CLAIMED_ROBOT.findall(match.group("expression"))
    results = ast.literal_eval(match.group("result"))
    assert len(robots) == len(results), f"{len(robots)} robots against {len(results)} results"
    return list(zip(robots, results, strict=True))


class TestTheCatalogPublishesTheJoin:
    """The hardware catalog lists what a driver can build, and which one does."""

    def test_every_robot_a_driver_can_build_is_catalogued(self) -> None:
        """The 11 arms only a native driver reaches must not be left out."""
        published = _printed_catalogue()
        assert set(published) == set(_drivable()), (
            f"catalogued {len(published)} robots, {len(_drivable())} can be built: "
            f"missing {sorted(set(_drivable()) - set(published))}, "
            f"extra {sorted(set(published) - set(_drivable()))}"
        )

    def test_every_row_names_the_drivers_that_build_it(self) -> None:
        """A row that names no driver is the ``?`` cell a reader cannot act on."""
        drivable = _drivable()
        published = _printed_catalogue()
        wrong = {name: sorted(named) for name, named in published.items() if named != set(drivable.get(name, ()))}
        assert not wrong, f"rows name drivers the join does not: {wrong}"

    def test_the_catalog_reaches_the_registry(self) -> None:
        """An empty or truncated table is reported instead of read as clean."""
        assert len(_printed_catalogue()) >= _MINIMUM_CATALOGUED


class TestTheDocumentedCoverageResultsHold:
    """A worked ``list_driver_coverage()`` result on the page is the live one."""

    def test_each_documented_result_matches_the_join(self) -> None:
        """The ``coverage["panda"] == ()`` shape must fail once a driver ships."""
        live = list_driver_coverage()
        stale = {
            robot: {"documented": documented, "live": live[robot]}
            for robot, documented in documented_claims(_PAGE.read_text(encoding="utf-8"))
            if live[robot] != documented
        }
        assert not stale, f"the page states coverage results the join does not: {stale}"

    def test_the_page_works_the_result_for_several_robots(self) -> None:
        """One robot cannot teach both a populated tuple and the driver gap."""
        claims = documented_claims(_PAGE.read_text(encoding="utf-8"))
        assert len(claims) >= _MINIMUM_CLAIMED
        assert any(not result for _, result in claims), "no claim shows the empty-tuple driver gap"


class TestTheGradersAreLoadBearing:
    """Each parse reports a planted offender, and only the offender."""

    def test_a_row_naming_no_driver_is_reported(self) -> None:
        """The pre-fix layout - a lerobot type where the driver belongs - is read as none."""
        rows = catalogue_rows(
            "name             canonical        lerobot_type             category\n"
            "-------------------------------------------------------------------\n"
            "booster_t1       booster_t1       ?                        humanoid\n"
            "so100            so100            so100_follower           arm\n"
            "\n"
            "Drive any of them for real with, e.g.:\n"
        )
        assert rows == {"booster_t1": set(), "so100": set()}

    def test_a_row_naming_both_drivers_is_read_as_both(self) -> None:
        """The fixed layout's join column is what the rule grades."""
        rows = catalogue_rows(
            "name             driver=            lerobot_type             native driver\n"
            "---------------------------------------------------------------------------\n"
            "earthrover       lerobot strands    earthrover_mini_plus     EarthRoverDriver\n"
            "panda            strands            -                        FrankaDriver\n"
            "\n"
        )
        assert rows == {"earthrover": {"lerobot", "strands"}, "panda": {"strands"}}

    def test_a_stale_documented_result_is_reported(self) -> None:
        """A tuple the join contradicts is caught; the ones it agrees with are not."""
        claims = documented_claims("coverage[\"so101\"], coverage[\"panda\"]\n# (('lerobot', 'strands'), ())\n")
        live = list_driver_coverage()
        assert [robot for robot, result in claims if live[robot] != result] == ["panda"]

    def test_a_result_the_page_does_not_work_is_not_invented(self) -> None:
        """A page with no worked result is reported as that, not silently passed."""
        with pytest.raises(AssertionError, match="no worked coverage result"):
            documented_claims("`list_driver_coverage()` reports the join.\n")
