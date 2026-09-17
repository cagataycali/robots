"""The docs entry surface: a browsable nav, and a landing example that builds.

Two things a reader meets before any page content, neither of them graded.

**The nav.** ``mkdocs.yml`` listed 24 top-level entries - a sidebar longer than
most of the pages it indexes, with three ROS variants and single pages such as
``Dashboard`` and ``Configuration`` sitting at the same level as ``Robots``.
MkDocs has nothing to say about that: a nav of any width builds clean under
``--strict``, and a page left out of the nav is an ``INFO`` line, so *shrinking*
the sidebar by dropping pages would build clean too. Both halves are pinned
here - at most eight top-level entries, nothing nested more than one level below
them, and every page under ``docs/`` reachable from the nav - because either one
alone can be satisfied by breaking the other.

**The landing example.** ``docs/index.md`` carries the first call a reader
copies. ``tests/test_docs_python_examples_are_callable.py`` grades keyword sets
across every page against the real signatures; what no guard could say is
whether the landing page's own call *builds a robot*. This module runs the call
the page spells - read out of the fence with :mod:`ast`, not restated here - and
hands the result to an ``Agent`` the way the next line does.

The nav is read from ``mkdocs.yml`` as text rather than as YAML: the file
carries ``!!python/name:`` tags that ``yaml.safe_load`` refuses, and indentation
is what a reader sees anyway.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest
from strands import Agent

from strands_robots import Robot
from strands_robots.drivers import list_driver_coverage
from strands_robots.simulation.base import SimEngine

REPO_ROOT = Path(__file__).resolve().parent.parent
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"
DOCS_DIR = REPO_ROOT / "docs"
LANDING_PAGE = DOCS_DIR / "index.md"

#: A sidebar a reader can take in at a glance. Sections group the pages; the
#: pages themselves are one level down, and nothing goes deeper.
MAX_TOP_LEVEL_ENTRIES = 8
MAX_DEPTH = 1

#: The landing page's budget. It is a page that shows the product, not a
#: chapter: install, one runnable example, where to go next.
MAX_LANDING_PAGE_LINES = 120

_NAV_ITEM = re.compile(r"^(?P<indent> *)- (?P<body>.+?)\s*$")
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _nav_items() -> list[tuple[int, str]]:
    """Every nav entry as ``(depth, target)``; ``depth`` 0 is top level.

    ``target`` is the page path an entry points at, or ``""`` for a section
    heading (``- Robots:``) that only groups the entries under it.
    """
    lines = MKDOCS_YML.read_text(encoding="utf-8").splitlines()
    start = lines.index("nav:")
    items: list[tuple[int, str]] = []
    for line in lines[start + 1 :]:
        if not line.strip():
            continue
        if not line.startswith((" ", "-")):  # the next top-level key ends the nav
            break
        match = _NAV_ITEM.match(line)
        assert match, f"mkdocs.yml nav line is not a list item: {line!r}"
        depth = len(match["indent"]) // 2
        body = match["body"]
        # "Title: path", a bare "path", or a section heading "Title:".
        target = body.split(":", 1)[1].strip() if ":" in body else body
        items.append((depth, target))
    assert items, "no nav entries parsed out of mkdocs.yml"
    return items


class TestTheNavStaysBrowsable:
    """A sidebar of grouped sections, covering every page, and no deeper."""

    def test_the_nav_lists_at_most_eight_top_level_entries(self) -> None:
        tops = [target for depth, target in _nav_items() if depth == 0]
        assert len(tops) <= MAX_TOP_LEVEL_ENTRIES, (
            f"mkdocs.yml nav has {len(tops)} top-level entries; at most "
            f"{MAX_TOP_LEVEL_ENTRIES} fit a sidebar a reader can scan. Group the "
            f"new page under an existing section instead of adding one."
        )

    def test_nothing_is_nested_more_than_one_level_below_a_section(self) -> None:
        deepest = max(depth for depth, _ in _nav_items())
        assert deepest <= MAX_DEPTH, (
            f"mkdocs.yml nav nests {deepest} levels deep; a third level hides "
            f"pages behind two clicks. Keep it to sections and their pages."
        )

    def test_every_nav_entry_points_at_a_page_on_disk(self) -> None:
        missing = sorted(target for _, target in _nav_items() if target and not (DOCS_DIR / target).is_file())
        assert not missing, f"mkdocs.yml nav points at pages that do not exist: {missing}"

    def test_every_page_under_docs_is_reachable_from_the_nav(self) -> None:
        """A narrow nav must not be bought by orphaning pages.

        MkDocs reports a page missing from the nav as ``INFO``, which
        ``--strict`` does not fail on, so an unreferenced page ships and is
        reachable only by search or a direct link from another page.
        """
        navigated = {target for _, target in _nav_items() if target}
        on_disk = {str(p.relative_to(DOCS_DIR)) for p in DOCS_DIR.rglob("*.md")}
        orphans = sorted(on_disk - navigated)
        assert not orphans, (
            f"pages under docs/ that no nav entry reaches: {orphans}. Add each to "
            f"the nav or delete it; an unnavigated page is found only by search."
        )


class TestTheLandingPageShowsTheProduct:
    """The first screen: a budget, a call that builds, and honest output."""

    def test_the_landing_page_fits_its_line_budget(self) -> None:
        lines = LANDING_PAGE.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= MAX_LANDING_PAGE_LINES, (
            f"docs/index.md is {len(lines)} lines; the landing page's budget is "
            f"{MAX_LANDING_PAGE_LINES}. Move the detail onto the page that owns it "
            f"and link to it."
        )

    def test_the_example_builds_a_robot_an_agent_can_drive(self) -> None:
        """The page's own ``Robot(...)`` call is executed, not restated.

        The arguments are read out of the fence, so a page edited to name a
        robot the registry does not hold, or a mode the factory refuses, fails
        here rather than on the reader's first line.
        """
        args, kwargs = _the_landing_robot_call()
        assert (args, kwargs) == (("so101",), {"mode": "sim"}), (
            f"docs/index.md's first example calls Robot(*{args}, **{kwargs}); the "
            f"landing example is Robot('so101', mode='sim') - simulation, no GPU."
        )
        arm = Robot(*args, **kwargs)
        try:
            assert isinstance(arm, SimEngine), f"mode='sim' built a {type(arm).__name__}"
            # The next line of the page hands it to an agent as a tool.
            agent = Agent(tools=[arm])
            named_after_the_robot = [name for name in agent.tool_registry.registry if args[0] in name]
            assert named_after_the_robot, (
                f"the landing example's robot did not register as a tool named after "
                f"{args[0]!r}: {sorted(agent.tool_registry.registry)}"
            )
        finally:
            arm.destroy()

    def test_every_method_the_page_names_on_that_object_exists(self) -> None:
        """Prose naming ``arm.<method>()`` names methods the object really has."""
        named = sorted(set(re.findall(r"`arm\.([a-z_]+)\(", LANDING_PAGE.read_text(encoding="utf-8"))))
        assert named, "the page no longer names a method on the example's robot"
        args, kwargs = _the_landing_robot_call()
        arm = Robot(*args, **kwargs)
        try:
            missing = [name for name in named if not hasattr(arm, name)]
            assert not missing, f"docs/index.md names methods the example's robot does not have: {missing}"
        finally:
            arm.destroy()

    def test_the_driver_coverage_output_is_what_the_call_returns(self) -> None:
        """The commented result beside ``list_driver_coverage()`` is derived."""
        page = LANDING_PAGE.read_text(encoding="utf-8")
        shown = re.search(r"list_driver_coverage\(\)\[\"(?P<robot>[a-z0-9_]+)\"\]\s*#\s*(?P<out>.+)", page)
        assert shown, "docs/index.md no longer shows a list_driver_coverage() lookup with its result"
        robot = shown["robot"]
        assert repr(list_driver_coverage()[robot]) == shown["out"].strip(), (
            f"docs/index.md says list_driver_coverage()[{robot!r}] is {shown['out'].strip()}; "
            f"it returns {list_driver_coverage()[robot]!r}."
        )


def _the_landing_robot_call() -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The ``(args, kwargs)`` of the first ``Robot(...)`` call on the landing page."""
    fences = _PYTHON_FENCE.findall(LANDING_PAGE.read_text(encoding="utf-8"))
    assert fences, "docs/index.md carries no python example"
    for source in fences:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Robot":
                return (
                    tuple(ast.literal_eval(a) for a in node.args),
                    {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg},
                )
    pytest.fail("docs/index.md carries no Robot(...) call for a reader to copy")
