"""The docs entry surface: a browsable nav, and a landing example that builds.

Two things a reader meets before any page content, neither of them graded.

**The nav.** ``mkdocs.yml`` once listed 24 top-level entries, a sidebar longer
than most of the pages it indexed. The rewrite puts six tabs in the strip
(Home, Start, Robots, Learn, Reference, Project); inside a tab the sidebar shows
sections and their pages, and nothing deeper. MkDocs has nothing to say about
that: a nav of any width builds clean under ``--strict``, and a page left out of
the nav is an ``INFO`` line, so *shrinking* the sidebar by dropping pages would
build clean too. Both halves are pinned: the width here (six tabs, a page at
most one section below its tab) and the coverage in
``tests/test_docs_two_lane_architecture.py`` (every page in the nav), because
either one alone can be satisfied by breaking the other.

**The landing example.** ``docs/index.md`` carries the first call a reader
copies. ``tests/test_docs_python_examples_are_callable.py`` grades keyword sets
across every page against the real signatures; what no guard could say is
whether the landing page's own call *builds a robot*. This module runs the call
the page spells, read out of the fence with :mod:`ast` and not restated here,
and hands the result to an ``Agent`` the way the next line does. Every method
the fence calls on that object is checked against the object it built.

**The first screen.** A reader arriving from a link decides in about thirty
seconds, on one image and three links. The page once opened on a control-loop
drawing, and its next-step cards pointed into pages of over a thousand words.
Now it opens on a live ``<robot-viewer>`` of the SO-101 that autoloads in the
hero, three proof numbers that the numbers hook derives from the tree, and six
cards. All are graded: the hero carries the viewer and no drawing precedes it,
the proof numbers are ``{{n:key}}`` tokens whose values agree with the package,
and every card lands on a page a newcomer finishes in one sitting.

The nav is read from ``mkdocs.yml`` as text rather than as YAML: the file
carries ``!!python/name:`` tags that ``yaml.safe_load`` refuses, and indentation
is what a reader sees anyway.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pytest
from strands import Agent

from strands_robots import Robot
from strands_robots.drivers import _SHIPPED_DRIVERS, HardwareDriver
from strands_robots.hardware_robot import Robot as HardwareRobot
from strands_robots.registry import list_robots
from strands_robots.registry.user_registry import _load_user_registry
from strands_robots.simulation.base import SimEngine

REPO_ROOT = Path(__file__).resolve().parent.parent
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"
DOCS_DIR = REPO_ROOT / "docs"
LANDING_PAGE = DOCS_DIR / "index.md"
FACTS_HOOK = DOCS_DIR / "hooks" / "facts.py"
VIEWER_MANIFEST = DOCS_DIR / "assets" / "viewer" / "robots.json"

#: The tab strip: Home plus the five sections of the information architecture.
#: Material lays the strip out in one row, so a seventh tab is clipped. Inside
#: a tab the sidebar shows sections (``Arms``, ``Policies``) and their pages,
#: two levels below the tab, and nothing goes deeper.
MAX_TOP_LEVEL_ENTRIES = 6
MAX_DEPTH = 2

#: The landing page's budget. It is a page that shows the product, not a
#: chapter: install, one runnable example, where to go next.
MAX_LANDING_PAGE_LINES = 120

#: What a next-step card may cost the reader who follows it. A page of this
#: length is read in one sitting; the exhaustive treatment is one link further
#: on, reached from there rather than from the landing page.
MAX_NEXT_STEP_WORDS = 800

#: Extensions that hold a drawing rather than a robot.
DRAWING_SUFFIXES = (".svg",)

_NAV_ITEM = re.compile(r"^(?P<indent> *)- (?P<body>.+?)\s*$")
_PYTHON_FENCE = re.compile(r"```python[^\n]*\n(.*?)```", re.DOTALL)
_IMAGE = re.compile(r"!\[[^\]]*\]\((?P<src>[^)\s]+)|<img\b[^>]*\bsrc=\"(?P<tag_src>[^\"]+)\"")
_VIEWER = re.compile(r"<robot-viewer\b(?P<attrs>[^>]*)>")
_HERO_DIV = re.compile(r'<div class="sr-hero"[^>]*>(?P<body>.*?)\n</div>\n\n', re.DOTALL)
_PROOF_DIV = re.compile(r'<div class="sr-proof"[^>]*>(?P<body>.*?)</div>\n\n', re.DOTALL)
_NUMBER_TOKEN = re.compile(r"\{\{\s*n:([a-z_]+)\s*\}\}")
_CARDS_DIV = re.compile(r'<div class="sr-grid" markdown>(?P<body>.*?)\n</div>\n\n', re.DOTALL)
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\((?P<target>[^)\s]+)\)")


def _nav_items() -> list[tuple[int, str]]:
    """Every nav entry as ``(depth, target)``; ``depth`` 0 is top level.

    ``target`` is the page path an entry points at, or ``""`` for a section
    heading (``- Robots:``) that only groups the entries under it.
    """
    lines = MKDOCS_YML.read_text(encoding="utf-8").splitlines()
    start = lines.index("nav:")
    items: list[tuple[int, str]] = []
    base: int | None = None
    for line in lines[start + 1 :]:
        if not line.strip():
            continue
        if not line.startswith((" ", "-")):  # the next top-level key ends the nav
            break
        match = _NAV_ITEM.match(line)
        assert match, f"mkdocs.yml nav line is not a list item: {line!r}"
        if base is None:
            base = len(match["indent"])
        depth = (len(match["indent"]) - base) // 4
        body = match["body"]
        # "Title: path", a bare "path", or a section heading "Title:".
        target = body.split(":", 1)[1].strip() if ":" in body else body
        items.append((depth, target))
    assert items, "no nav entries parsed out of mkdocs.yml"
    return items


class TestTheNavStaysBrowsable:
    """A sidebar of a few grouped sections, and nothing deeper."""

    def test_the_nav_lists_at_most_six_top_level_entries(self) -> None:
        tops = [target for depth, target in _nav_items() if depth == 0]
        assert tops, "no top-level nav entries parsed out of mkdocs.yml"
        assert len(tops) <= MAX_TOP_LEVEL_ENTRIES, (
            f"mkdocs.yml nav has {len(tops)} top-level entries; at most "
            f"{MAX_TOP_LEVEL_ENTRIES} fit the tab strip. Group the new page under "
            f"an existing tab instead of adding one."
        )

    def test_nothing_is_nested_more_than_one_section_below_a_tab(self) -> None:
        deepest = max(depth for depth, _ in _nav_items())
        assert deepest <= MAX_DEPTH, (
            f"mkdocs.yml nav nests {deepest} levels below the tabs; a further level "
            f"hides pages behind three clicks. Keep it to tabs, sections and their pages."
        )

    def test_every_nav_entry_points_at_a_page_on_disk(self) -> None:
        missing = sorted(target for _, target in _nav_items() if target and not (DOCS_DIR / target).is_file())
        assert not missing, f"mkdocs.yml nav points at pages that do not exist: {missing}"


class TestTheLandingPageShowsTheProduct:
    """The first screen: a budget, a call that builds, and honest output."""

    def test_the_landing_page_fits_its_line_budget(self) -> None:
        lines = LANDING_PAGE.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= MAX_LANDING_PAGE_LINES, (
            f"docs/index.md is {len(lines)} lines; the landing page's budget is "
            f"{MAX_LANDING_PAGE_LINES}. Move the detail onto the page that owns it "
            f"and link to it."
        )

    def test_the_page_leads_with_a_robot_not_a_drawing(self) -> None:
        """The first screen shows the product working, not how it is built.

        The hero carries a ``<robot-viewer>`` that autoloads: a robot the
        reader can move with a finger. The robot it names is one the viewer
        manifest can render, and no ``.svg`` (a hand-authored diagram on this
        site) comes before it. The diagram is not deleted; it belongs on the
        page that explains the design, where a reader has already asked.
        """
        page = LANDING_PAGE.read_text(encoding="utf-8")
        hero = _HERO_DIV.search(page)
        assert hero, "docs/index.md carries no sr-hero block; the first screen is the product working"
        viewer = _VIEWER.search(hero["body"])
        assert viewer, "the hero carries no <robot-viewer>; the first screen is a robot the reader can move"
        assert "autoload" in viewer["attrs"], "the hero viewer does not autoload; the first screen would be a poster"
        name = re.search(r'name="([a-z0-9_]+)"', viewer["attrs"])
        assert name, "the hero viewer names no robot"
        manifest = json.loads(VIEWER_MANIFEST.read_text(encoding="utf-8"))
        entries = manifest["robots"] if isinstance(manifest, dict) and "robots" in manifest else manifest
        entry = (
            entries.get(name[1])
            if isinstance(entries, dict)
            else next((e for e in entries if e.get("name") == name[1]), None)
        )
        assert entry and entry.get("sim"), f"the hero names {name[1]!r}, which the viewer manifest cannot render"
        for image in _IMAGE.finditer(page[: viewer.start() + hero.start()]):
            src = image["src"] or image["tag_src"]
            assert not src.endswith(DRAWING_SUFFIXES), (
                f"docs/index.md leads with {src}, a drawing, before the live robot. Move the "
                f"diagram to the page that explains the design and link to it."
            )

    def test_the_proof_numbers_are_derived_and_agree_with_the_package(self) -> None:
        """The three numbers under the hero are tokens the numbers hook fills.

        A typed count drifts (index.md once said 68 robots while the catalog
        said 73). Each number is a ``{{n:key}}`` the hook knows, and the two
        the package can answer directly, robots and native drivers, match it.
        """
        proof = _PROOF_DIV.search(LANDING_PAGE.read_text(encoding="utf-8"))
        assert proof, "docs/index.md carries no sr-proof block of numbers under the hero"
        tokens = _NUMBER_TOKEN.findall(proof["body"])
        assert len(tokens) >= 3, f"the proof block carries {len(tokens)} derived numbers; the design shows three"
        typed = re.findall(r"<strong>([^<]*\d[^<]*)</strong>", proof["body"])
        assert not typed, f"the proof block types a number by hand instead of a {{{{n:key}}}} token: {typed}"
        facts = _facts()
        unknown = sorted(set(tokens) - set(facts))
        assert not unknown, f"docs/index.md uses numbers keys the hook does not derive: {unknown}"
        shipped = {r["name"] for r in list_robots()} - set(_load_user_registry()["robots"])
        assert facts["robots"] == len(shipped), (
            f"the numbers hook says {facts['robots']} robots; the registry lists {len(shipped)} shipped robots."
        )
        assert facts["native_drivers"] == len(_SHIPPED_DRIVERS), (
            f"the numbers hook says {facts['native_drivers']} native drivers; "
            f"strands_robots.drivers ships {len(_SHIPPED_DRIVERS)}."
        )

    def test_every_next_step_card_lands_on_a_page_a_newcomer_finishes(self) -> None:
        """The cards hand the reader onward, not into the exhaustive treatment.

        The targets are read out of the page's card grid and measured the way
        the budget is stated, so a card re-pointed at a reference page fails
        here rather than on the reader's scroll.
        """
        targets = _next_step_card_targets()
        assert targets, "docs/index.md carries no next-step cards for a reader to follow"
        missing = sorted(target for target in targets if not (DOCS_DIR / target).is_file())
        assert not missing, f"docs/index.md cards point at pages that do not exist: {missing}"
        too_long = {target: len((DOCS_DIR / target).read_text(encoding="utf-8").split()) for target in targets}
        over = {t: n for t, n in too_long.items() if n > MAX_NEXT_STEP_WORDS}
        assert not over, (
            f"docs/index.md next-step cards land on pages over "
            f"{MAX_NEXT_STEP_WORDS} words: {over}. Point the card at the page a "
            f"newcomer reads first and let that page link onward, or trim it."
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
        """Every attribute the fences call on the robot exists on what they build.

        The sim fence is checked against the object it builds. The real fence
        needs an arm on USB, so its names are checked against the class the
        factory returns for its ``driver=``: the ``HardwareDriver`` protocol
        for ``driver="strands"``, the lerobot ``Robot`` wrapper otherwise.
        """
        calls = _attribute_uses_per_fence()
        assert calls, "the page no longer calls a method on the example's robot"
        sim_names = sorted({name for mode, _, names in calls if mode == "sim" for name in names})
        assert sim_names, "the sim fence calls nothing on the robot it builds"
        args, kwargs = _the_landing_robot_call()
        arm = Robot(*args, **kwargs)
        try:
            missing = [name for name in sim_names if not hasattr(arm, name)]
            assert not missing, f"docs/index.md calls methods the sim robot does not have: {missing}"
        finally:
            arm.destroy()
        for mode, driver, names in calls:
            if mode != "real":
                continue
            surface = HardwareDriver if driver == "strands" else HardwareRobot
            missing = [name for name in names if not hasattr(surface, name)]
            assert not missing, (
                f"docs/index.md's real fence (driver={driver!r}) calls names {surface.__name__} does not have: {missing}"
            )

    def test_the_two_fences_spell_one_api(self) -> None:
        """The sim fence and the real fence call the same verbs on the same object.

        The section is titled "One API, sim or real"; the claim is graded on the
        method names each fence uses, so a verb renamed in one fence fails here.
        """
        calls = {mode: names for mode, _, names in _attribute_uses_per_fence() if "send_action" in names}
        assert {"sim", "real"} <= set(calls), f"the page carries fences for {sorted(calls)}; it promises sim and real"
        shared = {"send_action", "cleanup"}
        for mode in ("sim", "real"):
            assert shared <= set(calls[mode]), f"the {mode} fence does not call {sorted(shared - set(calls[mode]))}"

    def test_the_second_pair_runs_one_checkpoint_in_sim_and_on_the_arm(self) -> None:
        """The section "The same checkpoint, sim or real" is graded on its fences.

        Both fences call ``run_policy`` then ``cleanup`` on the ``Robot`` they
        bind, and both name the same Hub checkpoint, so a fence that quietly
        drops back to poking joints, or runs a different checkpoint on the arm,
        fails here.
        """
        calls = {mode: names for mode, _, names in _attribute_uses_per_fence() if "run_policy" in names}
        assert {"sim", "real"} <= set(calls), f"the checkpoint pair carries fences for {sorted(calls)}; it promises sim and real"
        for mode in ("sim", "real"):
            assert "cleanup" in calls[mode], f"the {mode} checkpoint fence never calls cleanup"
        checkpoints = [
            set(re.findall(r"pretrained_name_or_path\W+([\w.-]+/[\w.-]+)", source))
            for source in _PYTHON_FENCE.findall(LANDING_PAGE.read_text(encoding="utf-8"))
        ]
        checkpoints = [c for c in checkpoints if c]
        assert len(checkpoints) == 2 and checkpoints[0] == checkpoints[1], (
            f"the two fences name different Hub checkpoints: {checkpoints}; the section promises one checkpoint"
        )


def _facts() -> dict[str, int]:
    """The numbers hook's table, loaded by path: the docs venv is not the test venv."""
    spec = importlib.util.spec_from_file_location("docs_facts_hook", FACTS_HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.numbers()


def _attribute_uses_per_fence() -> list[tuple[str, str, list[str]]]:
    """``(mode, driver, [attribute names])`` for every fence that binds a ``Robot(...)`` call.

    ``mode`` and ``driver`` are the call's keywords (factory defaults when
    absent); the names are every attribute read on the variable the call was
    bound to (``robot.send_action``, ``robot.tool_spec``).
    """
    out: list[tuple[str, str, list[str]]] = []
    for source in _PYTHON_FENCE.findall(LANDING_PAGE.read_text(encoding="utf-8")):
        tree = ast.parse(source)
        bound: dict[str, tuple[str, str]] = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", None) == "Robot"
                and isinstance(node.targets[0], ast.Name)
            ):
                keywords = {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords if kw.arg}
                bound[node.targets[0].id] = (keywords.get("mode", "sim"), keywords.get("driver", "auto"))
        for var, (mode, driver) in bound.items():
            names = sorted(
                {
                    node.attr
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == var
                }
            )
            out.append((mode, driver, names))
    return out


def _next_step_card_targets() -> list[str]:
    """Every page the landing page's card grid links to, in page order."""
    grid = _CARDS_DIV.search(LANDING_PAGE.read_text(encoding="utf-8"))
    assert grid, "docs/index.md no longer carries an sr-grid block of next-step cards"
    return [
        match["target"] for match in _MARKDOWN_LINK.finditer(grid["body"]) if not match["target"].startswith("http")
    ]


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
