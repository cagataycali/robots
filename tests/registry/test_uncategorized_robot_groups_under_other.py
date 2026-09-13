"""A robot that declares no category is grouped, named and ordered as ``other``.

``category`` is optional in both the package registry and the user overlay, and
``register_robot`` accepts ``category=""``, so a robot that declares none is a
reachable state. :func:`list_robots` reports its category as ``""`` - honest,
because that is what the entry says - and every grouping surface downstream then
has to name a group for it.

``list_robots_by_category`` read that key by presence
(``robot.get("category", "other")``) rather than by value. Its own input always
supplies the key, so the ``"other"`` fallback could never fire and such a robot
was returned under ``""``: a group with no name to switch on, which
``format_robot_table`` then rendered as a blank Category cell sorted ahead of
every declared custom category.

These pins hold the three surfaces that must agree on one group name: the
grouping, the rendered cell, and the display order.
"""

from __future__ import annotations

import pytest

from strands_robots.registry import list_robots, list_robots_by_category, register_robot
from strands_robots.registry.robots import format_robot_table


def _register(tmp_path, name: str, category: str) -> None:
    robot_dir = tmp_path / name
    robot_dir.mkdir(parents=True, exist_ok=True)
    (robot_dir / "bot.xml").write_text(f'<mujoco model="{name}"><worldbody/></mujoco>')
    register_robot(
        name=name,
        model_xml="bot.xml",
        asset_dir=str(robot_dir),
        category=category,
        joints=4,
        description=f"a {category or 'category-less'} robot",
        overwrite=True,
    )


@pytest.mark.parametrize("declared", ["", "   ", "\t"])
def test_a_robot_declaring_no_category_is_grouped_under_other(tmp_path, declared):
    """The group is named, not blank - and the name is the documented one."""
    _register(tmp_path, "mysterybot", declared)

    by_cat = list_robots_by_category()
    group = next(key for key, rows in by_cat.items() if any(r["name"] == "mysterybot" for r in rows))

    assert group == "other", f"grouped under {group!r}, which names no group a caller can switch on"
    assert "" not in by_cat


@pytest.mark.parametrize("declared", ["quadruped", "  quadruped  "])
def test_a_declared_category_is_still_its_own_group(tmp_path, declared):
    """The control: a category the display order does not enumerate keeps its name.

    A padded spelling joins that same group rather than opening a second one,
    while ``list_robots`` keeps reporting the entry verbatim.
    """
    _register(tmp_path, "quadbot", declared)

    by_cat = list_robots_by_category()

    assert [r["name"] for r in by_cat["quadruped"]] == ["quadbot"]
    assert "other" not in by_cat
    assert next(r["category"] for r in list_robots() if r["name"] == "quadbot") == declared


def test_every_robot_lands_in_exactly_one_group(tmp_path):
    """Grouping partitions the registry - naming the fallback cannot drop a row."""
    _register(tmp_path, "mysterybot", "")
    _register(tmp_path, "quadbot", "quadruped")

    by_cat = list_robots_by_category()

    grouped = [r["name"] for rows in by_cat.values() for r in rows]
    assert sorted(grouped) == sorted(r["name"] for r in list_robots())


def test_the_table_renders_the_group_it_grouped_the_robot_in(tmp_path):
    """The Category cell names the group, rather than being blank."""
    _register(tmp_path, "mysterybot", "")

    row = next(ln for ln in format_robot_table(max_width=1000).split("\n") if ln.startswith("mysterybot"))

    assert row.split()[1] == "other", f"Category cell reads {row.split()[1]!r}"


def test_a_category_less_robot_is_ordered_after_a_declared_one(tmp_path):
    """``other`` is the absence of a category, so it cannot outrank a real one."""
    _register(tmp_path, "mysterybot", "")
    _register(tmp_path, "quadbot", "quadruped")

    body = [
        ln.split()[0] for ln in format_robot_table(max_width=1000).split("\n")[2:] if ln.strip() and "Total:" not in ln
    ]

    assert body.index("mysterybot") > body.index("quadbot")
