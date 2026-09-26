# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A pure-RTPS "cannot tell it apart" claim names the graph metadata that differs.

Both hardware bridges publish the same topics, and a real ``rclpy`` subscriber
decodes their ``JointState`` field for field - that part of the promise holds.
What a bare DDS participant cannot carry is ROS 2 *graph* metadata: it has no
node name and no type hash, so ``ros2 node list`` does not list it, ``ros2 topic
info -v`` reports its publisher as ``_CREATED_BY_BARE_DDS_APP_`` with an
``INVALID`` type hash, and every ``rmw_cyclonedds_cpp`` subscriber logs one
"Failed to parse type hash" warning. A reader told the stock ``ros2`` CLI "cannot
tell them apart" is told the opposite by the first command that page suggests.

So a claim about what a ROS 2 observer cannot distinguish carries its condition,
in the surface that makes it - the same rule
``tests/rtps/test_install_hint_scopes_the_wheel_promise.py`` applies to the
``[ros2]`` extra's "self-contained wheel" promise.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

#: Where the four axes are written up; every qualification points here.
_PAGE = _ROOT / "docs" / "ros2" / "rtps-robot.md"
_SECTION = "## What a ROS 2 node can still tell apart"
_ANCHOR = "what-a-ros-2-node-can-still-tell-apart"

# A claim of indistinguishability ...
_CLAIM = re.compile(r"indistinguishable|cannot tell", re.IGNORECASE)
# ... whose arbiter is a ROS 2 observer rather than a byte comparison ...
# Prose wraps, so every multi-word arbiter tolerates a line break inside it.
_OBSERVER = re.compile(
    r"ROS\s+2\s+(?:node|graph|stack)|a\s+real\s+node|ros2\s+(?:topic|node)|rviz|nav2|stock\s+`?ros2`?",
    re.IGNORECASE,
)
# ... about the pure-RTPS path, named either outright or as the pair of
# transports (a claim about the two rclpy bridges alone really is one class, and
# is true) ...
_SUBJECT = re.compile(r"rtps|cyclonedds|bare\s+DDS|(?:the|two|both)\s+transports", re.IGNORECASE)
#: Which files can hold an offender at all - the narrower per-paragraph subject
#: above decides which paragraph in them does.
_RTPS = re.compile(r"rtps|cyclonedds|bare\s+DDS", re.IGNORECASE)
# ... must name what the graph still shows.
_LIMIT = re.compile(
    r"_CREATED_BY_BARE_DDS_APP_|type\s+hash|no\s+node\s+name|not\s+a\s+ROS\s+2\s+\*?node|" + _ANCHOR,
    re.IGNORECASE,
)


def _shipped_surfaces() -> list[Path]:
    """Every shipped page and module that mentions the pure-RTPS transport."""
    candidates = [
        *(_ROOT / "docs").rglob("*.md"),
        *(_ROOT / "examples").rglob("*.py"),
        *(_ROOT / "strands_robots").rglob("*.py"),
    ]
    return sorted(p for p in candidates if _RTPS.search(p.read_text(encoding="utf-8")))


def _unqualified_claims(text: str) -> list[str]:
    """Paragraphs claiming an RTPS path is indistinguishable, with no condition."""
    return [
        para
        for para in re.split(r"\n\s*\n", text)
        if _CLAIM.search(para) and _OBSERVER.search(para) and _SUBJECT.search(para) and not _LIMIT.search(para)
    ]


def test_the_page_documents_every_axis_the_graph_exposes() -> None:
    """The section every qualification cites names what each command reports."""
    text = _PAGE.read_text(encoding="utf-8")
    assert _SECTION in text, f"{_PAGE.name} carries no {_SECTION!r} section to point at"
    section = text.split(_SECTION, 1)[1].split("\n## ", 1)[0]
    for axis in ("ros2 node list", "ros2 topic info -v", "_CREATED_BY_BARE_DDS_APP_", "INVALID", "KEEP_LAST"):
        assert axis in section, f"{_SECTION!r} does not name {axis!r}"


def test_every_surface_citing_the_section_resolves_to_its_heading() -> None:
    """A qualification is only usable if the anchor it names exists."""
    citing = [p for p in _shipped_surfaces() if _ANCHOR in p.read_text(encoding="utf-8")]
    assert citing, f"no surface points at #{_ANCHOR}; the qualification has no write-up"
    slug = _SECTION.removeprefix("## ").lower().replace(" ", "-")
    assert slug == _ANCHOR, f"the section slugs to {slug!r}, not the cited {_ANCHOR!r}"


def test_no_shipped_surface_claims_a_ros2_observer_cannot_tell_the_rtps_path_apart() -> None:
    """The promise is about payloads; a graph-wide version of it is not true."""
    surfaces = _shipped_surfaces()
    assert len(surfaces) >= 10, f"only {len(surfaces)} surfaces mention RTPS; the harvest is broken"
    offenders = {p.relative_to(_ROOT).as_posix(): _unqualified_claims(p.read_text(encoding="utf-8")) for p in surfaces}
    unqualified = {path: paras for path, paras in offenders.items() if paras}
    assert not unqualified, "an RTPS indistinguishability claim names no graph-metadata limit:\n" + "\n".join(
        f"{path}:\n{paras[0].strip()}" for path, paras in unqualified.items()
    )


@pytest.mark.parametrize(
    ("label", "paragraph", "flagged"),
    [
        (
            "the shape this rule exists for",
            "The two transports emit byte-identical topics, so a real ROS 2 node (or\n"
            "`ros2 topic echo`) cannot tell them apart on the wire:",
            True,
        ),
        (
            "qualified by the anchor",
            "The two transports emit byte-identical topics; a real ROS 2 node reads either,\n"
            "though the graph differs: docs/ros2/rtps-robot.md#what-a-ros-2-node-can-still-tell-apart.",
            False,
        ),
        (
            "qualified by naming the metadata",
            "A real ROS 2 node cannot tell the cyclonedds samples from hardware's. It is not a\n"
            "ROS 2 node, so `ros2 topic info -v` reports `_CREATED_BY_BARE_DDS_APP_`.",
            False,
        ),
        (
            "qualified by naming the type hash",
            "rviz cannot tell this RTPS publisher from hardware; it carries no type hash.",
            False,
        ),
        (
            "the tool docstring's spelling: use_rtps behind an underscore, a bare real node",
            "Unlike ``use_ros``, an RTPS participant can act as a robot: publish topics a real\n"
            "node will consume - indistinguishable on the wire from hardware.",
            True,
        ),
        (
            "a payload claim with no observer as arbiter",
            "``HardwareRtpsBridge`` and ``HardwareRosBridge`` derive from one base, so they are\n"
            "byte-compatible on the wire by construction.",
            False,
        ),
        (
            "the rclpy sim/hardware pair, which really is one class",
            "A simulated robot and the real one it mirrors are indistinguishable on the ROS 2\n"
            "graph, because both bridges are subclasses of one rclpy publisher.",
            False,
        ),
        (
            "an RTPS paragraph making no such claim",
            "Type coverage is bounded by the IDL bundle: joint_states and image_raw are in,\n"
            "anything else needs the rclpy backend.",
            False,
        ),
    ],
)
def test_the_rule_flags_the_claim_and_not_its_neighbours(label: str, paragraph: str, flagged: bool) -> None:
    """Table-driven falsification: the pattern discriminates, in both directions."""
    assert bool(_unqualified_claims(paragraph)) is flagged, label
