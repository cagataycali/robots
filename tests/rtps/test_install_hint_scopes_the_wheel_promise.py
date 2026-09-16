# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``[ros2]`` extra's remedies scope their "self-contained wheel" promise.

``pip install 'strands-robots[ros2]'`` is a wheel on macOS, Windows and Linux
x86_64. No cyclonedds release publishes a Linux aarch64 wheel, so on a Jetson or
a robot's onboard PC the same command builds the sdist against a Cyclone DDS C
install (``CYCLONEDDS_HOME``). Every remedy that offers the extra must say so -
"self-contained pip wheel" alone sends the aarch64 user back to a command that
cannot succeed, and the rclpy refusal offers the extra as the *escape* from an
install that already failed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands_robots.hardware_robot import _RCLPY_TRANSPORT_INSTALL_HINT
from strands_robots.rtps.idl import _INSTALL_HINT
from tests._blocked_module import blocked

_DOCS = Path(__file__).resolve().parents[2] / "docs"
_ANCHOR = "docs/rtps-integration.md#linux-aarch64-jetson"
_HEADING = "### Linux aarch64 (Jetson)"


def _aarch64_section() -> str:
    text = (_DOCS / "rtps-integration.md").read_text()
    assert _HEADING in text, f"{_HEADING!r} is the section every remedy points at"
    return text.split(_HEADING, 1)[1]


@pytest.mark.parametrize(
    ("name", "hint"),
    [
        ("rtps.idl._INSTALL_HINT", _INSTALL_HINT),
        ("hardware_robot._RCLPY_TRANSPORT_INSTALL_HINT", _RCLPY_TRANSPORT_INSTALL_HINT),
    ],
)
def test_every_remedy_offering_the_extra_names_the_aarch64_route(name: str, hint: str) -> None:
    """A remedy that prints the install command must print its one condition too."""
    assert "pip install 'strands-robots[ros2]'" in hint, name
    assert "aarch64" in hint, f"{name} offers the extra without naming the arch that has no wheel:\n{hint}"
    assert "CYCLONEDDS_HOME" in hint, f"{name} names no C install to point at:\n{hint}"
    assert _ANCHOR in hint, f"{name} does not say where the route is written up:\n{hint}"


def test_the_rclpy_refusal_a_blocked_caller_reads_carries_the_condition() -> None:
    """The escape hatch from a missing rclpy must not be a second failing install.

    ``ros2_bridge=True`` without a sourced distro refuses with the RTPS transport
    as the alternative. On aarch64 that alternative needs a Cyclone DDS C install
    of its own, so the refusal names it rather than handing the caller the same
    ``pip install`` twice.

    Asserted on the message the production guard really raises, with the import
    made to fail by :func:`tests._blocked_module.blocked` rather than assumed
    absent. Reading the host instead does not hold either way: skipping when
    ``rclpy`` is importable grades nothing on a machine with a distro sourced,
    and on a machine without one the refusal still does not fire, because
    ``require_optional`` answers from its memo and a sibling suite leaves a fake
    ``rclpy`` in it. ``blocked`` clears that memo and restores it, so this holds
    wherever it runs and whatever ran before it.
    """
    from strands_robots.hardware_robot import Robot

    with blocked("rclpy"), pytest.raises(ImportError) as excinfo:
        Robot._check_ros2_bridge_deps(ros2_transport="rclpy")
    text = str(excinfo.value)
    assert "ros2_transport='rtps'" in text, text
    assert "Linux aarch64" in text, f"the offered alternative hides its aarch64 condition:\n{text}"
    assert _ANCHOR in text, text


def test_the_anchor_resolves_to_a_recipe_with_both_routes() -> None:
    section = _aarch64_section()
    assert "CYCLONEDDS_HOME=/opt/ros/$ROS_DISTRO" in section, "the sourced-distro route"
    assert "eclipse-cyclonedds/cyclonedds" in section, "the build-from-source route"
    for page in ("ros2-integration.md", "troubleshooting.md"):
        assert "rtps-integration.md#linux-aarch64-jetson" in (_DOCS / page).read_text(), page


def test_the_recipe_says_when_cyclonedds_home_is_needed_at_runtime() -> None:
    """``CYCLONEDDS_HOME`` is a loader override, not an unconditional runtime need.

    The binding tries a wheel's bundled library, then ``$CYCLONEDDS_HOME/lib``,
    then the normal loader path - so an install under the default ``/usr/local``
    prefix imports with no variable set, while a custom prefix needs one. Set to
    a wrong prefix it raises instead of falling back, so a stale export breaks an
    otherwise working install. A blanket "keep it set" hides both facts.
    """
    section = _aarch64_section()
    assert "$CYCLONEDDS_HOME/lib" in section, "the recipe should say what the variable overrides"
    assert "ldconfig" in section, "the default-prefix case needs no variable - say so"
    assert "CycloneDDSLoaderException" in section, "a stale prefix fails hard; name the error"
