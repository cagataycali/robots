"""Tests for the RTPS IDL bundle.

Skips the cyclonedds-dependent assertions when the [ros2] extra is not
installed, but always pins the no-backend error contract.
"""

from __future__ import annotations

import pytest

import strands_robots.rtps.idl as idl


def test_get_type_without_cyclonedds_raises_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(idl, "_HAVE_CYCLONEDDS", False)
    with pytest.raises(ImportError, match=r"strands-robots\[ros2\]"):
        idl.get_type("geometry_msgs/msg/Twist")


def test_have_cyclonedds_matches_registry_population() -> None:
    # When cyclonedds is present the registry is non-empty; otherwise empty.
    if idl.have_cyclonedds():
        assert idl.REGISTRY, "cyclonedds present but IDL REGISTRY is empty"
    else:
        assert idl.REGISTRY == {}


@pytest.mark.skipif(not idl.have_cyclonedds(), reason="requires the [ros2] extra (cyclonedds)")
def test_bundle_typenames_match_ros_dds_mapping() -> None:
    from strands_robots.rtps.mangling import dds_type_name

    for ros_type, cls in idl.REGISTRY.items():
        # The IDL dataclass typename must equal the ROS-on-DDS mangled name so a
        # real ROS 2 node accepts the sample.
        assert cls.__idl_typename__ == dds_type_name(ros_type), ros_type


@pytest.mark.skipif(not idl.have_cyclonedds(), reason="requires the [ros2] extra (cyclonedds)")
def test_unknown_type_lists_known() -> None:
    with pytest.raises(KeyError, match="not in the RTPS IDL bundle"):
        idl.get_type("custom_msgs/msg/Nope")
