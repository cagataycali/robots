# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every ROS 2 topic this tree publishes on is a name ROS 2 allows.

:class:`~strands_robots.ros_telemetry.RosTelemetryBase` is the single owner of
the topic names both hardware bridges publish on, and it builds them from
caller-supplied robot and camera names through ``_safe``, documented as mapping
a name to a *valid* ROS 2 topic segment. Nothing graded that claim against the
rule the tree already owns
(:data:`~strands_robots.rtps.mangling.ROS_TOPIC_RE`), and the two disagreed in
both directions:

* ``_safe`` admitted characters no ROS 2 name may hold, because ``str.isalnum``
  is true of every Unicode letter and digit - so a camera named ``cam2`` (with
  a superscript) produced ``/robot/cam2/image_raw``, which the RTPS bridge's
  own mangling then refused, from inside a telemetry publish rather than at the
  name that caused it.
* ``_safe`` emitted tokens ROS 2 refuses - a leading digit, repeated
  underscores - which the mangling minted anyway. Numeric names are this tree's
  own convention, not a contrived input: an so101's joints are named ``1``
  through ``6`` in its shipped sim MJCF, and a camera keyed by its device index
  is the ordinary case.

That second one is the silent half. DDS matching is by topic name, so the
participant advertises a topic ``rclpy`` refuses at ``create_publisher``, no
subscriber ever appears, and nothing anywhere reports it.

These cells close the loop: the producer's output is fed to the judge, so the
two cannot drift apart again.
"""

from __future__ import annotations

import pytest

from strands_robots.ros_telemetry import RosTelemetryBase
from strands_robots.rtps.mangling import ROS_TOPIC_RE, dds_topic_name, ros_topic_error

# Names a caller can genuinely arrive with. The observation keys these come from
# are whatever the robot reports, so every one of them reaches ``_safe``.
_HOSTILE_NAMES = (
    "front",  # the ordinary case, must be unchanged
    "wrist_cam",  # a single underscore is legal and must survive
    "0",  # a device index - ROS 2 forbids a leading digit
    "1",
    "3rd_person",
    "caf\u00e9",  # alnum to Python, not a ROS 2 name character
    "cam\u00b2",  # a superscript digit: isalnum() is True, isdigit() is True
    "arm0__wrist",  # the tree's own LeRobot-feature spelling of a namespaced camera
    "front  cam",  # a run of separators collapses to one underscore
    "front - cam",
    "!!!",  # nothing usable at all
    "_leading",
    "trailing_",
    "\u4e2d\u6587",  # non-Latin, entirely outside the ROS 2 character set
)


@pytest.mark.parametrize("name", _HOSTILE_NAMES)
def test_every_topic_the_telemetry_base_builds_is_a_ros2_name(name: str) -> None:
    """The producer's output has to satisfy the judge, for robot and camera alike."""
    for topic in (
        RosTelemetryBase.joint_states_topic(name),
        RosTelemetryBase.joint_command_topic(name),
        RosTelemetryBase.image_topic(name, name),
        RosTelemetryBase.image_topic("so101", name),
        RosTelemetryBase.image_topic(name, "front"),
    ):
        clause = ros_topic_error(topic)
        assert clause is None, f"{name!r} produced {topic!r}, which ROS 2 refuses: {clause}"
        # And the mangling mints it rather than raising from inside a publish.
        assert dds_topic_name(topic) == f"rt{topic}"


@pytest.mark.parametrize("name", _HOSTILE_NAMES)
def test_the_sanitised_segment_is_a_token_not_a_path(name: str) -> None:
    """A segment must not smuggle a separator, or one name becomes a namespace."""
    segment = RosTelemetryBase._safe(name)
    assert segment, f"{name!r} produced an empty segment"
    assert "/" not in segment
    assert ROS_TOPIC_RE.match(f"/{segment}"), segment


def test_an_ordinary_name_is_carried_through_untouched() -> None:
    """Tightening the sanitiser must not rename topics that were already valid.

    This is the control: the published names in the shipped examples and docs
    have to be byte-identical after the change, or the fix is a wire break.
    """
    assert RosTelemetryBase.joint_states_topic("so101") == "/so101/joint_states"
    assert RosTelemetryBase.joint_command_topic("so101") == "/so101/joint_command"
    assert RosTelemetryBase.image_topic("so101", "front") == "/so101/front/image_raw"
    assert RosTelemetryBase.image_topic("so101", "wrist_cam") == "/so101/wrist_cam/image_raw"


def test_two_index_named_cameras_stay_on_two_topics() -> None:
    """The digits have to be carried, not dropped.

    Prefixing is what makes a digit-leading name legal; replacing the digits
    with the fallback would put every index-named camera on one topic and
    interleave their frames.
    """
    topics = {RosTelemetryBase.image_topic("so101", str(index)) for index in range(4)}
    assert len(topics) == 4, topics
    assert RosTelemetryBase.image_topic("so101", "0") == "/so101/camera_0/image_raw"


def test_a_name_with_nothing_usable_names_what_it_stands_for() -> None:
    """The fallback says which seam it replaced, so a topic list stays readable."""
    assert RosTelemetryBase.joint_states_topic("!!!") == "/robot/joint_states"
    assert RosTelemetryBase.image_topic("so101", "!!!") == "/so101/camera/image_raw"
