"""Unit tests for ROS 2 <-> DDS name mangling (pure string transforms).

Beyond the happy-path transforms, these pin one property: every name this module
returns is a name it would itself accept, and a ROS 2 name it cannot map is
refused rather than mangled into something plausible.

Two edges settle that. ``dds_type_name`` maps *message* interfaces only - ROS 2
generates one DDS type per constituent message of a service or an action, so
there is no single type to return for one, and an invented
``pkg::srv::dds_::Name_`` is what the participant would then advertise in DDS
discovery for a struct ROS 2 never generates. ``ros_topic_name`` checks the name
it recovers against the same rule ``dds_topic_name`` applies, because a DDS graph
carries topics no ROS 2 node published, so stripping the prefix alone does not
yield a ROS 2 name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import strands_robots
import strands_robots.mesh.rtps_robot as rtps_robot_module
import strands_robots.tools.use_rtps as use_rtps_module
from strands_robots.rtps.idl import REGISTRY, have_cyclonedds
from strands_robots.rtps.mangling import (
    MAX_DDS_TOPIC_LENGTH,
    ROS_TOPIC_RE,
    dds_topic_name,
    dds_type_name,
    ros_topic_error,
    ros_topic_name,
)

# Names ROS 2 refuses, so this module's topic rule has to refuse them too.
# Shared by both directions so the two cannot come to disagree about what a
# valid ROS 2 topic name is.
#
# The rule is the mapping article's, not a subset of it: a name only this module
# accepts still maps to a DDS topic, and nothing downstream reports the
# divergence, because DDS matches by topic name and simply never finds a peer.
# So each clause the article states gets a row here, keyed by the clause, and
# the six that a single "invalid topic name" cannot distinguish are the reason
# the refusal names one.
_MALFORMED_ROS_TOPICS = (
    "cmd_vel",  # not absolute
    "",  # empty
    "/",  # ends with '/'
    "/a/",  # ends with '/'
    "/bad name",  # character outside [A-Za-z0-9_/]
    "/x;y",  # character outside [A-Za-z0-9_/]
    "/caf\u00e9",  # a Unicode letter is alnum to Python and not a ROS 2 name character
    "//cmd_vel",  # empty leading token: "the name //bar is not allowed"
    "/a//b",  # empty interior token
    "/1cam",  # token starts with a digit
    "/a/2b",  # interior token starts with a digit
    "/a__b",  # repeated underscore
    "/__leading",  # repeated underscore at the token start
)

# Names a real ROS 2 graph does carry. Held beside the refusals because
# tightening a rule is only correct if it narrows to exactly the article's set:
# a leading single underscore marks a hidden topic and is legal, and a digit
# inside a token (``turtle1``) is legal - only a leading one is not.
_WELL_FORMED_ROS_TOPICS = (
    "/turtle1/cmd_vel",
    "/cmd_vel",
    "/a/b/c",
    "/j/state",
    "/_hidden/joint_states",
    "/a1/b2",
    "/so101/camera_0/image_raw",
)


@pytest.mark.parametrize("ros", _WELL_FORMED_ROS_TOPICS)
def test_topic_roundtrip(ros: str) -> None:
    """Every name ROS 2 allows survives the mapping and comes back unchanged."""
    assert dds_topic_name(ros) == f"rt{ros}"
    assert ros_topic_name(f"rt{ros}") == ros


@pytest.mark.parametrize("bad", _MALFORMED_ROS_TOPICS)
def test_topic_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid ROS 2 topic"):
        dds_topic_name(bad)


@pytest.mark.parametrize("bad", [t for t in _MALFORMED_ROS_TOPICS if t.startswith("/")])
def test_ros_topic_name_never_hands_back_a_name_dds_topic_name_refuses(bad: str) -> None:
    """The documented inverse has to survive a DDS topic that is not a ROS 2 one.

    A subscriber enumerating a DDS graph feeds whatever it discovered here. If
    stripping the prefix is the whole transform, a caller gets a "ROS 2 topic
    name" this module refuses, and the failure surfaces at whatever it hands the
    name to next rather than at the name it came from.
    """
    dds_topic = "rt" + bad
    try:
        recovered = ros_topic_name(dds_topic)
    except ValueError:
        return
    try:
        dds_topic_name(recovered)
    except ValueError as exc:
        pytest.fail(
            f"ros_topic_name({dds_topic!r}) handed back {recovered!r}, a name dds_topic_name itself "
            f"refuses ({exc}), so the documented inverse does not hold"
        )
    pytest.fail(f"premise: dds_topic_name({recovered!r}) was expected to be refused")


@pytest.mark.parametrize("ros", _WELL_FORMED_ROS_TOPICS)
def test_a_valid_topic_survives_the_recovered_name_check(ros: str) -> None:
    """Checking the recovered name must not narrow what a valid graph can carry."""
    dds = f"rt{ros}"
    assert ros_topic_name(dds) == ros
    assert dds_topic_name(ros_topic_name(dds)) == dds


def test_ros_topic_name_requires_prefix() -> None:
    with pytest.raises(ValueError, match="does not carry"):
        ros_topic_name("/turtle1/cmd_vel")  # missing the rt prefix


@pytest.mark.parametrize(
    ("ros", "dds"),
    [
        ("geometry_msgs/msg/Twist", "geometry_msgs::msg::dds_::Twist_"),
        ("sensor_msgs/msg/LaserScan", "sensor_msgs::msg::dds_::LaserScan_"),
        ("turtlesim/msg/Pose", "turtlesim::msg::dds_::Pose_"),
    ],
)
def test_type_mangling(ros: str, dds: str) -> None:
    assert dds_type_name(ros) == dds


@pytest.mark.parametrize("bad", ["Twist", "geometry_msgs/Twist", "a/b/c/d", "pkg/badkind/Name"])
def test_type_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid ROS 2 type"):
        dds_type_name(bad)


@pytest.mark.parametrize(
    ("ros_type", "generated"),
    [
        (
            "example_interfaces/srv/AddTwoInts",
            ("example_interfaces::srv::dds_::AddTwoInts_Request_", "AddTwoInts_Response_"),
        ),
        ("std_srvs/srv/Trigger", ("std_srvs::srv::dds_::Trigger_Request_", "Trigger_Response_")),
        (
            "control_msgs/action/FollowJointTrajectory",
            ("control_msgs::action::dds_::FollowJointTrajectory_Goal_", "_Result_"),
        ),
    ],
)
def test_a_service_interface_is_refused_rather_than_given_a_type_ros2_never_generates(
    ros_type: str, generated: tuple[str, ...]
) -> None:
    """A service has no single DDS type, so there is nothing to return for one.

    ``rosidl`` renders its message template once per constituent message, so a
    service yields ``Name_Request_`` and ``Name_Response_`` structs (an action
    yields goal/result/feedback plus two nested services) and never a ``Name_``.
    Returning ``pkg::srv::dds_::Name_`` is what the participant then advertises in
    DDS discovery, for a struct that exists nowhere in the ROS 2 type system, and
    nothing reports it: matching is by topic name, so the wrong name does not
    even surface as a failure to connect. The refusal has to name the types ROS 2
    does generate, or it just moves the dead end one call further out.
    """
    try:
        minted = dds_type_name(ros_type)
    except ValueError as exc:
        message = str(exc)
        for expected in generated:
            assert expected in message, f"the refusal {message!r} does not name {expected!r}"
        return
    pytest.fail(
        f"dds_type_name({ros_type!r}) returned {minted!r}, but ROS 2 generates no such struct: the "
        f"wire types are {' / '.join(generated)}. That name is what the participant would advertise "
        "in DDS discovery, and nothing reports the disagreement."
    )


def test_the_malformed_type_refusal_does_not_offer_an_interface_kind_it_will_not_map() -> None:
    """A refusal must not send the caller after a spelling that is also refused."""
    with pytest.raises(ValueError) as excinfo:
        dds_type_name("Twist")
    message = str(excinfo.value)
    assert "pkg/msg/Name" in message
    for unmappable in ("pkg/srv/Name", "pkg/action/Name"):
        assert unmappable not in message, f"the refusal {message!r} offers {unmappable!r}, which it also refuses"


def test_an_unknown_interface_kind_is_still_reported_as_malformed() -> None:
    """Only ``srv``/``action`` get the mapping explanation; anything else is a typo."""
    with pytest.raises(ValueError, match="invalid ROS 2 type"):
        dds_type_name("pkg/badkind/Name")


@pytest.mark.skipif(not have_cyclonedds(), reason="cyclonedds not installed ([ros2] extra)")
def test_every_bundled_message_type_mangles_to_the_name_cyclonedds_puts_on_the_wire() -> None:
    """The message path is graded against the binding that carries it, not a copy.

    ``cyclonedds`` derives a topic's wire type name from the IDL dataclass, so
    the bundle's ``typename=`` annotations are what a real ROS 2 node matches
    against. Mangling has to agree with them for every bundled type.
    """
    assert len(REGISTRY) >= 9, f"premise: the IDL bundle looks empty ({sorted(REGISTRY)})"
    for ros_type, idl_cls in sorted(REGISTRY.items()):
        on_the_wire = idl_cls.__idl_typename__
        assert dds_type_name(ros_type) == on_the_wire, ros_type


def test_a_refusal_names_the_clause_the_name_broke() -> None:
    """One message for every mistake makes the caller guess which one they made.

    The rule has six independent clauses and a name can break exactly one of
    them, so a refusal reading only "invalid topic name" leaves the caller to
    re-derive the grammar. Each row below asserts the reported clause, which is
    also what pins that the clauses are reachable rather than dead branches
    behind whichever check happens to run first.
    """
    reported = {}
    for name in _MALFORMED_ROS_TOPICS:
        clause = ros_topic_error(name)
        assert clause is not None, f"{name!r} is in the malformed table and was accepted"
        reported[name] = clause
    assert "absolute" in reported["cmd_vel"]
    assert "absolute" in reported[""]
    assert "must not end with '/'" in reported["/a/"]
    assert "ASCII" in reported["/caf\u00e9"]
    assert "ASCII" in reported["/bad name"]
    assert "token must not be empty" in reported["//cmd_vel"]
    assert "token must not be empty" in reported["/a//b"]
    assert "'1cam'" in reported["/1cam"] and "digit" in reported["/1cam"]
    assert "'2b'" in reported["/a/2b"] and "digit" in reported["/a/2b"]
    assert "repeated underscores" in reported["/a__b"]
    # Six distinct clauses are reachable, so the message carries information
    # the caller could not have got from the refusal alone.
    assert len(set(reported.values())) >= 6, sorted(set(reported.values()))


@pytest.mark.parametrize("ros", _WELL_FORMED_ROS_TOPICS)
def test_ros_topic_error_reports_nothing_for_a_name_ros2_allows(ros: str) -> None:
    assert ros_topic_error(ros) is None


def test_ros_topic_error_and_the_pattern_are_one_verdict() -> None:
    """The message renders the pattern's verdict; it must not be a second rule.

    Two spellings of one rule is the defect this module's own docstring warns
    about, so the renderer is graded against the pattern over both tables rather
    than trusted to agree with it.
    """
    for name in _WELL_FORMED_ROS_TOPICS + _MALFORMED_ROS_TOPICS:
        assert (ros_topic_error(name) is None) is bool(ROS_TOPIC_RE.match(name)), name


def test_the_dds_length_bound_is_enforced_in_both_directions() -> None:
    """The mapping bounds the DDS name, so the bound is checked on the mangled form.

    A ROS topic one character inside the bound has a DDS name two characters
    longer, so a bound applied to the ROS name would let the DDS name past it.
    """
    longest_ros = "/" + "a" * (MAX_DDS_TOPIC_LENGTH - len("rt/"))
    assert len(dds_topic_name(longest_ros)) == MAX_DDS_TOPIC_LENGTH
    assert ros_topic_name(dds_topic_name(longest_ros)) == longest_ros

    over = longest_ros + "a"
    with pytest.raises(ValueError, match=f"bounds a DDS topic name at {MAX_DDS_TOPIC_LENGTH}"):
        dds_topic_name(over)
    with pytest.raises(ValueError, match=f"bounds a DDS topic name at {MAX_DDS_TOPIC_LENGTH}"):
        ros_topic_name(f"rt{over}")


def test_the_topic_rule_is_spelled_once_in_the_package() -> None:
    """Every seam that gates "is this a ROS 2 topic" reads the one pattern.

    The rule was spelled three times - here, in the ``use_rtps`` tool, and in
    the RTPS mobile-base robot - as byte-identical copies, so tightening one
    left the other two admitting names the mangling refuses. Grading the source
    is what keeps a fourth seam from starting its own copy: a regex literal is
    invisible to any behavioural test that does not happen to drive that seam.
    """
    package = Path(strands_robots.__file__).parent
    literal = ROS_TOPIC_RE.pattern
    spellings = sorted(
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if literal in path.read_text(encoding="utf-8")
    )
    assert spellings == ["rtps/mangling.py"], spellings

    # And the two seams that had a copy now read this one. ``use_rtps`` reads it
    # through ``ros_topic_error`` - the rendering of this pattern's verdict, which
    # early-returns on it - rather than by binding the pattern under a name of its
    # own. That is the stronger form of the property this cell is here to hold:
    # the tool cannot answer "is this a ROS 2 topic" differently from the mangling
    # because it does not answer the question at all, and it keeps no topic-rule
    # name that a later edit could quietly repoint. Asserting an alias no caller
    # read would grade a vestige instead of the seam.
    assert use_rtps_module.ros_topic_error is ros_topic_error
    assert not any(name.endswith("_TOPIC_RE") for name in vars(use_rtps_module))
    assert rtps_robot_module.RtpsRobot._TOPIC_RE is ROS_TOPIC_RE
