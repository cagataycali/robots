"""A ``use_rtps`` publish whose values do not fit the type is a named refusal, and nothing joins the graph.

Field *names* were already checked (``unknown field 'linar' for Twist``), but a
value of the wrong kind - ``3`` for the nested ``linear``, ``"fast"`` for a
float64, a string in ``JointState.position`` - reached cyclonedds' encoder inside
``writer.write`` and raised a bare ``Exception`` out of the tool, after a writer
had already been created on the topic. An unknown type's refusal also arrived
wrapped in the quotes ``str(KeyError)`` adds.
"""

from __future__ import annotations

import pytest

import strands_robots.rtps.idl as idl

pytestmark = pytest.mark.skipif(not idl.have_cyclonedds(), reason="requires the [ros2] extra (cyclonedds)")


def _text(result):
    return result["content"][0]["text"]


@pytest.fixture()
def no_writer(monkeypatch):
    from strands_robots.rtps import participant

    made = []
    monkeypatch.setattr(participant._backend, "writer", lambda topic, type: made.append(topic))
    monkeypatch.setattr(participant._backend, "available", lambda: True)
    return made


@pytest.mark.parametrize(
    ("ros_type", "fields", "member"),
    [
        ("geometry_msgs/msg/Twist", {"linear": 3}, "linear"),
        ("geometry_msgs/msg/Twist", {"linear": {"x": "fast"}}, "linear"),
        ("sensor_msgs/msg/JointState", {"name": ["a"], "position": ["x"]}, "position"),
        ("std_msgs/msg/Int32", {"data": 2**40}, "data"),
    ],
)
def test_a_value_of_the_wrong_kind_is_refused_before_a_writer_exists(no_writer, ros_type, fields, member):
    from strands_robots.tools.use_rtps import use_rtps

    result = use_rtps(action="publish", topic="/qa/kind", type=ros_type, fields=fields)
    assert result["status"] == "error"
    assert f"fields do not fit {ros_type}" in _text(result)
    assert f"member {member}" in _text(result)
    assert no_writer == []


def test_an_unknown_type_is_named_without_repr_quotes():
    from strands_robots.tools.use_rtps import use_rtps

    text = _text(use_rtps(action="publish", topic="/qa/kind", type="foo_msgs/msg/Bar", fields={}))
    assert "publish failed: 'foo_msgs/msg/Bar' is not in the RTPS IDL bundle" in text
