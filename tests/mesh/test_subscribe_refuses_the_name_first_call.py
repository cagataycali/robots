"""``Mesh.subscribe(topic, callback=None, name=None)`` refuses arguments in the wrong places.

The mesh pages showed ``a.mesh.subscribe("imu", "strands/arm-b/imu", lambda key, payload: ...)``,
name first. Run as written it subscribed to the literal key ``imu`` (so the
callback never fired), stored the lambda as the subscription name, and the next
``stop()`` raised ``TypeError: sequence item 0: expected str instance, function
found``. The docs now pass the topic first, and the call is checked.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import Mesh
from tests.mesh.test_deep_mesh import FakeRobot, clean_state, mock_session  # noqa: F401


def test_the_documented_name_first_call_is_refused_before_anything_is_recorded(mock_session) -> None:  # noqa: F811
    m = Mesh(FakeRobot(), peer_id="sub-order-1")
    m.start()
    with pytest.raises(TypeError, match="did you pass the name first"):
        m.subscribe("imu", "strands/arm-b/imu", lambda key, payload: None)  # type: ignore[arg-type]
    assert "imu" not in m.inbox
    m.stop()  # the lambda never became a subscription name


def test_a_non_string_name_or_topic_is_refused(mock_session) -> None:  # noqa: F811
    m = Mesh(FakeRobot(), peer_id="sub-order-2")
    m.start()
    with pytest.raises(TypeError, match="name must be a string"):
        m.subscribe("strands/arm-b/imu", None, lambda k, p: None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="topic must be"):
        m.subscribe("", None)
    m.stop()


def test_the_corrected_call_subscribes_under_its_name(mock_session) -> None:  # noqa: F811
    pytest.importorskip("zenoh")
    m = Mesh(FakeRobot(), peer_id="sub-order-3")
    m.start()
    assert m.subscribe("strands/arm-b/imu", lambda key, payload: None, name="imu") == "imu"
    key = mock_session.declare_subscriber.call_args_list[-1].args[0]
    assert key == "strands/arm-b/imu"
    m.stop()
