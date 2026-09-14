"""Every teleop door refuses a device it cannot poll for an action.

A teleoperator is consumed identically wherever one is accepted: a background
loop calls ``get_action()`` once per tick and forwards the result. Only the LOCAL
attach door graded that contract. The mesh publish path did not, at either of
its two entry points, so a device with no callable ``get_action`` was started
rather than refused:

* ``Robot.start_teleop_publish`` returned ``status="success"`` with the topic and
  the ``start_teleop_receive(...)`` call for peers to subscribe with, while the
  loop it started raised ``AttributeError`` every tick, counted an error and
  published nothing. ``stats`` reported ``running``, and the loop stopped logging
  once its error budget was spent.
* Worse, that entry point STOPS a publisher already registered under the same
  ``device_name`` before constructing the new one. A device that could never be
  polled therefore cost the caller the working stream it was replacing.
* :class:`InputPublisher` accepted it at construction too, though the same
  constructor already refuses an unusable ``hz`` for exactly this reason - the
  loop it feeds runs on a thread where the mistake surfaces as a dead publisher
  that still reports ``running``.

All three now share one domain,
:func:`strands_robots.utils.teleoperator_contract_error`, so a device one door
turns away cannot be started by the next. The attach door's rows below are
unchanged by that convergence: they are the control that shows the two mesh doors
moved onto the answer the local one already gave.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.mesh.input import InputPublisher
from strands_robots.utils import teleoperator_contract_error
from tests.test_teleop import FakeHost, FakeTeleop
from tests.test_teleop_rate_and_duration_guards import _FakeMesh, _hardware_robot, _LivePublisher, _spin_until


class _NonCallableGetAction:
    """The attribute is present but cannot be called - the truthiness trap."""

    get_action = "yes"


class _NoneGetAction:
    get_action = None


class _NamedUnpollable:
    """An unpollable device with a deterministic repr, so refusal text compares."""

    def __repr__(self) -> str:
        return "<SO101Leader port=/dev/ttyACM0>"


#: Devices no teleop loop can poll. A bare ``object()`` is included because it is
#: what several suites passed as a stand-in teleoperator while nothing checked.
UNPOLLABLE: list[Any] = [None, object(), _NonCallableGetAction(), _NoneGetAction(), "so101_leader", 42]


def _pollable() -> FakeTeleop:
    return FakeTeleop({"a.pos": 1.0})


def _attach_door_refuses(device: Any) -> bool:
    try:
        FakeHost().attach_teleop(device, name="lead")
    except ValueError:
        return True
    return False


def _publish_entry_point_refuses(device: Any) -> bool:
    hw = _hardware_robot()
    result = hw.start_teleop_publish(teleoperator=device, hz=200.0)
    if result["status"] == "error":
        return True
    hw._input_publishers["leader"].stop()
    return False


def _constructor_refuses(device: Any) -> bool:
    try:
        InputPublisher(mesh=_FakeMesh(), teleoperator=device, hz=200.0)  # type: ignore[arg-type]
    except ValueError:
        return True
    return False


#: The three doors named in this module's docstring, each reduced to "did it
#: refuse?" so they can be compared to one another and to the shared domain.
DOORS = [
    ("attach_teleop", _attach_door_refuses),
    ("start_teleop_publish", _publish_entry_point_refuses),
    ("InputPublisher", _constructor_refuses),
]


class TestEveryDoorSharesOneDeviceDomain:
    """No door may start a device another door turns away."""

    @pytest.mark.parametrize("device", UNPOLLABLE, ids=lambda d: type(d).__name__)
    @pytest.mark.parametrize(("door", "refuses"), DOORS, ids=[d[0] for d in DOORS])
    def test_an_unpollable_device_is_refused_at_every_door(self, door, refuses, device):
        assert refuses(device) is True

    @pytest.mark.parametrize(("door", "refuses"), DOORS, ids=[d[0] for d in DOORS])
    def test_a_pollable_device_is_accepted_at_every_door(self, door, refuses):
        assert refuses(_pollable()) is False

    @pytest.mark.parametrize("device", [*UNPOLLABLE, _pollable()], ids=lambda d: type(d).__name__)
    def test_the_doors_agree_with_the_shared_domain(self, device):
        """One rule, one owner: each door's verdict IS the shared domain's."""
        shared_refuses = teleoperator_contract_error(device, "d", "c") is not None
        for _name, refuses in DOORS:
            assert refuses(device) is shared_refuses

    def test_every_door_refuses_in_the_shared_domains_words(self):
        """One rule, one wording - each door adds only its own name and param."""
        device = _NamedUnpollable()
        hw = _hardware_robot()
        envelope = hw.start_teleop_publish(teleoperator=device, hz=200.0)
        assert envelope["content"][0]["text"] == teleoperator_contract_error(
            device, "teleoperator", "start_teleop_publish"
        )
        with pytest.raises(ValueError) as constructed:
            InputPublisher(mesh=_FakeMesh(), teleoperator=device, hz=200.0)  # type: ignore[arg-type]
        assert str(constructed.value) == teleoperator_contract_error(device, "teleoperator", "InputPublisher")
        with pytest.raises(ValueError) as attached:
            FakeHost().attach_teleop(device, name="lead")
        assert str(attached.value) == teleoperator_contract_error(device, "device_or_spec", "attach_teleop")
        # The refused device is named, and so is the method the contract needs.
        assert repr(device) in envelope["content"][0]["text"]
        assert "must expose a callable get_action()" in envelope["content"][0]["text"]


class TestARefusedDeviceCostsNoLiveStream:
    """The refusal precedes the teardown the entry point performs."""

    def test_a_refused_device_leaves_a_live_publisher_running(self):
        hw = _hardware_robot()
        live = _LivePublisher()
        hw._input_publishers = {"leader": live}
        result = hw.start_teleop_publish(teleoperator=object(), device_name="leader", hz=200.0)
        assert result["status"] == "error"
        assert live.stopped is False
        assert hw._input_publishers == {"leader": live}

    def test_a_refused_device_registers_no_publisher_and_publishes_nothing(self):
        hw = _hardware_robot()
        assert hw.start_teleop_publish(teleoperator=object(), hz=200.0)["status"] == "error"
        assert getattr(hw, "_input_publishers", {}) == {}
        assert hw.mesh.published == []

    def test_a_pollable_device_replaces_the_live_publisher_and_puts_frames_on_the_wire(self):
        """The mirror: the teardown the guard precedes still happens when accepted."""
        hw = _hardware_robot()
        live = _LivePublisher()
        hw._input_publishers = {"leader": live}
        result = hw.start_teleop_publish(teleoperator=_pollable(), device_name="leader", hz=200.0)
        pub = hw._input_publishers["leader"]
        try:
            assert result["status"] == "success"
            assert live.stopped is True
            assert pub is not live
            assert _spin_until(lambda: len(hw.mesh.published) > 0)
            assert pub.stats["errors"] == 0
        finally:
            pub.stop()
