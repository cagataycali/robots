"""Every DDS decoder absorbs a sample it cannot read, on all three DDS drivers.

A decoder runs on a thread the vendor SDK owns, which is why the in-tree
docstrings state the consequence of raising there rather than the mechanics:
``BoosterDriver._on_low_state`` says an exception "would kill the subscription
and leave the driver silently blind", and ``G1Driver._on_pressure`` reads its
fields defensively so a renamed one yields ``None`` "rather than raising on the
DDS thread". No caller is on that stack, so nothing can report the failure and
nothing re-subscribes: the cache simply stops advancing while every gate that
consults it keeps answering from the last frame that happened to decode.

Three of the eleven decoders read a member of the sample OUTSIDE their guard,
and each of those reads can fail on its own - the member is produced by a
compiled IDL binding, which is exactly why the guarded coercion beside it
anticipates ``TypeError``:

* ``Go2Driver._on_lowstate`` had no guard at all, and it is the topic behind
  both Go2 write gates (the battery floor and the measured pose ``send_action``
  holds uncommanded joints at).
* ``Go2Driver._on_sportmode`` had no guard, costing the rollout cross-check.
* ``BoosterDriver._on_fall_state`` guarded its second read but not the first,
  and the value it resolves is the fall gate ``send_action`` refuses writes on.

The population is DERIVED from the driver classes rather than listed, so the
eight decoders that already absorbed are passing controls and a decoder added
later is held to the rule without anyone remembering to add a case.
"""

from __future__ import annotations

import asyncio
import logging
import types
from typing import Any

import pytest

from strands_robots.drivers import booster, g1, go2

#: How to build each DDS driver without an SDK, keyed by class. The decoders are
#: read off the class, so the builder only has to produce an instance whose
#: caches are empty.
_BUILDERS: dict[type, Any] = {
    g1.G1Driver: lambda: g1.G1Driver(tool_name="g1", port="1.2.3.4"),
    go2.Go2Driver: lambda: go2.Go2Driver(tool_name="go2", port="192.168.123.161"),
    booster.BoosterDriver: lambda: booster.BoosterDriver(tool_name="t1"),
}

#: Every ``_on_*`` decoder the three DDS drivers register with a subscriber,
#: derived so a new one enrols itself.
_DECODERS: list[tuple[type, str]] = sorted(
    (
        (cls, name)
        for cls in _BUILDERS
        for name, member in vars(cls).items()
        if name.startswith("_on_") and callable(member)
    ),
    key=lambda pair: f"{pair[0].__name__}.{pair[1]}",
)

_IDS = [f"{cls.__name__}.{name}" for cls, name in _DECODERS]


class _Unreadable:
    """A delivered sample whose every member read fails.

    ``TypeError`` because that is what the decoders' own guards already declare
    they can meet: a member produced by a compiled binding is not obliged to be
    the type the coercion beside it expects, and ``list()`` over a member that
    is not a sequence raises exactly this.
    """

    def __getattr__(self, name: str) -> Any:
        raise TypeError(f"decoder: cannot read {name!r} out of this sample")


def test_the_derivation_covers_all_three_dds_drivers() -> None:
    """A scan that found no decoder, or only one driver's, would pass vacuously."""
    assert len(_DECODERS) >= 11, f"only {len(_DECODERS)} decoders derived: {_IDS}"
    assert {"G1Driver", "Go2Driver", "BoosterDriver"} == {cls.__name__ for cls, _ in _DECODERS}


@pytest.mark.parametrize(("cls", "decoder"), _DECODERS, ids=_IDS)
def test_a_sample_it_cannot_read_does_not_reach_the_sdk_thread(
    cls: type, decoder: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The frame is dropped, and dropping it is visible in the log."""
    driver = _BUILDERS[cls]()
    logger_name = type(driver).__module__
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        getattr(driver, decoder)(_Unreadable())
    assert [r for r in caplog.records if r.name == logger_name], (
        f"{cls.__name__}.{decoder} dropped the sample without saying so"
    )


def test_an_unreadable_lowstate_leaves_the_go2_write_gates_on_the_last_reading() -> None:
    """The battery floor and the held pose both read this cache, so it must survive."""
    driver = go2.Go2Driver(tool_name="go2", port="192.168.123.161")
    motors = [types.SimpleNamespace(q=0.1 * slot, dq=0.0, tau_est=0.0) for slot in range(20)]
    driver._on_lowstate(
        types.SimpleNamespace(
            imu_state=types.SimpleNamespace(
                quaternion=[1.0, 0.0, 0.0, 0.0], gyroscope=None, accelerometer=None, rpy=None
            ),
            bms_state=types.SimpleNamespace(soc=88.0, current=1.0, cycle=3),
            motor_state=motors,
        )
    )
    seeded = driver.state
    assert seeded["battery"] == {"pct": 88.0, "current": 1.0, "cycle": 3}
    assert seeded["joints"], "the fixture seeded no joints, so the cell grades nothing"

    driver._on_lowstate(_Unreadable())

    assert driver.state["battery"] == seeded["battery"]
    assert driver.state["joints"] == seeded["joints"]


@pytest.mark.parametrize(
    "unusable",
    [
        pytest.param(_Unreadable(), id="member_raises"),
        pytest.param(types.SimpleNamespace(other_field=1), id="member_absent"),
    ],
)
def test_an_unusable_fall_frame_leaves_the_t1_fall_gate_on_the_last_reading(unusable: Any) -> None:
    """``send_action`` refuses a write unless this cache reads ``IS_READY``.

    Absence and an unreadable member are the same answer here - neither is
    evidence about the robot - so neither may overwrite the last frame that was.
    """
    driver = booster.BoosterDriver(tool_name="t1")
    driver._on_fall_state(types.SimpleNamespace(fall_down_state=types.SimpleNamespace(value=2)))
    assert asyncio.run(driver.get_status())["content"][0]["json"]["fall_state"] == "HAS_FALLEN"

    driver._on_fall_state(unusable)

    assert asyncio.run(driver.get_status())["content"][0]["json"]["fall_state"] == "HAS_FALLEN"
