# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A finished rollout names what ended it, to a poller that arrives late.

Three native drivers roll a policy on their own thread and answer
``get_task_status`` from a snapshot: the UR, the G1 and the Go2. The loop clears
the driver's handle on its way out and stashes its terminal snapshot, so *that
stash* is the only thing a poller arriving after the thread is gone can read -
and ``exit_reason`` in it is the whole answer to "why did my robot stop moving".

The Go2's own ``_run`` docstring lists ``n_steps``, ``duration``, ``gate``,
``policy``, ``publish`` and the caller's ``stop_task`` / ``stop`` / ``cleanup``
as the terminal reasons, and promises the stash carries whichever fired. The
caller's three did not: :meth:`_ControlLoop.stop` recorded its reason *after*
``join()`` returned, and the loop's ``finally`` stashes while that ``join()`` is
still blocked - so the stash predated the reason and a caller-stopped rollout
reported ``exit_reason=None``, collapsing the three caller words into one
absence. The UR is the reference for the fix: its loop names the reason on the
stop path itself (``self._finish("stopped")``), before it can be read.

So this suite grades the reason at the *late* poll, once per way a rollout can
end, on both drivers that stash. No robot, no DDS bus, no ``unitree_sdk2py``:
the SDK stub and the recording publisher are the ones the neighbouring driver
suites already install.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from strands_robots.drivers.g1 import G1Driver
from strands_robots.drivers.go2 import Go2Driver, _ControlLoop
from tests.drivers.test_g1_control_loop_terminal_observability import (
    _fake_driver as _fake_g1_driver,
)
from tests.drivers.test_g1_control_loop_terminal_observability import (
    _install_sdk_stub as _install_g1_sdk_stub,
)
from tests.drivers.test_go2_driver import (
    _RecordingPublisher,
    _released_driver,
    install_unitree_sdk_stub,
)

#: Longer than any cell here needs, so a hung loop fails loudly instead of
#: hanging the suite.
_JOIN_S = 5.0


@pytest.fixture
def stub_unitree_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Go2 SDK stub the neighbouring driver suite installs."""
    install_unitree_sdk_stub(monkeypatch)


def _await_loop_exit(driver: Go2Driver | G1Driver) -> None:
    """Block until the loop has cleared itself off ``driver``.

    That clearing is what makes the poll a *late* one: with ``_loop`` gone,
    ``get_task_status`` can only answer from the stash.

    Args:
        driver: The driver whose rollout is expected to end.

    Raises:
        AssertionError: The loop was still installed after :data:`_JOIN_S`.
    """
    deadline = time.monotonic() + _JOIN_S
    while time.monotonic() < deadline:
        with driver._task_admission:
            if driver._loop is None:
                return
        time.sleep(0.005)
    raise AssertionError(f"the control loop still holds {driver._tool_name} after {_JOIN_S}s")


def _late_poll(driver: Go2Driver | G1Driver) -> dict[str, Any]:
    """The payload a caller sees once the rollout's thread is gone."""
    _await_loop_exit(driver)
    payload = driver.get_task_status()["content"][0]["json"]
    assert payload["running"] is False, payload
    return dict(payload)


def _hold(_state: dict[str, Any]) -> dict[str, float]:
    """A policy that commands one leg joint forever."""
    return {"FL_thigh_joint": 0.0}


# --------------------------------------------------------------------------- #
# The Go2: one row per way a rollout can end.                                 #
# --------------------------------------------------------------------------- #


def _end_on_n_steps(driver: Go2Driver, _pub: _RecordingPublisher) -> None:
    driver.run_policy(_hold, n_steps=2, duration=_JOIN_S)


def _end_on_duration(driver: Go2Driver, _pub: _RecordingPublisher) -> None:
    driver.run_policy(_hold, duration=0.02)


def _end_on_gate(driver: Go2Driver, _pub: _RecordingPublisher) -> None:
    def revoke_the_release(state: dict[str, Any]) -> dict[str, float]:
        # The real gate, not a mock: the onboard controller taking the legs back
        # is what the per-step re-check exists to catch.
        driver._sport_mode_released = False
        return _hold(state)

    driver.run_policy(revoke_the_release, duration=_JOIN_S)


def _end_on_policy_raising(driver: Go2Driver, _pub: _RecordingPublisher) -> None:
    def raises(_state: dict[str, Any]) -> dict[str, float]:
        raise ValueError("inference failed")

    driver.run_policy(raises, duration=_JOIN_S)


def _end_on_unusable_action(driver: Go2Driver, _pub: _RecordingPublisher) -> None:
    driver.run_policy(lambda _state: {"no_such_joint": 0.0}, duration=_JOIN_S)


def _end_on_publish(driver: Go2Driver, pub: _RecordingPublisher) -> None:
    pub.publish_should_return = "no writer matched on rt/lowcmd"
    driver.run_policy(_hold, duration=_JOIN_S)


def _end_on_stop_task(driver: Go2Driver, _pub: _RecordingPublisher) -> None:
    driver.run_policy(_hold, duration=_JOIN_S)
    time.sleep(0.02)
    assert driver.stop_task()["status"] == "success"


def _end_on_the_stop_verb(driver: Go2Driver, _pub: _RecordingPublisher) -> None:
    driver.run_policy(_hold, duration=_JOIN_S)
    time.sleep(0.02)
    asyncio.run(driver.stop())


def _end_on_cleanup(driver: Go2Driver, _pub: _RecordingPublisher) -> None:
    driver.run_policy(_hold, duration=_JOIN_S)
    time.sleep(0.02)
    driver.cleanup()


@pytest.mark.parametrize(
    ("arrange", "exit_reason"),
    [
        (_end_on_n_steps, "n_steps"),
        (_end_on_duration, "duration"),
        (_end_on_gate, "gate"),
        (_end_on_policy_raising, "policy"),
        (_end_on_unusable_action, "policy"),
        (_end_on_publish, "publish"),
        (_end_on_stop_task, "stop_task"),
        (_end_on_the_stop_verb, "stop"),
        (_end_on_cleanup, "cleanup"),
    ],
    ids=[
        "n_steps",
        "duration",
        "gate",
        "policy_raised",
        "unusable_action",
        "publish",
        "stop_task",
        "stop_verb",
        "cleanup",
    ],
)
def test_the_go2_stash_names_whichever_reason_ended_the_rollout(
    stub_unitree_sdk: None,
    arrange: Any,
    exit_reason: str,
) -> None:
    """Every terminal reason the loop documents round-trips to a late poller.

    One table because it is one behaviour, and the three caller rows are the
    ones that reported nothing: ``stop_task``, the mesh's ``stop`` verb and
    ``cleanup`` each write a distinct word, and a caller who cannot read it
    cannot tell an operator's halt from a teardown from a battery gate.
    """
    del stub_unitree_sdk
    driver, pub = _released_driver()
    arrange(driver, pub)

    payload = _late_poll(driver)
    assert payload["exit_reason"] == exit_reason, payload
    # A stash that carries a reason must not also carry the "nothing ran" note.
    assert "reason" not in payload, payload


def test_the_go2_stash_carries_the_gate_refusal_text(stub_unitree_sdk: None) -> None:
    """``exit_detail`` says which gate refused, so the reason is actionable.

    ``exit_reason="gate"`` alone cannot distinguish the sport-mode release from
    the battery floor - the two gates the driver checks per step - and naming
    them is the whole point of refusing separately.
    """
    del stub_unitree_sdk
    driver, _pub = _released_driver()

    def drain_the_battery(state: dict[str, Any]) -> dict[str, float]:
        driver._battery = {"pct": 1.0}
        return _hold(state)

    driver.run_policy(drain_the_battery, duration=_JOIN_S)

    payload = _late_poll(driver)
    assert payload["exit_reason"] == "gate"
    assert "battery 1.0%" in str(payload["exit_detail"])


def test_a_refusal_with_no_text_block_still_names_the_gate(stub_unitree_sdk: None) -> None:
    """A gate refusing without prose degrades to ``"refused"``, not to a crash.

    The refusal envelope is read for logging, so a driver whose gate answers
    with a json-only block must still exit ``gate`` rather than raising inside
    the loop's own error path.
    """
    del stub_unitree_sdk
    driver, _pub = _released_driver()
    driver._check_motion_gates = lambda scope: (  # type: ignore[method-assign]
        None if scope == "run_policy" else {"status": "error", "content": [{"json": {"gate": scope}}]}
    )

    driver.run_policy(_hold, duration=_JOIN_S)

    payload = _late_poll(driver)
    assert payload["exit_reason"] == "gate"
    assert payload["exit_detail"] == "refused"


def test_a_budget_that_already_expired_outranks_the_callers_word() -> None:
    """A loop that ended itself keeps its own, more specific reason.

    Recording the caller's reason before signalling must not overwrite one the
    loop already reached - a rollout that ran its steps out and is then stopped
    by a shutdown is still a rollout that ran its steps out.
    """
    loop = _ControlLoop(driver=None, policy=_hold, duration=1.0, n_steps=1)  # type: ignore[arg-type]
    loop._set_exit("n_steps")

    assert loop.stop("cleanup") is True, "an unstarted loop has no thread to wait for"
    assert loop.snapshot()["exit_reason"] == "n_steps"


def test_a_single_use_loop_refuses_a_second_start() -> None:
    """Restarting a loop would silently reuse the previous rollout's counters."""
    loop = _ControlLoop(driver=None, policy=_hold, duration=1.0, n_steps=1)  # type: ignore[arg-type]
    assert loop.snapshot()["elapsed_s"] is None, "a loop that never started has run for no time"

    loop._thread = threading.Thread(target=lambda: None)  # stands in for a spawned loop
    with pytest.raises(RuntimeError, match="called twice"):
        loop.start()


# --------------------------------------------------------------------------- #
# The G1 runs the same loop shape, so it had the same silence.                 #
# --------------------------------------------------------------------------- #


def test_the_g1_stash_names_the_callers_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stopped G1 rollout reports ``stop_task`` to a late poller too.

    The G1's ``stop_task`` return already named it; the stash a later
    ``get_task_status`` reads did not, so the same rollout answered two
    different things depending on when it was asked.
    """
    _install_g1_sdk_stub(monkeypatch)
    driver = _fake_g1_driver(monkeypatch)

    assert (
        driver.run_policy(policy_object=lambda _obs: {"left_shoulder_pitch": 0.0}, duration=_JOIN_S)["status"]
        == "success"
    )
    time.sleep(0.02)
    stopped = driver.stop_task()
    assert stopped["content"][0]["json"]["exit_reason"] == "stop_task"

    assert _late_poll(driver)["exit_reason"] == "stop_task"
