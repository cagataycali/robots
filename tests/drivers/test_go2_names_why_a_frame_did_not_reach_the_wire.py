"""A Go2 frame that never reached the wire is named by the path that owns it.

The Go2's low-level write path is "build, then publish", and both halves can
fail without a robot being present: the SDK that builds and seals a ``LowCmd_``
is imported lazily, and the DDS publisher is a resource
:meth:`~strands_robots.drivers.go2.Go2Driver.cleanup` and a failed
``connect_eagerly`` both release. What makes the failures worth pinning together
is that the driver answers each one in the vocabulary of the surface that
noticed:

* Both write doors - :meth:`~strands_robots.drivers.go2.Go2Driver.send_action`
  and :meth:`~strands_robots.drivers.go2.Go2Driver.run_policy` - refuse a driver
  with no publisher in an error envelope, before either builds anything, and
  ``run_policy`` admits no rollout on the way out.
* A publisher that reports a reason has that reason forwarded verbatim; the
  driver does not restate it.
* A frame the SDK cannot seal is refused rather than published unsealed.
  Firmware verifies the CRC and drops a frame whose stamp does not match, so an
  unsealed publish is the worst shape available: a success envelope for a frame
  that commanded nothing.
* Inside a rollout the publisher can disappear between frames, and the loop says
  so twice - once as the terminal reason for the frame that did not go out, and
  once for the soft stop that then had no wire to reach. A quadruped standing on
  twelve position-controlled joints is what that zero-torque frame exists for, so
  its absence is logged rather than passed over.

No cell needs a Go2, a DDS bus or ``unitree_sdk2py``. The recording publisher the
Go2 write-path suite already drives stands in for the bus, and an absent SDK
submodule is modelled the way the interpreter models one - a ``None`` entry in
:mod:`sys.modules`, which makes ``import`` raise ``ImportError`` - so the lazy
import under test is the production one hardware runs.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Any

import pytest

from strands_robots.drivers.go2 import Go2Driver, build_zero_torque_lowcmd
from tests.drivers.test_go2_driver import (
    _released_driver,
    _text,
    install_unitree_sdk_stub,
)

#: The driver's logger, whose records carry what the loop could not publish.
_DRIVER_LOGGER = "strands_robots.drivers.go2"

#: One real Go2 joint name, enough for a well-formed single-joint action.
_A_JOINT = "FL_hip_joint"

#: The refusal both write doors give a driver that holds no publisher.
_NO_PUBLISHER = "publisher not initialised - call connect_eagerly() first"


def _break_sdk_submodule(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Install the SDK stub, then make importing one of its submodules fail.

    ``None`` in :mod:`sys.modules` is how the interpreter itself reports a module
    that cannot be imported, so the driver's lazy ``import`` takes its real
    ``ImportError`` branch. The rest of the stub stays in place, which is what
    isolates the failure to the one stage under test.

    Args:
        monkeypatch: The requesting test's patcher, which owns the teardown.
        name: Dotted name of the submodule to make unimportable.
    """
    install_unitree_sdk_stub(monkeypatch)
    monkeypatch.setitem(sys.modules, name, None)


@pytest.mark.parametrize(
    ("door", "write"),
    [
        pytest.param("send_action", lambda d: d.send_action({_A_JOINT: 0.0}), id="send_action"),
        pytest.param(
            "run_policy",
            lambda d: d.run_policy(lambda _state: {_A_JOINT: 0.0}, n_steps=1, duration=5.0),
            id="run_policy",
        ),
    ],
)
def test_both_write_doors_refuse_a_driver_that_holds_no_publisher(
    monkeypatch: pytest.MonkeyPatch, door: str, write: Callable[[Go2Driver], dict[str, Any]]
) -> None:
    """Neither door builds a frame it has nowhere to send, and neither raises.

    ``_pubs`` is ``None`` on a driver that never connected and on one whose
    ``cleanup`` released it, so this is the ordinary out-of-order call rather than
    a corrupted state. ``run_policy`` additionally must not leave a rollout
    behind: a loop admitted here would publish nothing for its whole budget while
    ``get_task_status`` reported it running.
    """
    del door
    install_unitree_sdk_stub(monkeypatch)
    driver, _pub = _released_driver()
    driver._pubs = None

    result = write(driver)

    assert result["status"] == "error"
    assert _text(result) == _NO_PUBLISHER
    assert driver._loop is None
    assert driver.get_task_status()["content"][0]["json"]["running"] is False


def test_a_publisher_that_reports_a_reason_has_it_forwarded_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """The publisher's own reason is the refusal, not a restatement of it.

    ``DDSPublisher.publish`` returns ``None`` or a reason, and the reasons it
    gives (an unmatched writer, a closed publisher) are the ones a caller needs to
    read. Wrapping them in a driver-level summary would drop the only text that
    says which of them happened.
    """
    install_unitree_sdk_stub(monkeypatch)
    driver, pub = _released_driver()
    pub.publish_should_return = "no writer matched on rt/lowcmd"

    result = driver.send_action({_A_JOINT: 0.25})

    assert result["status"] == "error"
    assert _text(result) == "no writer matched on rt/lowcmd"
    assert pub.writes == []


def test_a_frame_the_sdk_cannot_seal_is_refused_rather_than_published_unsealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sealer that cannot stamp the CRC refuses the write; nothing reaches the wire.

    The CRC is the last write before a frame is published and firmware drops a
    frame whose stamp does not match, silently. Publishing the unsealed frame
    anyway would therefore answer ``success`` for a command the robot discards -
    strictly worse than a refusal naming the missing SDK.
    """
    _break_sdk_submodule(monkeypatch, "unitree_sdk2py.utils.crc")
    driver, pub = _released_driver()

    result = driver.send_action({_A_JOINT: 0.25})

    assert result["status"] == "error"
    assert "unitree_sdk2py is not installed" in _text(result)
    assert pub.writes == []


@pytest.mark.parametrize(
    ("broken", "stage"),
    [
        pytest.param("unitree_sdk2py.idl.default", "the constructor", id="no-idl"),
        pytest.param("unitree_sdk2py.utils.crc", "the sealer", id="no-crc"),
    ],
)
def test_the_soft_stop_builder_refuses_a_frame_it_could_not_finish(
    monkeypatch: pytest.MonkeyPatch, broken: str, stage: str
) -> None:
    """``build_zero_torque_lowcmd`` reports either build stage, and returns no frame.

    The two stages fail independently - the IDL constructor makes the frame, the
    sealer stamps it - and the contract is the same for both: ``(None, reason)``,
    never a half-built frame a caller could mistake for the soft stop.
    """
    del stage
    _break_sdk_submodule(monkeypatch, broken)

    cmd, reason = build_zero_torque_lowcmd()

    assert cmd is None
    assert reason is not None
    assert broken in reason


def test_releasing_the_publisher_under_a_live_rollout_is_named_twice(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A rollout that loses its publisher names the lost frame and the lost soft stop.

    ``_publish`` re-reads the driver's publisher every frame rather than caching
    it, so a concurrent teardown is noticed at the next write instead of raising
    on a stale handle. The loop ends on that reason, and its ``finally`` then
    finds the same absent publisher - so the zero-torque frame it always sends on
    the way out is reported missing rather than assumed sent.
    """
    install_unitree_sdk_stub(monkeypatch)
    driver, pub = _released_driver()

    def policy(_state: dict[str, Any]) -> dict[str, Any]:
        driver._pubs = None  # the teardown a caller can run at any time
        return {_A_JOINT: 0.1}

    with caplog.at_level(logging.WARNING, logger=_DRIVER_LOGGER):
        started = driver.run_policy(policy, n_steps=3, duration=5.0)
        assert started["status"] == "success", _text(started)
        assert driver._loop is not None
        driver._loop._thread.join(timeout=5.0)  # type: ignore[union-attr]

    snapshot = driver.get_task_status()["content"][0]["json"]
    assert snapshot["running"] is False
    assert snapshot["exit_reason"] == "publish"
    assert snapshot["exit_detail"] == "publisher was released while the loop was running"
    assert snapshot["steps"] == 0
    assert pub.writes == []
    assert "no publisher at shutdown; no zero-torque frame sent" in caplog.text


def test_a_soft_stop_frame_that_cannot_be_built_is_named_not_silently_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The loop logs the reason its zero-torque frame was never built.

    The soft stop runs in a ``finally``, where there is no envelope left to refuse
    into - so the reason is logged at ERROR, carrying the builder's own text. The
    same missing sealer ends the rollout itself, and the two failures are reported
    separately: the commanded frame as the loop's terminal reason, the soft stop in
    the record here.
    """
    _break_sdk_submodule(monkeypatch, "unitree_sdk2py.utils.crc")
    driver, pub = _released_driver()

    with caplog.at_level(logging.ERROR, logger=_DRIVER_LOGGER):
        started = driver.run_policy(lambda _state: {_A_JOINT: 0.1}, n_steps=1, duration=5.0)
        assert started["status"] == "success", _text(started)
        assert driver._loop is not None
        driver._loop._thread.join(timeout=5.0)  # type: ignore[union-attr]

    snapshot = driver.get_task_status()["content"][0]["json"]
    assert snapshot["exit_reason"] == "policy"
    assert "unitree_sdk2py is not installed" in str(snapshot["exit_detail"])
    assert pub.writes == []

    soft_stop_records = [r for r in caplog.records if "zero-torque frame" in r.getMessage()]
    assert len(soft_stop_records) == 1
    args = soft_stop_records[0].args
    assert isinstance(args, tuple)
    assert "unitree_sdk2py is not installed" in str(args[0])
