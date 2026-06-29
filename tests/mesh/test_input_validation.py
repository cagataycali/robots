"""Tests for teleop input-frame validation.

InputReceiver._on_input must validate frames via
security.validate_input_frame before applying them to the robot, so a
LAN-adjacent peer cannot drive joints with unbounded / non-finite /
malformed values.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from strands_robots.mesh import security
from strands_robots.mesh.input import InputReceiver

# --- validate_input_frame unit tests -------------------------------------


def test_valid_frame_passes_through():
    frame = {"motor.pos": 0.5, "shoulder_pan": -1.25, "j0": 10}
    out = security.validate_input_frame(frame)
    assert out == {"motor.pos": 0.5, "shoulder_pan": -1.25, "j0": 10.0}
    assert all(isinstance(v, float) for v in out.values())


def test_non_dict_rejected():
    with pytest.raises(security.ValidationError):
        security.validate_input_frame([1, 2, 3])


def test_too_many_keys_rejected():
    frame = {f"j{i}": 0.0 for i in range(security.MAX_INPUT_FRAME_KEYS + 1)}
    with pytest.raises(security.ValidationError):
        security.validate_input_frame(frame)


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_non_finite_rejected(bad):
    with pytest.raises(security.ValidationError):
        security.validate_input_frame({"j0": bad})


def test_out_of_range_rejected():
    with pytest.raises(security.ValidationError):
        security.validate_input_frame({"j0": security.MAX_INPUT_VALUE_ABS * 2})


def test_bad_key_charset_rejected():
    with pytest.raises(security.ValidationError):
        security.validate_input_frame({"../etc/passwd": 0.0})


def test_bool_value_rejected():
    with pytest.raises(security.ValidationError):
        security.validate_input_frame({"j0": True})


def test_non_numeric_value_rejected():
    with pytest.raises(security.ValidationError):
        security.validate_input_frame({"j0": "0.5"})


@pytest.mark.parametrize("np_bool", [np.bool_(True), np.bool_(False)])
def test_numpy_bool_value_rejected(np_bool):
    """A numpy boolean must be rejected just like a python bool.

    ``np.bool_`` is not a python ``bool``, so a bool gate placed before the
    ``.item()`` numpy-scalar coercion lets it through; once coerced it becomes
    a python ``bool`` that passes the ``isinstance(value, (int, float))`` check
    (``bool`` subclasses ``int``) and silently maps to a 1.0 / 0.0 actuator
    command. A LAN-adjacent peer streaming numpy-typed frames could otherwise
    drive a joint with ``np.True_`` despite the explicit bool defense.
    """
    with pytest.raises(security.ValidationError):
        security.validate_input_frame({"j0": np_bool})


def test_numpy_scalar_value_accepted_and_coerced():
    """Legitimate numpy scalars (the common case for arm reads) coerce to
    plain python floats and pass the validator."""
    out = security.validate_input_frame({"j0": np.float32(0.5), "j1": np.int64(3)})
    assert out == {"j0": 0.5, "j1": 3.0}
    assert all(type(v) is float for v in out.values())


def test_numpy_non_finite_value_rejected():
    """A numpy non-finite scalar is unwrapped then caught by the finite check."""
    with pytest.raises(security.ValidationError):
        security.validate_input_frame({"j0": np.float64("nan")})


def test_non_string_key_rejected():
    with pytest.raises(security.ValidationError):
        security.validate_input_frame({123: 0.5})


def test_empty_key_rejected():
    with pytest.raises(security.ValidationError):
        security.validate_input_frame({"": 0.5})


def test_overlong_key_rejected():
    with pytest.raises(security.ValidationError):
        security.validate_input_frame({"x" * (security.MAX_INPUT_KEY_LEN + 1): 0.5})


# --- InputReceiver wiring tests ------------------------------------------


class _FakeMesh:
    peer_id = "follower-1"

    def subscribe(self, *a, **k):
        return "sub"

    def unsubscribe(self, *a, **k):
        pass


def _make_receiver():
    applied: list[dict] = []
    recv = InputReceiver(
        mesh=_FakeMesh(),
        robot=object(),
        source_peer_id="leader-1",
        apply_fn=lambda robot, action: applied.append(action),
    )
    recv._running = True
    return recv, applied


def test_on_input_applies_valid_frame():
    recv, applied = _make_receiver()
    recv._on_input(recv.topic, {"action": {"j0": 0.1, "j1": 0.2}, "seq": 0, "t": time.time()})
    assert applied == [{"j0": 0.1, "j1": 0.2}]
    assert recv._frame_count == 1
    assert recv._rejected == 0


def test_on_input_rejects_malicious_frame():
    recv, applied = _make_receiver()
    # non-finite value would otherwise reach send_action()
    recv._on_input(recv.topic, {"action": {"j0": math.inf}, "seq": 0, "t": time.time()})
    assert applied == []  # never applied
    assert recv._frame_count == 0
    assert recv._rejected == 1


def test_on_input_rejects_giant_frame():
    recv, applied = _make_receiver()
    giant = {f"j{i}": 0.0 for i in range(security.MAX_INPUT_FRAME_KEYS + 5)}
    recv._on_input(recv.topic, {"action": giant, "seq": 0, "t": time.time()})
    assert applied == []
    assert recv._rejected == 1


# --- cross-session teleop replay / freshness -----------------------------


def test_on_input_rejects_stale_frame():
    """A frame with a timestamp older than the freshness window (default 60s)
    is a cross-session replay and must be dropped before reaching the robot."""
    recv, applied = _make_receiver()
    stale_t = time.time() - 3600.0  # 1 hour old
    recv._on_input(recv.topic, {"action": {"j0": 0.1}, "seq": 0, "t": stale_t})
    assert applied == []
    assert recv._frame_count == 0
    assert recv._rejected == 1


def test_on_input_rejects_future_frame():
    """A frame skewed far into the future (beyond forward-skew tolerance) is
    rejected -- protects against clock-spoofed envelopes."""
    recv, applied = _make_receiver()
    future_t = time.time() + 3600.0  # 1 hour ahead
    recv._on_input(recv.topic, {"action": {"j0": 0.1}, "seq": 0, "t": future_t})
    assert applied == []
    assert recv._frame_count == 0
    assert recv._rejected == 1


def test_on_input_rejects_missing_timestamp():
    """The publisher always sets ``t``; a frame without it is malformed or a
    hand-crafted replay envelope and must be rejected (matches presence M-3)."""
    recv, applied = _make_receiver()
    recv._on_input(recv.topic, {"action": {"j0": 0.1}, "seq": 0})
    assert applied == []
    assert recv._frame_count == 0
    assert recv._rejected == 1


@pytest.mark.parametrize("bad_t", [True, False, "now", None, [1], {"t": 1}])
def test_on_input_rejects_non_numeric_timestamp(bad_t):
    """Non-numeric (incl. bool) ``t`` values are treated as missing -> rejected."""
    recv, applied = _make_receiver()
    recv._on_input(recv.topic, {"action": {"j0": 0.1}, "seq": 0, "t": bad_t})
    assert applied == []
    assert recv._rejected == 1


def test_on_input_accepts_fresh_frame_within_skew():
    """A frame slightly in the future but within the forward-skew tolerance
    (default 5s) is accepted -- legitimate clock drift must not be rejected."""
    recv, applied = _make_receiver()
    recv._on_input(recv.topic, {"action": {"j0": 0.1}, "seq": 0, "t": time.time() + 1.0})
    assert applied == [{"j0": 0.1}]
    assert recv._frame_count == 1
    assert recv._rejected == 0
