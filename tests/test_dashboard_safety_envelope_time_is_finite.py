"""A forged ``{"t": NaN}`` e-stop must not poison the fleet view (f032).

``json.loads`` accepts the bare ``NaN`` / ``Infinity`` tokens, so one envelope
published on ``strands/safety/estop`` with ``t: NaN`` reached
``safety_state.apply_event``, which only asked ``isinstance(t, (int, float))``.
The lockout then carried ``since=nan``, every peer's card in ``snapshot()``
carried it too, ``/ws/mesh`` wrote it with ``json.dumps`` as a bare ``NaN``
token, and the browser's ``JSON.parse`` refused the whole snapshot on every
reconnect: the fleet view never loaded again until a finite ``t`` arrived or
the dashboard restarted. The peers themselves refuse such an envelope
(``Mesh._check_safety_envelope_timing`` reads ``as_wire_timestamp``), so the
dashboard was also showing a lockout the fleet never applied.

Now the dashboard reads the same rule: a ``t`` that is present but not a
finite wire timestamp drops the envelope with a log line, and the ``/ws/mesh``
boundary never emits a non-finite float for any field.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any
from unittest import mock

import pytest

from strands_robots.dashboard import safety_state
from strands_robots.dashboard.mesh_bridge import MeshBridge
from strands_robots.dashboard.routes_mesh import wire_json


def _sample(key: str, body: bytes) -> Any:
    sample = mock.MagicMock(spec=["payload", "key_expr"])
    sample.payload.to_bytes.return_value = body
    sample.key_expr = key
    return sample


@pytest.fixture
def bridge() -> MeshBridge:
    b = MeshBridge(peer_id="dash")
    b._running = True
    presence = {"robot_id": "arm-1", "robot_type": "robot", "timestamp": time.time()}
    b._on_presence(_sample("strands/arm-1/presence", json.dumps(presence).encode()))
    return b


class TestApplyEvent:
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_t_drops_the_event(self, bad: float) -> None:
        before = safety_state.Lockout()
        after = safety_state.apply_event(before, kind="estop", data={"source": "evil", "t": bad}, now=100.0)
        assert after == before, "an envelope every peer refuses is not a fleet lockout"

    def test_a_missing_t_still_locks_on_the_dashboard_clock(self) -> None:
        # Unchanged contract: absent is not malformed.
        after = safety_state.apply_event(safety_state.Lockout(), kind="estop", data={"source": "x"}, now=100.0)
        assert after.state == "locked" and after.since == 100.0

    def test_a_boolean_t_is_not_a_timestamp(self) -> None:
        after = safety_state.apply_event(safety_state.Lockout(), kind="estop", data={"t": True}, now=100.0)
        assert after == safety_state.Lockout()

    def test_the_rule_is_the_sdks_own(self) -> None:
        from strands_robots.mesh import security

        for value in (math.nan, math.inf, True, "1.0"):
            assert security.as_wire_timestamp(value) is None
            assert safety_state.envelope_refusal({"t": value}) is not None, value
        assert safety_state.envelope_refusal({"t": 1.0}) is None
        assert safety_state.envelope_refusal({}) is None


class TestTheBridge:
    def test_a_nan_estop_is_dropped_and_logged(self, bridge: MeshBridge, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            bridge._on_safety(_sample("strands/safety/estop", b'{"source":"evil","t":NaN}'))
        assert bridge._lockout.state == "unknown"
        assert any("estop" in r.getMessage() and "t" in r.getMessage() for r in caplog.records), caplog.text
        assert bridge.activity[-1]["action"] == "estop_refused" and bridge.activity[-1]["ok"] is False

    def test_the_snapshot_after_a_nan_estop_is_valid_json(self, bridge: MeshBridge) -> None:
        bridge._on_safety(_sample("strands/safety/estop", b'{"source":"evil","t":NaN}'))
        text = wire_json(bridge.snapshot())
        assert "NaN" not in text
        doc = json.loads(text, parse_constant=_refuse_constant)
        assert doc["peers"]["arm-1"]["lockout"]["state"] == "unknown"

    def test_a_finite_estop_still_locks(self, bridge: MeshBridge) -> None:
        bridge._on_safety(_sample("strands/safety/estop", json.dumps({"source": "op", "t": time.time()}).encode()))
        assert bridge._lockout.state == "locked"
        assert wire_json(bridge.snapshot())


def _refuse_constant(name: str) -> Any:
    raise AssertionError(f"a browser's JSON.parse refuses the token {name}")


class TestTheWireBoundary:
    def test_a_non_finite_float_anywhere_becomes_null_not_a_bare_token(self, caplog: pytest.LogCaptureFixture) -> None:
        doc = {"peers": {"p": {"joints": [0.1, math.nan, math.inf], "t": -math.inf}}, "t": 5.0}
        with caplog.at_level(logging.WARNING):
            text = wire_json(doc)
        parsed = json.loads(text, parse_constant=_refuse_constant)
        assert parsed == {"peers": {"p": {"joints": [0.1, None, None], "t": None}}, "t": 5.0}
        assert any("non-finite" in r.getMessage() for r in caplog.records)

    def test_a_clean_document_is_unchanged(self) -> None:
        doc = {"a": [1, 2.5, "x", None, True], "b": {"c": 0.0}}
        assert json.loads(wire_json(doc)) == doc
