"""A robot whose clock lags the operator still latches the e-stop.

Reproduced 2026-09-30 (IoT actor/critic lane, C11): with the robot's wall
clock 30 s behind, ``strands/safety/estop`` from an operator on true time was
refused as "``t`` in future (forward_skew_s=5.0)", while the same operator's
``set_joints`` (cmd envelopes carry no ``t``) was dispatched. Driveable and
unstoppable, with nothing on the operator's side to say so. An early stop is
not a replay - a replay is by definition old - so the forward-skew rule keeps
guarding ``resume`` (where an early envelope could pre-arm a replay) and the
freshness window still bounds ``estop``, but an estop from the future within
that window engages the lockout and audits the skew.
"""

from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.mesh import Mesh
from strands_robots.mesh.transport.iot_transport import _MqttSample


@pytest.fixture
def mesh(monkeypatch: pytest.MonkeyPatch):
    m = Mesh(SimpleNamespace(tool_name_str="arm"), peer_id="ac-arm-01")
    audits: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(m, "_audit", lambda **kw: audits.append((kw.get("event_type"), kw)))
    monkeypatch.setattr(m, "_audit_local", lambda e, p: audits.append((e, p)))
    return m, audits


def _estop(t: float) -> _MqttSample:
    env = {"peer_id": "ac-ops-01", "t": t, "responses_received": 0, "peers_not_stopped": []}
    return _MqttSample("strands/safety/estop", json.dumps(env).encode())


def _resume(t: float) -> _MqttSample:
    env = {"peer_id": "ac-ops-01", "t": t, "override_code": "x" * 32}
    return _MqttSample("strands/safety/resume", json.dumps(env).encode())


class TestAnEarlyEstopStillStops:
    def test_an_estop_30s_in_the_future_engages_the_lockout_and_audits_the_skew(self, mesh, caplog):
        m, audits = mesh
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
            m._on_safety_estop(_estop(time.time() + 30.0))
        assert m._estop_lockout.is_set(), "a stop that is early is still a stop"
        events = [e for e, _ in audits]
        assert "estop_clock_skew" in events
        assert "clock" in caplog.text and "30" in caplog.text and "ac-ops-01" in caplog.text

    def test_an_estop_beyond_the_freshness_window_ahead_is_still_refused(self, mesh):
        m, _audits = mesh
        m._on_safety_estop(_estop(time.time() + 3600.0))
        assert not m._estop_lockout.is_set()

    def test_an_early_estop_is_cached_so_it_does_not_replay_forever(self, mesh):
        m, _audits = mesh
        t = time.time() + 30.0
        m._on_safety_estop(_estop(t))
        assert m._estop_lockout.is_set()
        assert float(t) in m._estop_replay_cache, "the early estop occupies a replay slot keyed by its t"

    def test_a_resume_from_the_future_stays_refused(self, mesh, monkeypatch, caplog):
        m, _audits = mesh
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "x" * 32)
        m._on_safety_estop(_estop(time.time()))
        assert m._estop_lockout.is_set()
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
            m._on_safety_resume(_resume(time.time() + 30.0))
        assert m._estop_lockout.is_set(), "an early resume could pre-arm a replay; it is refused"
        assert "refusing remote resume" in caplog.text
