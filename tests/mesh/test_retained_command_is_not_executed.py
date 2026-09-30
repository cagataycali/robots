"""A command the broker stored (RETAIN) is refused, not executed, and no policy lets an operator store one.

Reproduced live 2026-09-30 (account 947951559549, Thing ac-arm-01): one
``aws iot-data publish --retain`` of ``{"command": {"action": "execute",
"policy_provider": "mock", ...}}`` on ``strands/ac-arm-01/cmd`` made the robot
run that rollout at its next start with nobody present (audit
``command_executed``, sim joints moving at t=15 s), and a retained
``set_joints`` on the child topic was dispatched on 3/3 starts. MQTT delivers
a retained message to every new subscription, ``Mesh.start()`` subscribes
``cmd`` at every boot, and the replay cache is per process, so a stored
command is a motion-on-boot implant until someone clears the topic.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.mesh import Mesh
from strands_robots.mesh.iot import provision
from strands_robots.mesh.transport.iot_transport import _MqttSample


def _cmd_sample(topic: str, *, retain: bool) -> _MqttSample:
    envelope = {"sender_id": "ac-ops-01", "turn_id": "c08-retained", "command": {"action": "status"}}
    return _MqttSample(topic, json.dumps(envelope).encode(), retain=retain)


@pytest.fixture
def mesh(monkeypatch: pytest.MonkeyPatch) -> tuple[Mesh, list[tuple[str, dict[str, Any]]], list[dict[str, Any]]]:
    m = Mesh(SimpleNamespace(tool_name_str="arm"), peer_id="ac-arm-01")
    audits: list[tuple[str, dict[str, Any]]] = []
    executed: list[dict[str, Any]] = []
    monkeypatch.setattr(m, "_audit_local", lambda e, p: audits.append((e, p)))
    monkeypatch.setattr(m, "_exec_cmd", lambda data, **kw: executed.append(data))
    return m, audits, executed


class TestARetainedCommandIsRefused:
    @pytest.mark.parametrize("topic", ["strands/ac-arm-01/cmd", "strands/broadcast"])
    def test_a_stored_command_is_not_dispatched_and_the_audit_says_why(self, mesh, topic, caplog) -> None:
        m, audits, executed = mesh
        with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
            m._on_cmd(_cmd_sample(topic, retain=True))
        time.sleep(0.05)
        assert executed == []
        (event, payload) = audits[-1]
        assert event == "command_refused"
        assert (
            payload["reason"] == "retained"
            and payload["sender"] == "ac-ops-01"
            and payload["turn_id"] == "c08-retained"
        )
        assert "retained" in caplog.text and topic in caplog.text and "aws iot-data publish" in caplog.text

    def test_a_live_command_still_dispatches(self, mesh) -> None:
        m, _audits, executed = mesh
        m._on_cmd(_cmd_sample("strands/ac-arm-01/cmd", retain=False))
        assert len(executed) == 1

    def test_the_warning_is_once_per_topic(self, mesh, caplog) -> None:
        m, audits, _executed = mesh
        with caplog.at_level("WARNING", logger="strands_robots.mesh.core"):
            m._on_cmd(_cmd_sample("strands/ac-arm-01/cmd", retain=True))
            m._on_cmd(_cmd_sample("strands/ac-arm-01/cmd", retain=True))
        assert sum("retained" in r.getMessage() for r in caplog.records) == 1
        assert len([a for a in audits if a[0] == "command_refused"]) == 2, "every refusal is audited"


class TestNoPolicyLetsAnOperatorStoreACommand:
    def test_the_operator_fleet_publish_is_publish_only(self) -> None:
        st = next(s for s in provision._OPERATOR_POLICY_DOC["Statement"] if s["Sid"] == "OperatorPublishToFleet")
        assert "iot:RetainPublish" not in st["Action"], st
        assert any(r.endswith("/cmd") for r in st["Resource"]) and any(r.endswith("/broadcast") for r in st["Resource"])

    def test_the_operator_still_retains_its_own_presence(self) -> None:
        st = next(s for s in provision._OPERATOR_POLICY_DOC["Statement"] if s["Sid"] == "OperatorAnnounceSelf")
        assert "iot:RetainPublish" in st["Action"]
