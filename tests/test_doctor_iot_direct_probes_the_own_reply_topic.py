#!/usr/bin/env python3
"""``check_iot_direct``: the doctor row for AWS IoT Core Direct Messaging.

No AWS, no awsiot. The transport class is replaced by a stand-in whose
``connect`` and ``send_direct`` answer a script, so each test pins one verdict:

  - SKIP for any backend but ``iot`` / ``bridge``, and for ``STRANDS_MESH_IOT_DIRECT=0``
    (a deliberate posture, not a fault);
  - FAIL naming the missing variables before any request is made; the MQTT
    SDK is not needed for the HTTPS probe;
  - the probe is HTTPS only (never an MQTT connect, which would take over the
    running robot's session) and addresses the peer's OWN reply topic
    ``strands/{thing}/response/{thing}/<turn>`` with confirmation: the one
    topic a robot identity may send itself a direct message on;
  - 200 and 404 are both PASS (grant in place; the robot is, or is not,
    connected right now), a 403 names the certificate CN when it can be read
    and it is not the Thing name, unavailable is FAIL, anything else WARN;
  - the transport is closed on every path.
"""

from __future__ import annotations

import re
import sys
import types
from typing import Any

import pytest

from strands_robots import doctor
from strands_robots.mesh.transport.base import DirectResult

_EP = "x-ats.iot.us-west-2.amazonaws.com"


class _Transport:
    instances: list[_Transport] = []

    connect_ok = True
    result = DirectResult(delivered=True, reason="", latency_ms=123.4)

    def __init__(self, connect_timeout: float = 15.0) -> None:
        import os
        from pathlib import Path

        self.thing_name = os.environ.get("STRANDS_IOT_THING_NAME", "")
        self._endpoint = os.environ.get("STRANDS_IOT_ENDPOINT", "")
        self._cert_dir = Path(os.environ.get("STRANDS_IOT_CERT_DIR", "/nonexistent"))
        self.connect_timeout = connect_timeout
        self.connected = False
        self.closed = False
        self.sent: list[dict[str, Any]] = []
        _Transport.instances.append(self)

    def connect(self) -> bool:
        raise AssertionError("the doctor must not open an MQTT session under the robot's client id")

    result_for_cmd: DirectResult | None = None

    def send_direct(self, peer_id: str, key: str, data: dict[str, Any], **kw: Any) -> DirectResult:
        self.sent.append({"peer_id": peer_id, "key": key, "data": data, **kw})
        if key.endswith("/cmd") and self.result_for_cmd is not None:
            return self.result_for_cmd
        return self.result

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def iot_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(doctor, "_NO_COLOR", True)
    monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")
    monkeypatch.setenv("STRANDS_IOT_THING_NAME", "thor-arm")
    monkeypatch.setenv("STRANDS_IOT_ENDPOINT", _EP)
    monkeypatch.delenv("STRANDS_MESH_IOT_DIRECT", raising=False)
    monkeypatch.delenv("STRANDS_IOT_DIRECT_AUTH", raising=False)
    monkeypatch.setitem(sys.modules, "awsiot", types.ModuleType("awsiot"))
    import strands_robots.mesh.transport.iot_transport as iot_mod

    monkeypatch.setattr(iot_mod, "IotMqttTransport", _Transport)
    _Transport.instances.clear()
    _Transport.connect_ok = True
    _Transport.result = DirectResult(delivered=True, reason="", latency_ms=123.4)
    _Transport.result_for_cmd = None
    yield


class TestSkips:
    def test_zenoh_backend_is_a_skip(self, monkeypatch):
        monkeypatch.setattr(doctor, "_NO_COLOR", True)
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "zenoh")
        line = doctor.check_iot_direct()
        assert line.startswith("  SKIP  ")
        assert "STRANDS_MESH_BACKEND=zenoh" in line

    def test_switch_off_is_a_skip_not_a_failure(self, iot_env, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_IOT_DIRECT", "0")
        line = doctor.check_iot_direct()
        assert line.startswith("  SKIP  ")
        assert "STRANDS_MESH_IOT_DIRECT=0" in line
        assert _Transport.instances == []

    def test_bridge_backend_is_probed(self, iot_env, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "bridge")
        assert doctor.check_iot_direct().startswith("  PASS  ")


class TestFailuresBeforeTheWire:
    def test_missing_thing_or_endpoint_fails_without_connecting(self, iot_env, monkeypatch):
        monkeypatch.delenv("STRANDS_IOT_THING_NAME")
        line = doctor.check_iot_direct()
        assert line.startswith("  FAIL  ")
        assert "STRANDS_IOT_THING_NAME" in line and "provision_robot" in line
        assert all(not t.connected for t in _Transport.instances)

    def test_the_probe_needs_no_awsiot_sdk(self, iot_env, monkeypatch):
        # HTTPS only: the MQTT SDK is not consulted, so a box with boto3 and a
        # certificate but no awsiotsdk still gets a verdict.
        monkeypatch.delitem(sys.modules, "awsiot")
        assert doctor.check_iot_direct().startswith("  PASS  ")

    def test_no_mqtt_session_is_opened(self, iot_env):
        # connect() raises in the stand-in; a PASS proves it was never called.
        assert doctor.check_iot_direct().startswith("  PASS  ")
        assert _Transport.instances[0].closed


class TestTheProbe:
    def test_addresses_the_own_reply_topic_with_confirmation(self, iot_env):
        line = doctor.check_iot_direct()
        (t,) = _Transport.instances
        (call,) = t.sent
        assert call["peer_id"] == "thor-arm"
        assert re.fullmatch(r"strands/thor-arm/response/thor-arm/[0-9a-f]{32}", call["key"])
        assert call["confirm"] is True
        assert call["data"]["responder_id"] == "thor-arm"
        assert call["data"]["turn_id"] == call["key"].rsplit("/", 1)[1]
        assert t.closed
        assert line == "  PASS  iot direct: robot grant OK, thor-arm is connected and answered in 123 ms (x509)"

    def test_pass_names_the_auth_mode(self, iot_env, monkeypatch):
        monkeypatch.setenv("STRANDS_IOT_DIRECT_AUTH", "sigv4")
        assert doctor.check_iot_direct().endswith("(sigv4)")

    def test_a_404_is_a_pass_the_grant_is_in_place_and_the_robot_is_simply_not_running_here(self, iot_env):
        _Transport.result = DirectResult(delivered=False, reason="offline", latency_ms=81.0, detail="not connected")
        line = doctor.check_iot_direct()
        assert line.startswith("  PASS  ")
        assert "grant OK" in line and "not running here" in line and "81 ms" in line

    def test_forbidden_with_a_foreign_cn_names_it(self, iot_env, monkeypatch, tmp_path):
        _Transport.result = DirectResult(
            delivered=False, reason="forbidden", latency_ms=80.0, detail="Authorization failed"
        )
        monkeypatch.setattr(doctor, "_certificate_cn", lambda path: "AWS IoT Certificate")
        line = doctor.check_iot_direct()
        assert line.startswith("  FAIL  ")
        assert "certificate CN is 'AWS IoT Certificate'" in line
        assert "CN='thor-arm'" in line and "provision_robot('thor-arm')" in line

    def test_forbidden_on_both_grants_with_the_right_cn_points_at_the_policy_version(self, iot_env, monkeypatch):
        _Transport.result = DirectResult(
            delivered=False, reason="forbidden", latency_ms=80.0, detail="Authorization failed"
        )
        monkeypatch.setattr(doctor, "_certificate_cn", lambda path: "thor-arm")
        line = doctor.check_iot_direct()
        assert "neither as a robot" in line and "nor as an operator" in line
        assert "predates the direct messaging grants" in line
        (t,) = _Transport.instances
        keys = [c["key"] for c in t.sent]
        assert keys[1] == "strands/thor-arm/cmd" and keys[0].startswith("strands/thor-arm/response/thor-arm/")

    def test_an_operator_identity_passes_on_the_cmd_grant(self, iot_env, monkeypatch):
        # A robot-grant 403 followed by a cmd-grant 404: the identity is an
        # operator whose grant is in place; not a failure.
        _Transport.result = DirectResult(delivered=False, reason="forbidden", latency_ms=80.0)
        _Transport.result_for_cmd = DirectResult(delivered=False, reason="offline", latency_ms=90.0)
        monkeypatch.setattr(doctor, "_certificate_cn", lambda path: "thor-arm")
        line = doctor.check_iot_direct()
        assert line.startswith("  PASS  ") and "operator grant OK" in line and "90 ms" in line

    def test_a_403_that_names_the_target_client_is_the_target_side_verdict_operator(self, iot_env, monkeypatch):
        # Measured on the account (iot-deep lane, D5): with the operator's own
        # Mesh running, the cmd probe to its own client id returns 403 with the
        # body "Authorization failed for the target client". The broker checked
        # the SENDER grant first and passed it; the TARGET (the operator itself)
        # simply holds no iot:Receive on a cmd topic, because an operator does
        # not take commands. That is the grant in place, not a policy problem,
        # and the earlier advice to re-provision changed nothing.
        _Transport.result = DirectResult(delivered=False, reason="forbidden", latency_ms=80.0)
        _Transport.result_for_cmd = DirectResult(
            delivered=False, reason="forbidden", latency_ms=95.0, detail="Authorization failed for the target client"
        )
        monkeypatch.setattr(doctor, "_certificate_cn", lambda path: "thor-op")
        line = doctor.check_iot_direct()
        assert line.startswith("  PASS  "), line
        assert "operator grant OK" in line and "95 ms" in line
        assert "target" in line and "iot:Receive" in line
        assert "re-run provision" not in line

    def test_a_403_that_names_the_target_client_on_the_robot_probe_passes_the_robot_grant(self, iot_env, monkeypatch):
        # The same verdict on the first probe: the robot's sender grant held and
        # the running peer under that id holds no iot:Receive on the reply
        # topic. Role stays robot; no second probe is needed.
        _Transport.result = DirectResult(
            delivered=False, reason="forbidden", latency_ms=70.0, detail="Authorization failed for the target client"
        )
        monkeypatch.setattr(doctor, "_certificate_cn", lambda path: "thor-arm")
        line = doctor.check_iot_direct()
        assert line.startswith("  PASS  ") and "robot grant OK" in line, line
        (t,) = _Transport.instances
        assert len(t.sent) == 1

    def test_a_403_with_an_empty_body_is_still_the_sender_side_refusal(self, iot_env, monkeypatch):
        # The sender-side 403 (no SendDirectMessage grant) carries an empty
        # body; that one stays a FAIL that points at the policy version.
        _Transport.result = DirectResult(delivered=False, reason="forbidden", latency_ms=80.0, detail="")
        _Transport.result_for_cmd = DirectResult(delivered=False, reason="forbidden", latency_ms=80.0, detail="")
        monkeypatch.setattr(doctor, "_certificate_cn", lambda path: "thor-arm")
        line = doctor.check_iot_direct()
        assert line.startswith("  FAIL  ") and "predates the direct messaging grants" in line

    def test_forbidden_without_a_readable_cn_still_explains(self, iot_env, monkeypatch):
        _Transport.result = DirectResult(delivered=False, reason="forbidden", latency_ms=80.0)
        monkeypatch.setattr(doctor, "_certificate_cn", lambda path: None)
        assert "(403)" in doctor.check_iot_direct()

    def test_unavailable_is_a_failure(self, iot_env):
        _Transport.result = DirectResult(delivered=False, reason="unavailable", latency_ms=1.0, detail="no creds")
        line = doctor.check_iot_direct()
        assert line.startswith("  FAIL  ") and "no credential" in line

    @pytest.mark.parametrize("reason", ["throttled", "unconfirmed", "error"])
    def test_transient_reasons_are_warnings(self, iot_env, reason):
        _Transport.result = DirectResult(delivered=False, reason=reason, latency_ms=1.0)
        line = doctor.check_iot_direct()
        assert line.startswith("  WARN  ") and reason in line
        assert _Transport.instances[0].closed


class TestTheRowIsInTheTable:
    def test_iot_direct_follows_mesh(self):
        labels = [label for label, _ in doctor.CHECKS]
        assert labels.index("IoT Direct") == labels.index("Mesh") + 1
        assert dict(doctor.CHECKS)["IoT Direct"] == "check_iot_direct"


class TestCertificateCn:
    def test_reads_the_cn_from_a_real_certificate(self, tmp_path):
        pytest.importorskip("cryptography")
        import datetime

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives.serialization import Encoding
        from cryptography.x509.oid import NameOID

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "thor-arm")])
        now = datetime.datetime.now(datetime.UTC)
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(1)
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=1))
            .sign(key, hashes.SHA256())
        )
        path = tmp_path / "c.pem"
        path.write_bytes(cert.public_bytes(Encoding.PEM))
        assert doctor._certificate_cn(path) == "thor-arm"

    def test_unreadable_certificate_is_none(self, tmp_path):
        assert doctor._certificate_cn(tmp_path / "missing.pem") is None
        bad = tmp_path / "bad.pem"
        bad.write_text("not a certificate")
        assert doctor._certificate_cn(bad) is None
