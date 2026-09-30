"""A robot's child peers share the Thing's IoT key space, and nothing else does.

A simulation that joins the mesh as Thing ``so101-arm-01`` attaches every robot
in it as a child peer ``so101-arm-01__so101`` that publishes presence, state and
cameras over the SAME MQTT session as its parent (the process holds one
transport under the Thing's client id). AWS IoT answers a publish the connected
Thing's policy does not grant by ending the session, so with a policy that
granted ``strands/<thing>/*`` alone every child heartbeat reconnected the robot
(a connect/disconnect cycle about every 150 ms, presence heard once per 30 s,
nothing above DEBUG in the log). These tests pin the repair on all four sides:

* the robot policy documents (both estop postures) grant the child key space
  ``strands/${iot:Connection.Thing.ThingName}__*/...`` next to the Thing's own,
  in every statement the child needs (publish, reply, direct reply, subscribe,
  receive), and the operator policy needs nothing new;
* the grant cannot alias another Thing: a policy evaluator that substitutes the
  variable and matches ``*`` the way AWS does shows Thing ``a``'s certificate
  reaching ``strands/a__x/...`` and neither ``strands/b/...`` nor
  ``strands/ax/...``, and a Thing name containing the separator is refused at
  provisioning, so ``a__evil`` can never exist as a second Thing inside ``a``'s
  key space (verified live on 2026-09-29: 574 publishes on the child topic in
  60 s with zero disconnects, the two foreign topics ended the session with
  reason code 135);
* the transport WARNs once per topic, naming the topic, when the broker ends
  the session within a second of a publish, and stays quiet otherwise;
* the doctor row ``IoT Child Peers`` reads the account's default policy version
  and fails with the reprovision command when the grant is missing;
* ``reprovision_thing`` republishes the module-owned policy so an existing
  fleet picks the grant up (pinned next to the rotation tests in
  :mod:`tests.mesh.test_iot_provision_csr_and_direct_policy`).
"""

from __future__ import annotations

import fnmatch
import json
import logging
import types
from typing import Any

import pytest

from strands_robots import doctor
from strands_robots.mesh.iot import provision as prov
from strands_robots.mesh.iot.provision import (
    _OPERATOR_POLICY_DOC,
    CHILD_PEER_SEPARATOR,
    _robot_policy_doc,
    child_key_space_granted,
)
from strands_robots.mesh.transport import iot_transport
from strands_robots.mesh.transport.iot_transport import DISCONNECT_AFTER_PUBLISH_WINDOW_S, IotMqttTransport

THING_VAR = "${iot:Connection.Thing.ThingName}"
CN_VAR = "${iot:Certificate.Subject.CommonName}"


def _stmt(doc: dict[str, Any], sid: str) -> dict[str, Any]:
    return next(s for s in doc["Statement"] if s.get("Sid") == sid)


def _as_list(value: Any) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


@pytest.fixture(params=[True, False], ids=["robot", "robot-no-estop"])
def robot_doc(request) -> dict[str, Any]:
    return _robot_policy_doc(allow_estop_publish=request.param)


class TestRobotPolicyGrantsTheChildKeySpace:
    def test_own_topics_publish_covers_the_children(self, robot_doc):
        resources = _as_list(_stmt(robot_doc, "AllowOwnTopics")["Resource"])
        assert f"arn:aws:iot:*:*:topic/strands/{THING_VAR}/*" in resources
        assert f"arn:aws:iot:*:*:topic/strands/{THING_VAR}{CHILD_PEER_SEPARATOR}*/*" in resources

    def test_reply_publish_covers_the_children(self, robot_doc):
        resources = _as_list(_stmt(robot_doc, "AllowResponseToAnyOperator")["Resource"])
        assert f"arn:aws:iot:*:*:topic/strands/*/response/{THING_VAR}{CHILD_PEER_SEPARATOR}*/*" in resources

    def test_direct_reply_condition_covers_the_children(self, robot_doc):
        topics = _as_list(_stmt(robot_doc, "AllowDirectResponseToAnyOperator")["Condition"]["StringLike"]["iot:Topic"])
        assert f"strands/*/response/{CN_VAR}{CHILD_PEER_SEPARATOR}*/*" in topics

    def test_subscribe_and_receive_cover_the_children(self, robot_doc):
        subs = _as_list(_stmt(robot_doc, "AllowOwnSubscriptions")["Resource"])
        assert f"arn:aws:iot:*:*:topicfilter/strands/{THING_VAR}{CHILD_PEER_SEPARATOR}*/*" in subs
        recv = _as_list(_stmt(robot_doc, "AllowReceiveScoped")["Resource"])
        assert f"arn:aws:iot:*:*:topic/strands/{THING_VAR}{CHILD_PEER_SEPARATOR}*/cmd" in recv
        assert f"arn:aws:iot:*:*:topic/strands/{THING_VAR}{CHILD_PEER_SEPARATOR}*/response/*" in recv

    def test_receive_stays_off_the_children_health_and_state(self, robot_doc):
        # The same asymmetry the Thing's own topics keep: the robot publishes
        # them, the operator consumes them, the robot never receives its own copy.
        recv = json.dumps(_stmt(robot_doc, "AllowReceiveScoped")["Resource"])
        for suffix in ("/state", "/health", "/presence", "/safety/event"):
            assert f"{THING_VAR}{CHILD_PEER_SEPARATOR}*{suffix}" not in recv

    def test_the_shadow_stays_the_things_own(self, robot_doc):
        assert CHILD_PEER_SEPARATOR not in json.dumps(_stmt(robot_doc, "AllowShadow"))

    def test_the_operator_policy_needs_nothing_new(self):
        # ``strands/+/state`` matches ``strands/a__so101/state``: ``__`` is not a
        # topic level separator, so the operator's single-level wildcards already
        # reach every child.
        assert CHILD_PEER_SEPARATOR not in json.dumps(_OPERATOR_POLICY_DOC)
        assert "arn:aws:iot:*:*:topicfilter/strands/+/state" in json.dumps(_OPERATOR_POLICY_DOC)

    def test_documents_serialise(self, robot_doc):
        json.dumps(robot_doc)


def _may_publish(doc: dict[str, Any], thing: str, topic: str) -> bool:
    """AWS IoT's Allow evaluation for ``iot:Publish``: substitute the variable, then ``*`` matches any run."""
    for st in doc["Statement"]:
        if st.get("Effect") != "Allow" or "iot:Publish" not in _as_list(st["Action"]):
            continue
        for resource in _as_list(st["Resource"]):
            pattern = resource.replace(THING_VAR, thing)
            if fnmatch.fnmatchcase(f"arn:aws:iot:us-west-2:1:topic/{topic}", pattern):
                return True
    return False


class TestTheGrantCannotAliasAnotherThing:
    def test_a_reaches_its_children_and_no_other_thing(self, robot_doc):
        assert _may_publish(robot_doc, "childfix-a", "strands/childfix-a/state")
        assert _may_publish(robot_doc, "childfix-a", "strands/childfix-a__so101/state")
        assert _may_publish(robot_doc, "childfix-a", "strands/childfix-a__so101/camera/front")
        assert _may_publish(robot_doc, "childfix-a", "strands/childfix-op/response/childfix-a__so101/turn-1")
        assert not _may_publish(robot_doc, "childfix-a", "strands/childfix-b/state")
        assert not _may_publish(robot_doc, "childfix-a", "strands/childfix-ax/state")
        assert not _may_publish(robot_doc, "childfix-a", "strands/childfix-a_so101/state")
        assert not _may_publish(robot_doc, "childfix-a", "strands/childfix-op/response/childfix-b/turn-1")

    def test_a_thing_name_with_the_separator_is_refused(self):
        with pytest.raises(ValueError, match="child peer separator") as e:
            prov._validate_thing_name("childfix-a__evil")
        # The refusal names the Thing that could publish as it.
        assert "'childfix-a'" in str(e.value)
        for name in ("childfix-a", "childfix_a", "a_b_c", "a-b"):
            prov._validate_thing_name(name)

    def test_provision_and_reprovision_refuse_it_before_any_aws_call(self, monkeypatch, tmp_path):
        monkeypatch.setattr(prov, "_require_boto3", lambda: pytest.fail("boto3 was reached for a refused name"))
        with pytest.raises(ValueError, match="child peer separator"):
            prov.provision_robot("childfix-a__evil", cert_dir=tmp_path)
        with pytest.raises(ValueError, match="child peer separator"):
            prov.provision_operator("op__evil", cert_dir=tmp_path)
        with pytest.raises(ValueError, match="child peer separator"):
            prov.reprovision_thing("childfix-a__evil", cert_dir=tmp_path)


class _Packet:
    def __init__(self, reason_code: int | None) -> None:
        self.reason_code = reason_code


def _disconnect(reason_code: int | None = 135) -> Any:
    return types.SimpleNamespace(exception=None, disconnect_packet=_Packet(reason_code))


class _Client:
    def __init__(self) -> None:
        self.published: list[str] = []

    def publish(self, packet: Any) -> None:
        self.published.append(packet.topic)


@pytest.fixture
def transport(monkeypatch) -> IotMqttTransport:
    pytest.importorskip("awscrt")
    t = IotMqttTransport(thing_name="childfix-a", endpoint="x-ats.iot.us-west-2.amazonaws.com")
    t._client = _Client()
    t._connected.set()
    return t


class TestTheTransportNamesTheTopicThatEndedTheSession:
    LOGGER = "strands_robots.mesh.transport.iot_transport"

    def _warnings(self, caplog) -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]

    def test_a_disconnect_right_after_a_publish_warns_with_the_topic_and_the_command(self, transport, caplog):
        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            transport.put("strands/childfix-a__so101/state", {"t": 1.0})
            transport._on_disconnection(_disconnect(135))
        (w,) = self._warnings(caplog)
        assert "strands/childfix-a__so101/state" in w
        assert "reason code 135" in w
        assert "strands-robots iot reprovision childfix-a" in w
        assert "strands/childfix-a__*/*" in w
        assert not transport.is_alive()

    def test_the_warning_is_once_per_topic(self, transport, caplog):
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            for _ in range(5):
                transport._connected.set()
                transport.put("strands/childfix-a__so101/state", {"t": 1.0})
                transport._on_disconnection(_disconnect())
            transport._connected.set()
            transport.put("strands/childfix-a__so101/presence", {"t": 1.0})
            transport._on_disconnection(_disconnect())
        topics = [w.split(" after publishing ", 1)[1].split(" ", 1)[0] for w in self._warnings(caplog)]
        assert topics == ["strands/childfix-a__so101/state", "strands/childfix-a__so101/presence"]

    def test_a_disconnect_long_after_the_last_publish_is_not_blamed_on_it(self, transport, caplog, monkeypatch):
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            transport.put("strands/childfix-a/state", {"t": 1.0})
            at, topic = transport._last_publish
            transport._last_publish = (at - DISCONNECT_AFTER_PUBLISH_WINDOW_S - 0.5, topic)
            transport._on_disconnection(_disconnect(None))
        assert self._warnings(caplog) == []

    def test_a_disconnect_with_no_publish_at_all_is_quiet(self, transport, caplog):
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            transport._on_disconnection(_disconnect())
            transport._on_disconnection(types.SimpleNamespace())
        assert self._warnings(caplog) == []

    def test_a_reason_code_that_is_not_a_number_is_left_out(self, transport, caplog):
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            transport.put("strands/childfix-a__so101/state", {"t": 1.0})
            transport._on_disconnection(_disconnect(None))
        (w,) = self._warnings(caplog)
        assert "reason code" not in w

    def test_the_window_is_a_second(self):
        assert DISCONNECT_AFTER_PUBLISH_WINDOW_S == 1.0
        assert iot_transport.DISCONNECT_AFTER_PUBLISH_WINDOW_S is DISCONNECT_AFTER_PUBLISH_WINDOW_S


class _Account:
    """A control-plane stand-in: one Thing, one certificate, the policies given."""

    def __init__(self, policies: dict[str, dict[str, Any]], principals: int = 1) -> None:
        self.policies = policies
        self.principals = [f"arn:aws:iot:us-west-2:1:cert/{i}" for i in range(principals)]
        self.calls: list[str] = []

    def list_thing_principals(self, thingName: str) -> dict[str, Any]:
        self.calls.append("list_thing_principals")
        return {"principals": list(self.principals)}

    def list_attached_policies(self, target: str) -> dict[str, Any]:
        self.calls.append("list_attached_policies")
        return {"policies": [{"policyName": n} for n in self.policies]}

    def get_policy(self, policyName: str) -> dict[str, Any]:
        self.calls.append("get_policy")
        return {
            "policyName": policyName,
            "defaultVersionId": "3",
            "policyDocument": json.dumps(self.policies[policyName]),
        }


OLD_DOC = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowOwnTopics",
            "Effect": "Allow",
            "Action": ["iot:Publish", "iot:RetainPublish"],
            "Resource": [f"arn:aws:iot:*:*:topic/strands/{THING_VAR}/*"],
        }
    ],
}


class TestChildKeySpaceGranted:
    def test_the_current_document_grants_it(self):
        iot = _Account({"strands-robot-no-estop": _robot_policy_doc(allow_estop_publish=False)})
        granted, detail = child_key_space_granted(iot, "childfix-a")
        assert granted is True
        assert detail == "strands-robot-no-estop v3 grants strands/childfix-a__*/*"

    def test_the_document_from_before_the_grant_does_not(self):
        iot = _Account(
            {"strands-robot-no-estop": OLD_DOC, "customer-policy": {"Version": "2012-10-17", "Statement": []}}
        )
        granted, detail = child_key_space_granted(iot, "childfix-a")
        assert granted is False
        assert detail == "strands-robot-no-estop, customer-policy grant strands/childfix-a/* only"

    def test_a_deny_or_a_subscribe_only_mention_does_not_count(self):
        deny = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Deny",
                    "Action": "iot:Publish",
                    "Resource": f"arn:aws:iot:*:*:topic/strands/{THING_VAR}__*/*",
                },
                {
                    "Effect": "Allow",
                    "Action": "iot:Subscribe",
                    "Resource": f"arn:aws:iot:*:*:topicfilter/strands/{THING_VAR}__*/*",
                },
            ],
        }
        assert child_key_space_granted(_Account({"p": deny}), "childfix-a")[0] is False

    def test_nothing_to_judge_is_none_with_the_reason(self):
        assert child_key_space_granted(_Account({}, principals=0), "childfix-a") == (
            None,
            "no certificate is attached to childfix-a",
        )
        assert child_key_space_granted(_Account({}), "childfix-a") == (
            None,
            "no policy is attached to childfix-a's certificates",
        )

    def test_an_unparseable_account_document_is_treated_as_not_granting(self):
        iot = _Account({"p": OLD_DOC})
        iot.get_policy = lambda policyName: {"defaultVersionId": "1", "policyDocument": "{not json"}  # type: ignore[method-assign]
        assert child_key_space_granted(iot, "childfix-a")[0] is False


class _Boto3:
    def __init__(self, account: _Account) -> None:
        self.account = account
        self.regions: list[str | None] = []

    def client(self, service: str, region_name: str | None = None) -> _Account:
        assert service == "iot"
        self.regions.append(region_name)
        return self.account


@pytest.fixture
def iot_env(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_BACKEND", "iot")
    monkeypatch.setenv("STRANDS_IOT_THING_NAME", "childfix-a")
    monkeypatch.setenv("STRANDS_IOT_ENDPOINT", "x-ats.iot.us-west-2.amazonaws.com")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")


def _install_boto3(monkeypatch, account: _Account) -> _Boto3:
    fake = _Boto3(account)
    monkeypatch.setitem(__import__("sys").modules, "boto3", fake)
    return fake


class TestDoctorRow:
    def test_listed_after_iot_direct(self):
        names = [n for n, _ in doctor.CHECKS]
        assert names.index("IoT Child Peers") == names.index("IoT Direct") + 1
        assert dict(doctor.CHECKS)["IoT Child Peers"] == "check_iot_child_peers"

    def test_skips_off_the_iot_backends(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_BACKEND", "zenoh")
        assert doctor.check_iot_child_peers().startswith("  SKIP  iot child peers: STRANDS_MESH_BACKEND=zenoh")

    def test_fails_without_a_thing_name(self, iot_env, monkeypatch):
        monkeypatch.delenv("STRANDS_IOT_THING_NAME")
        out = doctor.check_iot_child_peers()
        assert "FAIL" in out and "STRANDS_IOT_THING_NAME" in out

    def test_passes_on_the_current_policy_in_the_endpoints_region(self, iot_env, monkeypatch):
        fake = _install_boto3(monkeypatch, _Account({"strands-robot": _robot_policy_doc(allow_estop_publish=True)}))
        out = doctor.check_iot_child_peers()
        assert "PASS" in out and "strands-robot v3 grants strands/childfix-a__*/*" in out
        assert fake.regions == ["us-west-2"]
        assert fake.account.calls == ["list_thing_principals", "list_attached_policies", "get_policy"]

    def test_fails_with_the_reprovision_command_on_the_old_policy(self, iot_env, monkeypatch):
        _install_boto3(monkeypatch, _Account({"strands-robot-no-estop": OLD_DOC}))
        out = doctor.check_iot_child_peers()
        assert "FAIL" in out
        assert "strands-robot-no-estop grant strands/childfix-a/* only" in out
        assert "childfix-a__<robot>" in out
        assert "Fix: strands-robots iot reprovision childfix-a" in out

    def test_fails_when_nothing_is_attached(self, iot_env, monkeypatch):
        _install_boto3(monkeypatch, _Account({}, principals=0))
        out = doctor.check_iot_child_peers()
        assert "FAIL" in out and "no certificate is attached to childfix-a" in out

    def test_warns_when_the_control_plane_cannot_be_read(self, iot_env, monkeypatch):
        account = _Account({})

        def _boom(thingName: str) -> dict[str, Any]:
            raise RuntimeError("NoCredentialsError: Unable to locate credentials")

        account.list_thing_principals = _boom  # type: ignore[method-assign]
        _install_boto3(monkeypatch, account)
        out = doctor.check_iot_child_peers()
        assert "WARN" in out and "could not be read" in out and "NoCredentialsError" in out

    def test_warns_without_boto3(self, iot_env, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _import(name: str, *a: Any, **kw: Any) -> Any:
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _import)
        out = doctor.check_iot_child_peers()
        assert "WARN" in out and "boto3 not installed" in out and "mesh-iot" in out

    def test_documented(self):
        from pathlib import Path

        page = (Path(__file__).resolve().parents[2] / "docs" / "start" / "doctor.md").read_text(encoding="utf-8")
        assert "| IoT Child Peers |" in page
