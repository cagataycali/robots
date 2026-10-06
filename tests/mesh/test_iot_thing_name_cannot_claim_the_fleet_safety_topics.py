"""A robot certificate's own key space never reaches the fleet-wide topics.

Both robot policies grant ``iot:Publish`` and ``iot:RetainPublish`` on
``strands/${iot:Connection.Thing.ThingName}/*``. The fleet-wide topics live
under literal first segments of the same namespace (``strands/safety/estop``,
``strands/safety/resume``, ``strands/broadcast``), so a Thing NAMED ``safety``
was granted the fleet stop and its release through its own prefix, with the
``strands-robot-no-estop`` policy attached. ``_validate_thing_name`` checked
the charset and the child separator only, and the Fleet Provisioning hook
checked the serial, the allowlist and the certificate CN but never the Thing
name itself, which the device chooses.

Now: the fleet-wide segments are reserved names on the Python path and in the
hook; the hook binds the Thing name to the allowlisted serial; every robot
certificate carries an explicit Deny on retained safety publishes and the
no-estop posture denies publishing to the safety topics at all; a certificate
issued before the default flipped can be moved to the no-estop policy with
:func:`withdraw_fleet_stop_grant`, and :func:`reprovision_thing` no longer
carries the stop grant over without being told to.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh.iot import bootstrap as b
from strands_robots.mesh.iot import provision as prov
from strands_robots.mesh.iot.cli import _parser, main

_THING_VAR = "${iot:Connection.Thing.ThingName}"


def _statements(doc: dict[str, Any], effect: str) -> list[dict[str, Any]]:
    return [st for st in doc["Statement"] if st.get("Effect") == effect]


def _as_list(value: Any) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def _denied(doc: dict[str, Any], action: str) -> set[str]:
    """Every resource an explicit Deny on *action* names in *doc*."""
    out: set[str] = set()
    for st in _statements(doc, "Deny"):
        if action in _as_list(st["Action"]):
            out.update(_as_list(st["Resource"]))
    return out


def _literal_first_segments(docs: list[dict[str, Any]]) -> set[str]:
    """The literal ``strands/<segment>`` first segments the policy documents name."""
    found: set[str] = set()
    pattern = re.compile(r"arn:aws:iot:\*:\*:(?:topic|topicfilter)/strands/([^/]+)")
    for doc in docs:
        for st in doc["Statement"]:
            for res in _as_list(st.get("Resource", [])):
                m = pattern.match(res)
                if not m:
                    continue
                segment = m.group(1)
                if segment in ("*", "+") or "${" in segment:
                    continue
                found.add(segment)
    return found


class TestReservedNames:
    def test_the_reserved_set_is_every_literal_fleet_segment_the_policies_grant(self):
        docs = [
            prov._robot_policy_doc(allow_estop_publish=True),
            prov._robot_policy_doc(allow_estop_publish=False),
            prov._ROBOT_CHILDREN_POLICY_DOC,
            prov._OPERATOR_POLICY_DOC,
            prov._OPERATOR_OBSERVE_POLICY_DOC,
        ]
        assert set(prov.RESERVED_THING_NAMES) == _literal_first_segments(docs)
        assert {"safety", "broadcast"} <= set(prov.RESERVED_THING_NAMES)

    @pytest.mark.parametrize("name", ["safety", "broadcast", "Safety", "SAFETY", "Broadcast"])
    def test_a_reserved_name_is_refused_before_any_aws_call(self, name):
        with pytest.raises(ValueError, match="reserved"):
            prov._validate_thing_name(name)

    @pytest.mark.parametrize("name", ["so101-arm-01", "safety-1", "my-safety", "broadcaster"])
    def test_a_name_that_merely_contains_a_reserved_word_is_fine(self, name):
        prov._validate_thing_name(name)

    def test_provision_robot_refuses_the_name_before_boto3_is_resolved(self, tmp_path):
        boto = MagicMock()
        with patch.object(prov, "_require_boto3", lambda: boto):
            with pytest.raises(ValueError, match="reserved"):
                prov.provision_robot("safety", cert_dir=tmp_path)
        assert boto.client.call_count == 0


class TestDenyStatements:
    def test_the_no_estop_posture_denies_publishing_to_the_safety_topics(self):
        doc = prov._robot_policy_doc(allow_estop_publish=False)
        for action in ("iot:Publish", "iot:RetainPublish"):
            assert "arn:aws:iot:*:*:topic/strands/safety/*" in _denied(doc, action)

    def test_every_robot_certificate_denies_a_retained_safety_message(self):
        # The children policy rides on every robot certificate next to either
        # robot posture, so a Deny there reaches the safety authority too.
        denied = _denied(prov._ROBOT_CHILDREN_POLICY_DOC, "iot:RetainPublish")
        assert "arn:aws:iot:*:*:topic/strands/safety/*" in denied

    def test_no_robot_posture_may_publish_into_the_broadcast_segment(self):
        denied = _denied(prov._ROBOT_CHILDREN_POLICY_DOC, "iot:Publish")
        assert {"arn:aws:iot:*:*:topic/strands/broadcast", "arn:aws:iot:*:*:topic/strands/broadcast/*"} <= denied

    def test_the_safety_authority_still_publishes_the_stop_and_its_release(self):
        doc = prov._robot_policy_doc(allow_estop_publish=True)
        allowed: set[str] = set()
        for st in _statements(doc, "Allow"):
            if "iot:Publish" in _as_list(st["Action"]):
                allowed.update(_as_list(st["Resource"]))
        assert {"arn:aws:iot:*:*:topic/strands/safety/estop", "arn:aws:iot:*:*:topic/strands/safety/resume"} <= allowed
        # ... and nothing in its own documents denies that exact publish.
        for d in (doc, prov._ROBOT_CHILDREN_POLICY_DOC):
            assert not {
                "arn:aws:iot:*:*:topic/strands/safety/estop",
                "arn:aws:iot:*:*:topic/strands/safety/*",
            } & _denied(d, "iot:Publish")

    def test_the_documents_still_fit_the_service_cap(self):
        for name, build in prov._OWNED_POLICY_DOCUMENTS.items():
            assert prov.policy_document_size_error(name, build()) is None

    def test_the_deny_does_not_read_as_a_child_key_space_grant(self):
        assert prov._grants_child_key_space(prov._ROBOT_CHILDREN_POLICY_DOC) is True
        deny_only = {
            "Version": "2012-10-17",
            "Statement": [st for st in prov._ROBOT_CHILDREN_POLICY_DOC["Statement"] if st["Effect"] == "Deny"],
        }
        assert prov._grants_child_key_space(deny_only) is False


# --- the Fleet Provisioning hook ------------------------------------------


def _run_hook(event: dict[str, Any], *, thing_exists: bool = False, serial_allowed: bool = True) -> dict[str, Any]:
    fake_boto3 = MagicMock()
    iot_client = MagicMock()
    ssm_client = MagicMock()
    iot_client.exceptions.ResourceNotFoundException = type("RNF", (Exception,), {})
    ssm_client.exceptions.ParameterNotFound = type("PNF", (Exception,), {})
    if thing_exists:
        iot_client.describe_thing.return_value = {"thingName": "x"}
    else:
        iot_client.describe_thing.side_effect = iot_client.exceptions.ResourceNotFoundException()
    if not serial_allowed:
        ssm_client.get_parameter.side_effect = ssm_client.exceptions.ParameterNotFound()
    fake_boto3.client.side_effect = lambda name, *a, **k: {"iot": iot_client, "ssm": ssm_client}[name]
    with patch.dict(sys.modules, {"boto3": fake_boto3}):
        g: dict[str, Any] = {}
        exec(compile(b._PROVISIONING_HOOK_SOURCE, "<hook>", "exec"), g)
        return g["lambda_handler"](event, MagicMock())


def _hook_constant(name: str) -> Any:
    tree = ast.parse(b._PROVISIONING_HOOK_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned in the hook source")


class TestProvisioningHook:
    @pytest.mark.parametrize("name", ["safety", "broadcast", "Safety"])
    def test_a_reserved_thing_name_is_denied_even_with_an_allowlisted_serial(self, name):
        res = _run_hook({"parameters": {"SerialNumber": name, "ThingName": name}})
        assert res == {"allowProvisioning": False}

    def test_the_hook_reserves_the_same_names_as_the_provisioner(self):
        assert set(_hook_constant("_RESERVED_THING_NAMES")) == set(prov.RESERVED_THING_NAMES)

    def test_a_thing_name_the_serial_does_not_derive_is_denied(self):
        res = _run_hook({"parameters": {"SerialNumber": "robot-001", "ThingName": "ops-console-9"}})
        assert res == {"allowProvisioning": False}

    def test_the_serial_itself_is_an_accepted_thing_name(self):
        res = _run_hook({"parameters": {"SerialNumber": "robot-001", "ThingName": "robot-001"}})
        assert res == {"allowProvisioning": True}

    def test_one_model_token_before_the_serial_is_the_documented_derivation(self):
        res = _run_hook({"parameters": {"SerialNumber": "robot-001", "ThingName": "g1-robot-001"}})
        assert res == {"allowProvisioning": True}

    @pytest.mark.parametrize("name", ["g1-x-robot-001", "g1_robot-001", "robot-001-g1", "robot-0011"])
    def test_any_other_shape_around_the_serial_is_denied(self, name):
        res = _run_hook({"parameters": {"SerialNumber": "robot-001", "ThingName": name}})
        assert res == {"allowProvisioning": False}

    @pytest.mark.parametrize("name", ["a__b", "robot-001_", "robot.001"])
    def test_a_thing_name_the_provisioner_would_refuse_is_denied_too(self, name):
        res = _run_hook({"parameters": {"SerialNumber": name, "ThingName": name}})
        assert res == {"allowProvisioning": False}

    def test_a_missing_thing_name_is_denied(self):
        res = _run_hook({"parameters": {"SerialNumber": "robot-001"}})
        assert res == {"allowProvisioning": False}

    def test_the_name_checks_run_before_any_aws_read(self):
        fake_boto3 = MagicMock()
        with patch.dict(sys.modules, {"boto3": fake_boto3}):
            g: dict[str, Any] = {}
            exec(compile(b._PROVISIONING_HOOK_SOURCE, "<hook>", "exec"), g)
            iot_calls_before = fake_boto3.client.return_value.describe_thing.call_count
            res = g["lambda_handler"]({"parameters": {"SerialNumber": "robot-001", "ThingName": "safety"}}, MagicMock())
        assert res == {"allowProvisioning": False}
        assert fake_boto3.client.return_value.describe_thing.call_count == iot_calls_before
        assert fake_boto3.client.return_value.get_parameter.call_count == 0


# --- moving certificates issued before the default flipped -----------------


class _Account:
    """A stand-in ``iot`` client holding certificates with policies attached."""

    def __init__(self, attached: dict[str, list[str]], things: dict[str, list[str]]) -> None:
        self.attached = {k: list(v) for k, v in attached.items()}
        self.things = things  # cert arn -> thing names
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.meta = MagicMock(region_name="us-west-2")
        self.exceptions = MagicMock()
        self.exceptions.ResourceNotFoundException = type("RNF", (Exception,), {})

    def __getattr__(self, name: str) -> Any:
        def _call(**kw: Any) -> Any:
            self.calls.append((name, kw))
            return self._answer(name, kw)

        return _call

    def _answer(self, name: str, kw: dict[str, Any]) -> Any:
        if name == "list_targets_for_policy":
            targets = [cert for cert, pols in self.attached.items() if kw["policyName"] in pols]
            return {"targets": targets}
        if name == "list_principal_things":
            return {"things": list(self.things.get(kw["principal"], []))}
        if name == "list_attached_policies":
            return {"policies": [{"policyName": p} for p in self.attached.get(kw["target"], [])]}
        if name == "attach_policy":
            pols = self.attached.setdefault(kw["target"], [])
            if kw["policyName"] not in pols:
                pols.append(kw["policyName"])
            return {}
        if name == "detach_policy":
            self.attached[kw["target"]].remove(kw["policyName"])
            return {}
        if name == "get_policy":
            raise self.exceptions.ResourceNotFoundException()
        if name == "create_policy":
            return {"policyArn": f"arn:aws:iot:us-west-2:1:policy/{kw['policyName']}"}
        return {}

    def names(self) -> list[str]:
        return [n for n, _ in self.calls]


CERT_A = "arn:aws:iot:us-west-2:1:cert/aaaa"
CERT_B = "arn:aws:iot:us-west-2:1:cert/bbbb"
CERT_C = "arn:aws:iot:us-west-2:1:cert/cccc"


@pytest.fixture
def account(monkeypatch: pytest.MonkeyPatch) -> _Account:
    acct = _Account(
        attached={
            CERT_A: ["strands-robot", "strands-robot-children"],
            CERT_B: ["strands-robot"],
            CERT_C: ["strands-robot-no-estop", "strands-robot-children"],
        },
        things={CERT_A: ["so101-arm-01"], CERT_B: ["watchdog-1"], CERT_C: ["so101-arm-02"]},
    )
    monkeypatch.setattr(prov, "_require_boto3", lambda: MagicMock(client=lambda *a, **k: acct))
    return acct


class TestWithdrawFleetStopGrant:
    def test_a_dry_run_reports_and_changes_nothing(self, account: _Account):
        report = prov.withdraw_fleet_stop_grant()
        assert report.applied is False
        assert report.moved == ("so101-arm-01", "watchdog-1")
        assert "attach_policy" not in account.names() and "detach_policy" not in account.names()

    def test_apply_moves_every_certificate_to_the_no_estop_policy_grant_first(self, account: _Account):
        report = prov.withdraw_fleet_stop_grant(apply=True)
        assert report.applied is True and report.moved == ("so101-arm-01", "watchdog-1")
        for cert in (CERT_A, CERT_B):
            assert "strands-robot" not in account.attached[cert]
            assert "strands-robot-no-estop" in account.attached[cert]
            # A robot certificate from before the child key space grant gets it too.
            assert "strands-robot-children" in account.attached[cert]
        names = account.names()
        assert names.index("attach_policy") < names.index("detach_policy"), "no gap without a policy"
        # The documents this module owns are republished first, so the new
        # Deny statements are in the account before anything is moved.
        assert names.index("create_policy") < names.index("attach_policy")

    def test_a_named_safety_authority_keeps_its_grant(self, account: _Account):
        report = prov.withdraw_fleet_stop_grant(safety_authorities=["watchdog-1"], apply=True)
        assert report.moved == ("so101-arm-01",) and report.kept == ("watchdog-1",)
        assert account.attached[CERT_B] == ["strands-robot"]

    def test_a_certificate_already_on_the_no_estop_policy_is_left_alone(self, account: _Account):
        prov.withdraw_fleet_stop_grant(apply=True)
        assert account.attached[CERT_C] == ["strands-robot-no-estop", "strands-robot-children"]

    def test_an_authority_name_is_validated_like_any_thing_name(self, account: _Account):
        with pytest.raises(ValueError, match="reserved"):
            prov.withdraw_fleet_stop_grant(safety_authorities=["safety"])

    def test_the_cli_verb_is_a_dry_run_unless_told_to_apply(self, account: _Account, capsys):
        assert main(["withdraw-estop-publish", "--region", "us-west-2"]) == 0
        out = capsys.readouterr().out
        assert "so101-arm-01" in out and "watchdog-1" in out and "--apply" in out
        assert "detach_policy" not in account.names()
        assert main(["withdraw-estop-publish", "--keep", "watchdog-1", "--apply"]) == 0
        assert account.attached[CERT_B] == ["strands-robot"]
        assert "strands-robot" not in account.attached[CERT_A]


class _Rotating(_Account):
    OLD = "arn:aws:iot:us-west-2:1:cert/old0000000000"

    def __init__(self, policies: list[str]) -> None:
        super().__init__(attached={self.OLD: policies}, things={})
        self.principals = [self.OLD]

    def _answer(self, name: str, kw: dict[str, Any]) -> Any:
        if name == "describe_thing":
            return {"thingArn": f"arn:aws:iot:us-west-2:1:thing/{kw['thingName']}"}
        if name == "list_thing_principals":
            return {"principals": list(self.principals)}
        if name == "attach_thing_principal":
            self.principals.append(kw["principal"])
            return {}
        if name == "detach_thing_principal":
            self.principals.remove(kw["principal"])
            return {}
        if name == "create_certificate_from_csr":
            return {
                "certificateArn": "arn:aws:iot:us-west-2:1:cert/abc",
                "certificateId": "abc",
                "certificatePem": "pem",
            }
        if name == "describe_endpoint":
            return {"endpointAddress": "x-ats.iot.us-west-2.amazonaws.com"}
        return super()._answer(name, kw)


@pytest.fixture
def rotating(monkeypatch: pytest.MonkeyPatch):
    def _make(policies: list[str]) -> _Rotating:
        client = _Rotating(policies)
        monkeypatch.setattr(prov, "_require_boto3", lambda: MagicMock(client=lambda *a, **k: client))
        monkeypatch.setattr(prov, "_ensure_ca", lambda ca_path: ca_path.write_text("ca", encoding="utf-8"))
        monkeypatch.setattr(prov, "_build_csr", lambda thing_name, key_path: "csr")
        return client

    return _make


class TestReprovisionDoesNotCarryTheStopGrantSilently:
    def test_a_certificate_on_the_estop_policy_is_refused_without_a_decision(self, rotating, tmp_path):
        iot = rotating(["strands-robot", "strands-robot-children"])
        with pytest.raises(ValueError, match="estop_publish"):
            prov.reprovision_thing("so101-r", cert_dir=tmp_path)
        assert "create_certificate_from_csr" not in iot.names(), "a refused rotation issues nothing"

    def test_keeping_it_is_explicit_and_logged(self, rotating, tmp_path, caplog):
        iot = rotating(["strands-robot", "strands-robot-children"])
        with caplog.at_level("WARNING", logger="strands_robots.mesh.iot.provision"):
            result = prov.reprovision_thing("so101-r", cert_dir=tmp_path, estop_publish=True)
        assert result.policy_name == "strands-robot"
        assert "strands-robot" in iot.attached["arn:aws:iot:us-west-2:1:cert/abc"]
        assert any("safety authority" in rec.getMessage() for rec in caplog.records)

    def test_dropping_it_moves_the_rotated_identity_to_the_no_estop_policy(self, rotating, tmp_path):
        iot = rotating(["strands-robot", "strands-robot-children"])
        result = prov.reprovision_thing("so101-r", cert_dir=tmp_path, estop_publish=False)
        assert result.policy_name == "strands-robot-no-estop"
        new = iot.attached["arn:aws:iot:us-west-2:1:cert/abc"]
        assert "strands-robot" not in new and "strands-robot-no-estop" in new and "strands-robot-children" in new

    def test_a_certificate_without_the_grant_needs_no_decision(self, rotating, tmp_path):
        rotating(["strands-robot-no-estop", "strands-robot-children"])
        result = prov.reprovision_thing("so101-r", cert_dir=tmp_path)
        assert result.policy_name == "strands-robot-no-estop"

    def test_the_cli_names_the_flag_in_its_refusal(self, rotating, tmp_path, capsys):
        rotating(["strands-robot"])
        assert main(["reprovision", "so101-r", "--cert-dir", str(tmp_path)]) == 1
        err = capsys.readouterr().err
        assert "--estop-publish" in err
        assert main(["reprovision", "so101-r", "--cert-dir", str(tmp_path), "--estop-publish", "drop"]) == 0

    def test_the_parser_offers_keep_and_drop_only(self):
        parser = _parser()
        args = parser.parse_args(["reprovision", "x", "--estop-publish", "keep"])
        assert args.estop_publish == "keep"
        with pytest.raises(SystemExit):
            parser.parse_args(["reprovision", "x", "--estop-publish", "yes"])


def test_the_template_documents_the_thing_name_derivation():
    # Operators read the template's docstring to name devices; it must state
    # the shape the hook accepts.
    src = b._provisioning_template_body.__doc__ or ""
    assert "SerialNumber" in src and "<model>-<SerialNumber>" in src
    assert json.dumps(b._provisioning_template_body())  # still a document
