#!/usr/bin/env python3
"""Provisioning for AWS IoT Core Direct Messaging: CSR certificates and the policy grants.

No AWS. A recording stand-in for the ``iot`` client answers every call, so
each test pins one observable:

  - the certificate is issued with ``create_certificate_from_csr`` from a
    request whose subject is ``CN=<thing_name>, O=strands-robots``, with a
    2048-bit RSA key generated locally and written owner-only; the old
    ``create_keys_and_certificate`` is never called; ``ProvisionedThing.subject_cn``
    equals the Thing name;
  - both CSR builders (``cryptography`` and the ``openssl`` command) produce a
    request the other can parse, and a machine with neither is refused with
    a message naming both, before any AWS call;
  - the robot policy (and its no-estop twin) carries
    ``AllowDirectResponseToAnyOperator`` on ``${iot:Certificate.Subject.CommonName}``
    and never on ``${iot:Connection.Thing.ThingName}`` (an HTTPS call has no
    connection to resolve it from); the operator policy carries
    ``AllowDirectCommandToAnyRobot`` on ``strands/*/cmd`` only (broadcast is
    pub/sub), ``OperatorAnnounceSelf`` on its own presence/health, and no
    ``Condition`` on ``OperatorShadow`` (AWS refuses the attribute key);
  - ``_ensure_policy`` leaves an identical document alone, publishes a
    changed one as the new default version, and evicts the oldest
    non-default version at the five-version cap.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from strands_robots.mesh.iot import provision as prov
from strands_robots.mesh.iot.provision import (
    _OPERATOR_POLICY_DOC,
    _ROBOT_POLICY_DOC,
    CSR_ORGANIZATION,
    _build_csr,
    _create_cert,
    _ensure_policy,
    _robot_policy_doc,
    provision_operator,
    provision_robot,
)


class _NotFound(Exception):
    pass


class _Iot:
    class exceptions:  # noqa: N801 - boto3 shape
        ResourceNotFoundException = _NotFound

    class meta:  # noqa: N801 - boto3 shape
        region_name = "us-west-2"

    def __init__(self, existing_policy: dict[str, Any] | None = None, versions: int = 1) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.existing_policy = existing_policy
        self.versions = versions

    def __getattr__(self, name: str) -> Any:
        def _call(**kw: Any) -> Any:
            self.calls.append((name, kw))
            return self._answer(name, kw)

        return _call

    def _answer(self, name: str, kw: dict[str, Any]) -> Any:
        if name == "describe_thing":
            raise _NotFound()
        if name == "create_thing":
            return {"thingArn": f"arn:aws:iot:us-west-2:1:thing/{kw['thingName']}"}
        if name == "get_policy":
            if self.existing_policy is None:
                raise _NotFound()
            return {
                "policyArn": f"arn:aws:iot:us-west-2:1:policy/{kw['policyName']}",
                "defaultVersionId": str(self.versions),
                "policyDocument": json.dumps(self.existing_policy),
            }
        if name == "create_policy":
            return {"policyArn": f"arn:aws:iot:us-west-2:1:policy/{kw['policyName']}"}
        if name == "list_policy_versions":
            return {
                "policyVersions": [
                    {"versionId": str(i), "isDefaultVersion": i == self.versions} for i in range(1, self.versions + 1)
                ]
            }
        if name == "create_policy_version":
            return {"policyVersionId": str(self.versions + 1)}
        if name == "list_thing_principals":
            return {"principals": []}
        if name == "create_certificate_from_csr":
            return {
                "certificateArn": "arn:aws:iot:us-west-2:1:cert/abc",
                "certificateId": "abc",
                "certificatePem": "PEM",
            }
        if name == "describe_endpoint":
            return {"endpointAddress": "x-ats.iot.us-west-2.amazonaws.com"}
        return {}

    def names(self) -> list[str]:
        return [n for n, _ in self.calls]


@pytest.fixture
def iot(monkeypatch, tmp_path):
    client = _Iot()
    monkeypatch.setattr(
        prov, "_require_boto3", lambda: type("B", (), {"client": staticmethod(lambda *a, **kw: client)})
    )
    monkeypatch.setattr(prov, "_ensure_ca", lambda ca_path: ca_path.write_text("ca"))
    return client


def _csr_subject(csr_pem: str) -> str:
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl not on PATH")
    out = subprocess.run(
        [openssl, "req", "-noout", "-subject", "-in", "/dev/stdin"], input=csr_pem, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    # OpenSSL 3.x prints the subject as "CN = x, O = y"; 1.x as "CN=x/O=y".
    # Normalise the spacing around '=' so the substring checks hold on both.
    return re.sub(r"\s*=\s*", "=", out.stdout.strip())


class TestCsrIssuance:
    def test_provision_robot_issues_from_a_csr_with_the_thing_name_as_cn(self, iot, tmp_path):
        result = provision_robot("so101-a", cert_dir=tmp_path)
        assert "create_certificate_from_csr" in iot.names()
        assert "create_keys_and_certificate" not in iot.names()
        (_, kw), *_ = [c for c in iot.calls if c[0] == "create_certificate_from_csr"]
        assert kw["setAsActive"] is True
        assert "CN=so101-a" in _csr_subject(kw["certificateSigningRequest"])
        assert f"O={CSR_ORGANIZATION}" in _csr_subject(kw["certificateSigningRequest"])
        assert result.subject_cn == "so101-a"
        assert result.cert_path.read_text() == "PEM"
        assert result.key_path.read_text().startswith("-----BEGIN")
        assert oct(result.key_path.stat().st_mode & 0o777) == "0o600"
        assert oct(result.cert_path.stat().st_mode & 0o777) == "0o600"

    def test_provision_operator_does_the_same(self, iot, tmp_path):
        result = provision_operator("operator-a", cert_dir=tmp_path)
        assert "create_certificate_from_csr" in iot.names()
        assert result.subject_cn == "operator-a"

    def test_the_csr_precedes_every_aws_call_that_needs_it(self, iot, tmp_path, monkeypatch):
        def _boom(thing_name: str, key_path: Path) -> str:
            raise RuntimeError("no builder")

        monkeypatch.setattr(prov, "_build_csr", _boom)
        with pytest.raises(RuntimeError, match="no builder"):
            _create_cert(iot, tmp_path / "c.pem", tmp_path / "k.pem", "so101-a")
        assert iot.calls == []
        assert not (tmp_path / "c.pem").exists()

    def test_a_re_run_overwrites_the_key_owner_only(self, tmp_path):
        key = tmp_path / "k.pem"
        key.write_text("old")
        os.chmod(key, 0o644)
        _build_csr("thing-x", key)
        assert key.read_text() != "old"
        assert oct(key.stat().st_mode & 0o777) == "0o600"


class TestCredentialWritesDoNotFollowSymlinks:
    """A planted symlink at a credential path is refused, and its target is untouched."""

    def test_a_symlinked_key_path_is_refused_before_any_key_is_written(self, tmp_path):
        target = tmp_path / "elsewhere.txt"
        target.write_text("untouched")
        link = tmp_path / "thing-x.private.key"
        link.symlink_to(target)
        with pytest.raises(RuntimeError, match="symlink"):
            _build_csr("thing-x", link)
        assert target.read_text() == "untouched"
        assert link.is_symlink()

    def test_a_symlinked_cert_path_is_refused_and_the_issued_pem_never_lands_elsewhere(self, iot, tmp_path):
        target = tmp_path / "elsewhere.pem"
        target.write_text("untouched")
        cert_link = tmp_path / "so101-l.cert.pem"
        cert_link.symlink_to(target)
        with pytest.raises(RuntimeError, match="symlink"):
            _create_cert(iot, cert_link, tmp_path / "so101-l.private.key", "so101-l")
        assert target.read_text() == "untouched"

    def test_a_regular_file_is_replaced_in_place(self, tmp_path):
        key = tmp_path / "k.pem"
        key.write_text("old")
        prov._write_private(key, "new")
        assert key.read_text() == "new"
        assert oct(key.stat().st_mode & 0o777) == "0o600"

    def test_the_open_uses_nofollow(self):
        import ast
        import inspect

        src = inspect.getsource(prov._write_private)
        opens = [
            n
            for n in ast.walk(ast.parse(textwrap_dedent(src)))
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "os.open"
        ]
        assert opens, "no os.open in _write_private"
        assert all("nofollow" in ast.unparse(c) for c in opens)


def textwrap_dedent(src: str) -> str:
    import textwrap

    return textwrap.dedent(src)


class TestBothBuildersAgree:
    def test_cryptography_builder(self, tmp_path):
        pytest.importorskip("cryptography")
        csr = _build_csr("thing-crypto", tmp_path / "k.pem")
        subject = _csr_subject(csr)
        assert "CN=thing-crypto" in subject and f"O={CSR_ORGANIZATION}" in subject
        assert "RSA" in (tmp_path / "k.pem").read_text() or "PRIVATE KEY" in (tmp_path / "k.pem").read_text()

    def test_openssl_builder(self, tmp_path, monkeypatch):
        if shutil.which("openssl") is None:
            pytest.skip("openssl not on PATH")
        import builtins

        real_import = builtins.__import__

        def _no_cryptography(name: str, *a: Any, **kw: Any) -> Any:
            if name == "cryptography" or name.startswith("cryptography."):
                raise ImportError(name)
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _no_cryptography)
        csr = _build_csr("thing-ssl", tmp_path / "k.pem")
        monkeypatch.undo()
        subject = _csr_subject(csr)
        assert "CN=thing-ssl" in subject and f"O={CSR_ORGANIZATION}" in subject
        assert oct((tmp_path / "k.pem").stat().st_mode & 0o777) == "0o600"

    def test_neither_builder_is_refused_naming_both(self, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _no_cryptography(name: str, *a: Any, **kw: Any) -> Any:
            if name == "cryptography" or name.startswith("cryptography."):
                raise ImportError(name)
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _no_cryptography)
        monkeypatch.setattr(prov.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="cryptography.*openssl"):
            _build_csr("thing-none", tmp_path / "k.pem")
        assert not (tmp_path / "k.pem").exists()

    def test_openssl_failure_is_reported(self, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _no_cryptography(name: str, *a: Any, **kw: Any) -> Any:
            if name == "cryptography" or name.startswith("cryptography."):
                raise ImportError(name)
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _no_cryptography)
        monkeypatch.setattr(prov.shutil, "which", lambda name: "/bin/false-openssl")
        monkeypatch.setattr(
            prov.subprocess,
            "run",
            lambda *a, **kw: subprocess.CompletedProcess(a, 1, stdout="", stderr="unable to load key"),
        )
        with pytest.raises(RuntimeError, match="openssl req failed.*unable to load key"):
            _build_csr("thing-fail", tmp_path / "k.pem")


def _stmt(doc: dict[str, Any], sid: str) -> dict[str, Any]:
    matches = [s for s in doc["Statement"] if s.get("Sid") == sid]
    assert len(matches) == 1, f"{sid}: {len(matches)} statements"
    return matches[0]


class TestPolicyGrants:
    @pytest.mark.parametrize(
        "doc", [_ROBOT_POLICY_DOC, _robot_policy_doc(allow_estop_publish=False)], ids=["robot", "robot-no-estop"]
    )
    def test_robot_direct_reply_grant_uses_the_certificate_cn(self, doc):
        st = _stmt(doc, "AllowDirectResponseToAnyOperator")
        assert st["Effect"] == "Allow"
        assert st["Action"] == "iot:SendDirectMessage"
        assert st["Resource"] == "arn:aws:iot:*:*:client/*"
        topic = st["Condition"]["StringLike"]["iot:Topic"]
        assert topic == "strands/*/response/${iot:Certificate.Subject.CommonName}/*"
        assert "Connection.Thing.ThingName" not in json.dumps(st)

    def test_robot_publish_reply_grant_is_still_there(self):
        # The fallback for a robot whose certificate predates the CSR default.
        st = _stmt(_ROBOT_POLICY_DOC, "AllowResponseToAnyOperator")
        assert "strands/*/response/${iot:Connection.Thing.ThingName}/*" in json.dumps(st["Resource"])

    def test_no_estop_twin_differs_only_by_the_estop_statement(self):
        full = {s["Sid"] for s in _ROBOT_POLICY_DOC["Statement"]}
        twin = {s["Sid"] for s in _robot_policy_doc(allow_estop_publish=False)["Statement"]}
        assert full - twin == {"AllowSafetyEstop"}

    def test_operator_direct_command_grant_is_cmd_only(self):
        st = _stmt(_OPERATOR_POLICY_DOC, "AllowDirectCommandToAnyRobot")
        assert st["Action"] == "iot:SendDirectMessage"
        assert st["Resource"] == "arn:aws:iot:*:*:client/*"
        assert st["Condition"] == {"StringLike": {"iot:Topic": "strands/*/cmd"}}
        assert "broadcast" not in json.dumps(st)

    def test_operator_announces_its_own_presence_and_health_only(self):
        st = _stmt(_OPERATOR_POLICY_DOC, "OperatorAnnounceSelf")
        assert set(st["Action"]) == {"iot:Publish", "iot:RetainPublish"}
        assert st["Resource"] == [
            "arn:aws:iot:*:*:topic/strands/${iot:Connection.Thing.ThingName}/presence",
            "arn:aws:iot:*:*:topic/strands/${iot:Connection.Thing.ThingName}/health",
        ]

    def test_operator_shadow_has_no_unsupported_condition(self):
        st = _stmt(_OPERATOR_POLICY_DOC, "OperatorShadow")
        assert "Condition" not in st
        assert "Connection.Thing.Attributes" not in json.dumps(_OPERATOR_POLICY_DOC)

    @pytest.mark.parametrize(
        "doc", [_ROBOT_POLICY_DOC, _OPERATOR_POLICY_DOC, _robot_policy_doc(allow_estop_publish=False)]
    )
    def test_documents_round_trip_as_json_with_unique_sids(self, doc):
        text = json.dumps(doc)
        assert json.loads(text) == doc
        sids = [s["Sid"] for s in doc["Statement"]]
        assert len(sids) == len(set(sids))


class TestEnsurePolicyVersions:
    DOC = {
        "Version": "2012-10-17",
        "Statement": [{"Sid": "A", "Effect": "Allow", "Action": "iot:Connect", "Resource": "*"}],
    }

    def test_absent_policy_is_created(self):
        iot = _Iot()
        arn = _ensure_policy(iot, "p", self.DOC)
        assert arn.endswith("policy/p")
        assert iot.names() == ["get_policy", "create_policy"]

    def test_identical_document_is_left_alone_regardless_of_key_order(self):
        reordered = {
            "Statement": [{"Resource": "*", "Action": "iot:Connect", "Effect": "Allow", "Sid": "A"}],
            "Version": "2012-10-17",
        }
        iot = _Iot(existing_policy=reordered)
        _ensure_policy(iot, "p", self.DOC)
        assert iot.names() == ["get_policy"]

    def test_changed_document_becomes_the_new_default_version(self):
        iot = _Iot(existing_policy={"Version": "2012-10-17", "Statement": []}, versions=2)
        _ensure_policy(iot, "p", self.DOC)
        assert iot.names() == ["get_policy", "list_policy_versions", "create_policy_version"]
        (_, kw), *_ = [c for c in iot.calls if c[0] == "create_policy_version"]
        assert kw["setAsDefault"] is True
        assert json.loads(kw["policyDocument"]) == self.DOC

    def test_at_the_cap_the_oldest_non_default_version_is_evicted_first(self):
        iot = _Iot(existing_policy={"Version": "2012-10-17", "Statement": []}, versions=5)
        _ensure_policy(iot, "p", self.DOC)
        assert iot.names() == ["get_policy", "list_policy_versions", "delete_policy_version", "create_policy_version"]
        (_, kw), *_ = [c for c in iot.calls if c[0] == "delete_policy_version"]
        assert kw["policyVersionId"] == "1"

    def test_provision_robot_updates_a_stale_account_policy(self, iot, tmp_path):
        iot.existing_policy = {"Version": "2012-10-17", "Statement": []}
        provision_robot("so101-b", cert_dir=tmp_path)
        assert "create_policy_version" in iot.names()


class _RotatingIot(_Iot):
    """An account with one Thing that already holds an old-style certificate on one policy."""

    OLD = "arn:aws:iot:us-west-2:1:cert/old0000000000"

    def __init__(self, fail_deactivate: bool = False) -> None:
        super().__init__()
        self.fail_deactivate = fail_deactivate
        self.principals = [self.OLD]
        self.attached: dict[str, list[str]] = {self.OLD: ["strands-robot"]}

    def _answer(self, name: str, kw: dict[str, Any]) -> Any:
        if name == "describe_thing":
            return {"thingArn": f"arn:aws:iot:us-west-2:1:thing/{kw['thingName']}"}
        if name == "list_thing_principals":
            return {"principals": list(self.principals)}
        if name == "list_attached_policies":
            return {"policies": [{"policyName": p} for p in self.attached.get(kw["target"], [])]}
        if name == "attach_policy":
            self.attached.setdefault(kw["target"], []).append(kw["policyName"])
            return {}
        if name == "attach_thing_principal":
            self.principals.append(kw["principal"])
            return {}
        if name == "detach_thing_principal":
            self.principals.remove(kw["principal"])
            return {}
        if name == "update_certificate":
            if self.fail_deactivate:
                raise RuntimeError("DeleteConflictException: certificate is attached elsewhere")
            return {}
        if name == "delete_certificate":
            return {}
        return super()._answer(name, kw)


@pytest.fixture
def rotating(monkeypatch):
    def _make(fail_deactivate: bool = False) -> _RotatingIot:
        client = _RotatingIot(fail_deactivate=fail_deactivate)
        monkeypatch.setattr(
            prov, "_require_boto3", lambda: type("B", (), {"client": staticmethod(lambda *a, **kw: client)})
        )
        monkeypatch.setattr(prov, "_ensure_ca", lambda ca_path: ca_path.write_text("ca"))
        return client

    return _make


class TestReprovisionThing:
    """``reprovision_thing`` rotates the certificate and keeps everything else."""

    def test_new_certificate_attached_and_activated_before_the_old_one_is_removed(self, rotating, tmp_path):
        iot = rotating()
        result = prov.reprovision_thing("so101-r", cert_dir=tmp_path)
        names = iot.names()
        assert "create_certificate_from_csr" in names
        assert names.index("attach_thing_principal") < names.index("detach_thing_principal")
        assert names.index("attach_policy") < names.index("update_certificate")
        # The old certificate is gone, the new one carries the same policy.
        assert iot.principals == ["arn:aws:iot:us-west-2:1:cert/abc"]
        assert iot.attached["arn:aws:iot:us-west-2:1:cert/abc"] == ["strands-robot"]
        assert result.policy_name == "strands-robot" and result.subject_cn == "so101-r"
        assert result.stale_certificates == ()
        # Nothing about the Thing itself is touched.
        assert "create_thing" not in names and "update_thing" not in names and "create_policy" not in names

    def test_a_missing_thing_is_refused_before_anything_is_issued(self, rotating, tmp_path):
        iot = rotating()
        original = iot._answer

        def _answer(name: str, kw: dict[str, Any]) -> Any:
            if name == "describe_thing":
                raise _NotFound()
            return original(name, kw)

        iot._answer = _answer  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="does not exist"):
            prov.reprovision_thing("ghost", cert_dir=tmp_path)
        assert "create_certificate_from_csr" not in iot.names()

    def test_a_failed_deactivation_is_a_warning_with_the_command_and_is_reported(self, rotating, tmp_path, caplog):
        rotating(fail_deactivate=True)
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.iot.provision"):
            result = prov.reprovision_thing("so101-r", cert_dir=tmp_path)
        assert result.stale_certificates == ("old0000000000",)
        (w,) = [r for r in caplog.records if "still attached and active" in r.getMessage()]
        assert "aws iot update-certificate --certificate-id old0000000000 --new-status INACTIVE" in w.getMessage()
        assert "DeleteConflictException" in w.getMessage()

    def test_provision_robot_reports_stale_certificates_too(self, rotating, tmp_path):
        iot = rotating(fail_deactivate=True)
        original = iot._answer

        def _answer(name: str, kw: dict[str, Any]) -> Any:
            if name == "describe_thing":
                raise _NotFound()  # provision_robot creates the Thing
            return original(name, kw)

        iot._answer = _answer  # type: ignore[method-assign]
        result = prov.provision_robot("so101-r", cert_dir=tmp_path)
        assert result.stale_certificates == ("old0000000000",)


class TestIotCli:
    """``python -m strands_robots iot <verb>``."""

    def test_reprovision_prints_the_identity_the_restart_note_and_the_exports(self, rotating, tmp_path, capsys):
        from strands_robots.mesh.iot.cli import main

        rotating()
        rc = main(["reprovision", "so101-r", "--region", "us-west-2", "--cert-dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "CN=so101-r" in out and "policy strands-robot" in out
        assert "restart the peer" in out
        assert "export STRANDS_IOT_THING_NAME=so101-r" in out and "export STRANDS_MESH_BACKEND=iot" in out

    def test_a_missing_thing_exits_1_with_the_reason(self, rotating, tmp_path, capsys):
        from strands_robots.mesh.iot.cli import main

        iot = rotating()
        original = iot._answer

        def _answer(name: str, kw: dict[str, Any]) -> Any:
            if name == "describe_thing":
                raise _NotFound()
            return original(name, kw)

        iot._answer = _answer  # type: ignore[method-assign]
        rc = main(["reprovision", "ghost", "--cert-dir", str(tmp_path)])
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err

    def test_stale_certificates_are_printed_to_stderr(self, rotating, tmp_path, capsys):
        from strands_robots.mesh.iot.cli import main

        rotating(fail_deactivate=True)
        rc = main(["reprovision", "so101-r", "--cert-dir", str(tmp_path)])
        assert rc == 0
        assert "old0000000000" in capsys.readouterr().err

    def test_the_dispatcher_carries_the_command(self):
        from strands_robots.__main__ import _COMMANDS

        assert "iot" in _COMMANDS

    def test_usage_error_exits_2(self):
        from strands_robots.mesh.iot.cli import main

        with pytest.raises(SystemExit) as exc:
            main(["frobnicate", "x"])
        assert exc.value.code == 2
