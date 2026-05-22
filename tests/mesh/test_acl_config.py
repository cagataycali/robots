"""Tests for :mod:`strands_robots.mesh._acl_config`."""

from __future__ import annotations

import json

import pytest

from strands_robots.mesh import _acl_config as ac


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("STRANDS_MESH_ACL_FILE", raising=False)


# ─── default ACL ────────────────────────────────────────────────────────


class TestDefaultACL:
    def test_default_permission_is_deny(self):
        acl = ac.default_acl("ns")
        assert acl["default_permission"] == "deny"

    def test_robot_subject_has_cn_glob(self):
        acl = ac.default_acl("ns")
        robot = next(s for s in acl["subjects"] if s["id"] == "robot_peer")
        assert "robot-*" in robot["cert_common_names"]

    def test_operator_subject_has_cn_glob(self):
        acl = ac.default_acl("ns")
        op = next(s for s in acl["subjects"] if s["id"] == "operator_peer")
        assert "op-*" in op["cert_common_names"]

    def test_robot_can_publish_telemetry(self):
        acl = ac.default_acl("ns")
        rule = next(r for r in acl["rules"] if r["id"] == "robot_publish_telemetry")
        assert rule["messages"] == ["put"]
        assert rule["flows"] == ["egress"]
        assert rule["permission"] == "allow"
        assert any("ns/*/presence" in k for k in rule["key_exprs"])
        assert any("ns/*/state/**" in k for k in rule["key_exprs"])

    def test_robot_can_receive_cmds(self):
        acl = ac.default_acl("ns")
        rule = next(r for r in acl["rules"] if r["id"] == "robot_subscribe_cmds")
        assert "declare_subscriber" in rule["messages"]
        assert rule["flows"] == ["ingress"]
        assert "ns/*/cmd" in rule["key_exprs"]
        assert "ns/broadcast" in rule["key_exprs"]

    def test_operator_can_publish_cmds(self):
        acl = ac.default_acl("ns")
        rule = next(r for r in acl["rules"] if r["id"] == "operator_publish_cmds")
        assert "ns/*/cmd" in rule["key_exprs"]
        assert "ns/broadcast" in rule["key_exprs"]

    def test_namespace_propagates_to_all_key_exprs(self):
        acl = ac.default_acl("ns")
        for rule in acl["rules"]:
            for kx in rule["key_exprs"]:
                assert kx.startswith("ns/"), f"{kx!r} does not start with namespace"

    def test_acl_block_serialises_to_json5(self):
        path, value = ac.acl_block("ns")
        assert path == "access_control"
        decoded = json.loads(value)
        assert decoded["default_permission"] == "deny"


# ─── ACL file loader ────────────────────────────────────────────────────


class TestACLFileLoader:
    def test_resolve_uses_default_when_unset(self):
        acl = ac.resolve_acl("ns")
        # Identity: matches default_acl shape.
        assert acl["default_permission"] == "deny"
        assert {s["id"] for s in acl["subjects"]} == {"robot_peer", "operator_peer"}

    def test_resolve_loads_from_file(self, monkeypatch, tmp_path):
        custom = {
            "default_permission": "deny",
            "rules": [],
            "subjects": [{"id": "x", "cert_common_names": ["foo-*"]}],
            "policies": [],
        }
        path = tmp_path / "acl.json"
        path.write_text(json.dumps(custom))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))

        loaded = ac.resolve_acl("ns")
        assert loaded["subjects"][0]["cert_common_names"] == ["foo-*"]

    def test_missing_file_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(tmp_path / "nope.json"))
        with pytest.raises(FileNotFoundError):
            ac.resolve_acl("ns")

    def test_oversize_file_rejected(self, monkeypatch, tmp_path):
        path = tmp_path / "huge.json"
        path.write_text("x" * (ac.ACL_FILE_MAX_BYTES + 1))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with pytest.raises(ValueError, match="refusing to load"):
            ac.resolve_acl("ns")

    def test_invalid_json_rejected(self, monkeypatch, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{this is not json")
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with pytest.raises(ValueError, match="not valid JSON5"):
            ac.resolve_acl("ns")

    def test_missing_required_field_rejected(self, monkeypatch, tmp_path):
        path = tmp_path / "incomplete.json"
        path.write_text(json.dumps({"default_permission": "deny", "rules": []}))
        # Missing 'subjects' and 'policies'.
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with pytest.raises(ValueError, match="missing required field"):
            ac.resolve_acl("ns")

    def test_invalid_default_permission_rejected(self, monkeypatch, tmp_path):
        path = tmp_path / "weird.json"
        path.write_text(
            json.dumps(
                {
                    "default_permission": "maybe",
                    "rules": [],
                    "subjects": [],
                    "policies": [],
                }
            )
        )
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with pytest.raises(ValueError, match="must be 'allow' or 'deny'"):
            ac.resolve_acl("ns")

    def test_default_allow_logs_warning(self, monkeypatch, tmp_path, caplog):
        path = tmp_path / "blacklist.json"
        path.write_text(
            json.dumps(
                {
                    "default_permission": "allow",
                    "rules": [],
                    "subjects": [],
                    "policies": [],
                }
            )
        )
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with caplog.at_level("WARNING"):
            ac.resolve_acl("ns")
        assert any("blacklist" in rec.message for rec in caplog.records)
