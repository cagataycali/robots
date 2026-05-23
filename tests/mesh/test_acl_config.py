"""Tests for :mod:`strands_robots.mesh._acl_config`.

The ACL semantics validated here against a live Zenoh session live in
``test_redteam_zenoh.py::TestACLEnforcement``. This file covers only
the static shape of the dict the builder emits and the JSON5-lite
loader.
"""

from __future__ import annotations

import json

import pytest

from strands_robots.mesh import _acl_config as ac


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("STRANDS_MESH_ACL_FILE", raising=False)


# ─── default ACL ────────────────────────────────────────────────────────


class TestDefaultACL:
    def test_enabled_is_true(self):
        # Without ``enabled: true`` Zenoh silently no-ops the ACL.
        assert ac.default_acl("strands")["enabled"] is True

    def test_default_permission_is_deny(self):
        assert ac.default_acl("strands")["default_permission"] == "deny"

    def test_subject_has_non_empty_interfaces(self):
        # Zenoh 1.x bug: subjects with empty/missing ``interfaces``
        # match nothing. Defaults must populate every local NIC.
        subj = ac.default_acl("strands")["subjects"][0]
        assert isinstance(subj["interfaces"], list)
        assert len(subj["interfaces"]) > 0

    def test_subscriber_rule_uses_egress_flow(self):
        # ``declare_subscriber`` lives in egress (subscriber emits
        # the declare to the publishing peer).
        rule = next(r for r in ac.default_acl("strands")["rules"] if r["id"] == "any_subscribe")
        assert rule["messages"] == ["declare_subscriber"]
        assert rule["flows"] == ["egress"]
        assert rule["permission"] == "allow"
        assert rule["key_exprs"] == ["**"]

    def test_publish_rule_uses_double_glob(self):
        # ``f"{namespace}/*/cmd"`` would never match (Zenoh strips
        # the namespace before matching against key_exprs); ``**`` is
        # the robust glob for the permissive default.
        rule = next(r for r in ac.default_acl("strands")["rules"] if r["id"] == "any_publish")
        assert rule["messages"] == ["put"]
        assert "ingress" in rule["flows"]
        assert rule["key_exprs"] == ["**"]

    def test_acl_block_serialises_to_json(self):
        path, value = ac.acl_block("strands")
        assert path == "access_control"
        decoded = json.loads(value)
        assert decoded["enabled"] is True
        assert decoded["default_permission"] == "deny"


# ─── ACL file loader ────────────────────────────────────────────────────


class TestACLFileLoader:
    def _good_acl_dict(self) -> dict:
        return {
            "enabled": True,
            "default_permission": "deny",
            "rules": [],
            "subjects": [{"id": "x", "cert_common_names": ["foo-*"]}],
            "policies": [],
        }

    def test_resolve_uses_default_when_unset(self):
        acl = ac.resolve_acl("strands")
        assert acl["enabled"] is True
        assert acl["default_permission"] == "deny"
        assert {s["id"] for s in acl["subjects"]} == {"any_authenticated_peer"}

    def test_resolve_loads_from_file(self, monkeypatch, tmp_path):
        path = tmp_path / "acl.json"
        path.write_text(json.dumps(self._good_acl_dict()))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))

        loaded = ac.resolve_acl("strands")
        assert loaded["enabled"] is True
        assert loaded["subjects"][0]["cert_common_names"] == ["foo-*"]

    def test_missing_file_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(tmp_path / "nope.json"))
        with pytest.raises(FileNotFoundError):
            ac.resolve_acl("strands")

    def test_oversize_file_rejected(self, monkeypatch, tmp_path):
        path = tmp_path / "huge.json"
        path.write_text("x" * (ac.ACL_FILE_MAX_BYTES + 1))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with pytest.raises(ValueError, match="refusing to load"):
            ac.resolve_acl("strands")

    def test_invalid_json_rejected(self, monkeypatch, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{this is not json")
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with pytest.raises(ValueError, match="not valid JSON5"):
            ac.resolve_acl("strands")

    def test_missing_required_field_rejected(self, monkeypatch, tmp_path):
        path = tmp_path / "incomplete.json"
        path.write_text(json.dumps({"enabled": True, "default_permission": "deny", "rules": []}))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with pytest.raises(ValueError, match="missing required field"):
            ac.resolve_acl("strands")

    def test_missing_enabled_rejected(self, monkeypatch, tmp_path):
        # Missing or false ``enabled`` silently disables the ACL in
        # Zenoh; the loader fails closed.
        no_enabled = self._good_acl_dict()
        del no_enabled["enabled"]
        path = tmp_path / "no_enabled.json"
        path.write_text(json.dumps(no_enabled))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with pytest.raises(ValueError, match="enabled: true"):
            ac.resolve_acl("strands")

    def test_explicit_enabled_false_rejected(self, monkeypatch, tmp_path):
        bad = self._good_acl_dict()
        bad["enabled"] = False
        path = tmp_path / "disabled.json"
        path.write_text(json.dumps(bad))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with pytest.raises(ValueError, match="enabled: true"):
            ac.resolve_acl("strands")

    def test_invalid_default_permission_rejected(self, monkeypatch, tmp_path):
        bad = self._good_acl_dict()
        bad["default_permission"] = "maybe"
        path = tmp_path / "weird.json"
        path.write_text(json.dumps(bad))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with pytest.raises(ValueError, match="must be 'allow' or 'deny'"):
            ac.resolve_acl("strands")

    def test_default_allow_logs_warning(self, monkeypatch, tmp_path, caplog):
        bad = self._good_acl_dict()
        bad["default_permission"] = "allow"
        path = tmp_path / "blacklist.json"
        path.write_text(json.dumps(bad))
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
        with caplog.at_level("WARNING"):
            ac.resolve_acl("strands")
        assert any("blacklist" in rec.message for rec in caplog.records)
