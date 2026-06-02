"""Pin test for wildcard-subject detection in ACL validator.

Review thread PRRT_kwDORUMiZs6GTwcv flagged that a subject with neither
``interfaces`` nor ``cert_common_names`` silently matches every peer on
every link -- effectively a wildcard subject that an operator may not
have intended. Combined with ``default_permission: "deny"`` and an
``allow`` rule, this produces an effectively-permissive ACL without
operator awareness.

Pin: the validator now emits a WARNING when a subject has only an ``id``
and no constraining fields. Pre-fix HEAD silently accepts the
wide-open subject with no diagnostic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from strands_robots.mesh import _acl_config


def _minimal_acl(subjects: list[dict]) -> dict:
    """Build a minimal valid ACL structure with the given subjects."""
    return {
        "enabled": True,
        "default_permission": "deny",
        "subjects": subjects,
        "rules": [
            {
                "id": "allow-all",
                "key_exprs": ["**"],
                "messages": ["put"],
                "flows": ["ingress"],
                "permission": "allow",
            }
        ],
        "policies": [
            {
                "rules": ["allow-all"],
                "subjects": [s["id"] for s in subjects],
            }
        ],
    }


def _write(tmp_path: Path, doc: dict) -> Path:
    p = tmp_path / "acl.json5"
    p.write_text(json.dumps(doc))
    return p


class TestWildcardSubjectWarning:
    """Validate that subjects with neither interfaces nor cert_common_names
    emit a WARNING during shape validation."""

    def test_subject_without_interfaces_or_cns_warns(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A subject with only an ``id`` triggers a wildcard warning."""
        doc = _minimal_acl([{"id": "wide-open"}])
        path = _write(tmp_path, doc)

        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh._acl_config"):
            _acl_config._validate_acl_shape(doc, path)

        assert any("wide-open" in msg and "matches every peer" in msg for msg in caplog.messages), (
            f"Expected wildcard-subject warning for 'wide-open', got: {caplog.messages}"
        )

    def test_subject_with_interfaces_only_does_not_warn(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A subject with ``interfaces`` set does not trigger the warning."""
        doc = _minimal_acl([{"id": "nic-bound", "interfaces": ["eth0"]}])
        path = _write(tmp_path, doc)

        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh._acl_config"):
            _acl_config._validate_acl_shape(doc, path)

        assert not any("matches every peer" in msg for msg in caplog.messages), (
            f"Unexpected wildcard warning: {caplog.messages}"
        )

    def test_subject_with_cns_only_does_not_warn(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A subject with ``cert_common_names`` set does not trigger the warning."""
        doc = _minimal_acl([{"id": "cn-bound", "cert_common_names": ["robot-1"]}])
        path = _write(tmp_path, doc)

        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh._acl_config"):
            _acl_config._validate_acl_shape(doc, path)

        assert not any("matches every peer" in msg for msg in caplog.messages), (
            f"Unexpected wildcard warning: {caplog.messages}"
        )

    def test_subject_with_both_does_not_warn(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A subject with both fields set does not trigger the warning."""
        doc = _minimal_acl([{"id": "fully-bound", "interfaces": ["wlan0"], "cert_common_names": ["op-1"]}])
        path = _write(tmp_path, doc)

        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh._acl_config"):
            _acl_config._validate_acl_shape(doc, path)

        assert not any("matches every peer" in msg for msg in caplog.messages), (
            f"Unexpected wildcard warning: {caplog.messages}"
        )

    def test_mixed_subjects_warns_only_for_unbounded(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Only the wildcard subject triggers the warning, not the bounded one."""
        doc = _minimal_acl(
            [
                {"id": "bounded", "cert_common_names": ["robot-1"]},
                {"id": "unbounded"},
            ]
        )
        # Fix policies to reference both subjects
        doc["policies"][0]["subjects"] = ["bounded", "unbounded"]
        path = _write(tmp_path, doc)

        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh._acl_config"):
            _acl_config._validate_acl_shape(doc, path)

        warning_msgs = [m for m in caplog.messages if "matches every peer" in m]
        assert len(warning_msgs) == 1
        assert "unbounded" in warning_msgs[0]
        assert "'bounded'" not in warning_msgs[0]
