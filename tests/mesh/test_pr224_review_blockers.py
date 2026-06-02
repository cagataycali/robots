"""Pin tests for PR #224 review-blocker batch (2026-06-02 sweep).

Covers:

* Thread session.py:296 -- ACL TOCTOU single-flight via thread-local snapshot.
* Thread _acl_config.py:456 -- ``_is_permissive_acl_shape`` recognises the
  ``default_permission: "deny"`` + wildcard-rule + wildcard-subject pattern
  (was wire-effectively permissive but bypassed the gate).
* Thread _acl_config.py:429 -- cache returns deep copy on hit
  (caller mutation does not poison the cache).
* Thread _acl_config.py:279 -- subjects with empty-list cert_common_names
  now also trigger the wildcard warning (was None-only).
* Thread core.py:121 -- dead ``except ImportError`` /
  ``hasattr(_acl_config, "snapshot_acl")`` fallback removed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_thread_local_single_flight_returns_same_dict() -> None:
    """When a snapshot is stashed, ``snapshot_acl`` returns the SAME dict
    (closes the gate-vs-build TOCTOU window per review session.py:296)."""
    from strands_robots.mesh import _acl_config

    sentinel = {
        "enabled": True,
        "default_permission": "deny",
        "rules": [],
        "subjects": [],
        "policies": [],
        "_marker": "stashed-by-Mesh.start",
    }
    _acl_config._set_thread_snapshot(sentinel)
    try:
        is_permissive, resolved = _acl_config.snapshot_acl("strands")
        # The returned dict IS the sentinel (identity, not just equality)
        assert resolved is sentinel
        assert is_permissive is False
    finally:
        _acl_config._clear_thread_snapshot()


def test_thread_local_cleared_after_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After ``_clear_thread_snapshot``, ``snapshot_acl`` re-resolves from disk."""
    from strands_robots.mesh import _acl_config

    sentinel = {"_marker": "stashed"}
    _acl_config._set_thread_snapshot(sentinel)
    _acl_config._clear_thread_snapshot()

    # Now without env var, returns built-in default (NOT the sentinel)
    monkeypatch.delenv("STRANDS_MESH_ACL_FILE", raising=False)
    is_permissive, resolved = _acl_config.snapshot_acl("strands")
    assert resolved is not sentinel
    assert is_permissive is True  # built-in default is permissive


def test_permissive_shape_detects_deny_plus_wildcard_rule_plus_wildcard_subject() -> None:
    """The wire-effectively permissive ACL shape that bypassed the
    previous narrow check (review _acl_config.py:456):

    ``default_permission: "deny"``  (looks safe)
    + a single ``key_exprs: ["**"], permission: "allow"`` rule
    + a wildcard subject (no interfaces, no cert_common_names)
    + a policy that ties them together
    """
    from strands_robots.mesh import _acl_config

    permissive_in_disguise = {
        "enabled": True,
        "default_permission": "deny",
        "rules": [
            {
                "id": "wide-open",
                "key_exprs": ["**"],
                "messages": ["put"],
                "flows": ["ingress"],
                "permission": "allow",
            }
        ],
        "subjects": [{"id": "anyone"}],  # no constraints -> wildcard
        "policies": [{"rules": ["wide-open"], "subjects": ["anyone"]}],
    }
    assert _acl_config._is_permissive_acl_shape(permissive_in_disguise) is True


def test_permissive_shape_does_not_flag_role_separated_acl() -> None:
    """A genuinely role-separated ACL with constrained subjects is NOT
    flagged (no false positive)."""
    from strands_robots.mesh import _acl_config

    safe = {
        "enabled": True,
        "default_permission": "deny",
        "rules": [
            {
                "id": "operator",
                "key_exprs": ["strands/safety/estop"],
                "messages": ["put"],
                "flows": ["egress"],
                "permission": "allow",
            }
        ],
        "subjects": [{"id": "op", "cert_common_names": ["operator-1"]}],
        "policies": [{"rules": ["operator"], "subjects": ["op"]}],
    }
    assert _acl_config._is_permissive_acl_shape(safe) is False


def test_permissive_shape_does_not_flag_unwired_wildcard() -> None:
    """A wildcard rule and a wildcard subject that are NOT tied together
    by any policy do not match the pattern."""
    from strands_robots.mesh import _acl_config

    not_wired = {
        "enabled": True,
        "default_permission": "deny",
        "rules": [
            {
                "id": "wide-open",
                "key_exprs": ["**"],
                "permission": "allow",
            }
        ],
        "subjects": [
            {"id": "anyone"},
            {"id": "operator", "cert_common_names": ["op-1"]},
        ],
        # Only the constrained subject is wired; wildcard subject sits unused.
        "policies": [{"rules": ["wide-open"], "subjects": ["operator"]}],
    }
    assert _acl_config._is_permissive_acl_shape(not_wired) is False


def test_cache_hit_returns_deep_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache poisoning protection: caller mutation of cache result does
    not corrupt the next caller's view (review _acl_config.py:429)."""
    from strands_robots.mesh import _acl_config

    acl = tmp_path / "acl.json5"
    acl.write_text(
        json.dumps(
            {
                "enabled": True,
                "default_permission": "deny",
                "rules": [{"id": "r", "key_exprs": ["foo"], "messages": ["put"], "flows": ["ingress"], "permission": "allow"}],
                "subjects": [{"id": "s", "cert_common_names": ["c1"]}],
                "policies": [{"rules": ["r"], "subjects": ["s"]}],
            }
        )
    )
    _acl_config._clear_acl_cache_for_test()

    first = _acl_config._load_acl_cached(acl)
    # Caller deliberately mutates the returned dict (the bad pattern)
    first["enabled"] = False
    first["rules"].clear()

    # Subsequent caller must see the ORIGINAL contents -- mutation didn't poison
    second = _acl_config._load_acl_cached(acl)
    assert second["enabled"] is True
    assert len(second["rules"]) == 1
    assert second is not first  # different object identity (deep copy)


def test_wildcard_warning_fires_on_empty_cert_common_names_list(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Subject with ``cert_common_names: []`` (empty list, not None) ALSO
    triggers the wildcard warning -- the previous gate only fired on
    ``None`` (review _acl_config.py:279)."""
    import logging

    from strands_robots.mesh import _acl_config

    doc = {
        "enabled": True,
        "default_permission": "deny",
        "subjects": [{"id": "wide-open", "cert_common_names": []}],
        "rules": [{"id": "r", "key_exprs": ["**"], "messages": ["put"], "flows": ["ingress"], "permission": "allow"}],
        "policies": [{"rules": ["r"], "subjects": ["wide-open"]}],
    }
    p = tmp_path / "acl.json5"
    p.write_text(json.dumps(doc))

    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh._acl_config"):
        _acl_config._validate_acl_shape(doc, p)

    assert any("wide-open" in m and "matches every peer" in m for m in caplog.messages), (
        f"expected wildcard warning for empty cert_common_names list, got: {caplog.messages}"
    )


def test_core_no_dead_importerror_fallback() -> None:
    """Static assertion: the dead ``except ImportError`` and
    ``hasattr(_acl_config, "snapshot_acl")`` branches were removed
    (review thread core.py:121)."""
    from strands_robots.mesh import core as core_mod

    src = Path(core_mod.__file__).read_text()
    # The gate body now imports _acl_config + _zenoh_config unconditionally
    # since PR-3 ships them in the same diff.
    assert "PR-3 (`_acl_config` + `_zenoh_config`) not on the tree yet" not in src, (
        "Expected the dead-fallback comment to be deleted; PR-3 ships these modules."
    )
    # And the hasattr-based gate-skip is gone.
    assert 'hasattr(_acl_config, "snapshot_acl")' not in src
