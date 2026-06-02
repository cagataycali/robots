"""Issue #218: ACL TOCTOU window between is_default_acl_in_use and resolve_acl.

The fix introduces ``snapshot_acl(namespace)`` which atomically returns
``(is_permissive, resolved_dict)`` from a single file read, plus
``acl_block_from(resolved)`` which builds the wire config block from
the snapshot dict. ``session.py`` is updated to use the snapshot pattern
so the refuse-to-start gate and the wire config builder share exactly
ONE ACL file read per ``Mesh.start()``.
"""

from __future__ import annotations


def test_snapshot_acl_returns_permissive_for_default():
    """No env var -> built-in default -> permissive=True, resolved=default_acl."""
    from strands_robots.mesh import _acl_config

    is_permissive, resolved = _acl_config.snapshot_acl("strands")
    assert is_permissive is True
    assert resolved == _acl_config.default_acl("strands")


def test_snapshot_acl_returns_non_permissive_for_role_separated_file(tmp_path, monkeypatch):
    """Operator-supplied role-separated ACL -> permissive=False, resolved=loaded dict."""
    import json

    from strands_robots.mesh import _acl_config

    acl_file = tmp_path / "acl.json5"
    acl_file.write_text(
        json.dumps(
            {
                "enabled": True,
                "default_permission": "deny",
                "rules": [
                    {
                        "id": "operator",
                        "permission": "allow",
                        "flows": ["egress"],
                        "messages": ["put"],
                        "key_exprs": ["strands/safety/estop"],
                    }
                ],
                "subjects": [{"id": "op", "cert_common_names": ["operator-1"]}],
                "policies": [{"rules": ["operator"], "subjects": ["op"]}],
            }
        )
    )

    monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(acl_file))
    is_permissive, resolved = _acl_config.snapshot_acl("strands")
    assert is_permissive is False
    assert resolved.get("default_permission") == "deny"


def test_snapshot_acl_single_file_read(tmp_path, monkeypatch):
    """Issue #218 core invariant: snapshot_acl reads the file ONCE.

    The previous two-call pattern (is_default_acl_in_use + resolve_acl)
    invalidated the identity-tuple cache and re-read the file. Pinning
    that snapshot_acl performs at most one _load_acl_file call.
    """
    import json

    from strands_robots.mesh import _acl_config

    acl_file = tmp_path / "acl.json5"
    acl_file.write_text(json.dumps({"enabled": True, "default_permission": "deny"}))

    monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(acl_file))
    # Clear the cache so we get a fresh read
    _acl_config._load_acl_cached.cache_clear() if hasattr(_acl_config._load_acl_cached, "cache_clear") else None

    call_count = [0]
    real_load_acl_file = _acl_config._load_acl_file

    def counted(path):
        call_count[0] += 1
        return real_load_acl_file(path)

    monkeypatch.setattr(_acl_config, "_load_acl_file", counted)
    # Also clear cache to ensure miss
    if hasattr(_acl_config._load_acl_cached, "cache_clear"):
        _acl_config._load_acl_cached.cache_clear()

    is_permissive, resolved = _acl_config.snapshot_acl("strands")
    # Sanity-check the return shape so the test fails loudly if the
    # signature changes (rather than silently passing on a refactor).
    assert isinstance(is_permissive, bool)
    assert isinstance(resolved, dict)
    # Core invariant: snapshot_acl performs at most ONE _load_acl_file call
    assert call_count[0] <= 1, f"snapshot_acl called _load_acl_file {call_count[0]} times; must be <= 1"


def test_acl_block_from_uses_provided_dict():
    """acl_block_from doesn't re-read the file -- it serialises the given dict."""
    import json

    from strands_robots.mesh import _acl_config

    custom = {"enabled": True, "default_permission": "deny", "marker": "from-snapshot"}
    key, value = _acl_config.acl_block_from(custom)
    assert key == "access_control"
    assert json.loads(value) == custom
    assert json.loads(value)["marker"] == "from-snapshot"
