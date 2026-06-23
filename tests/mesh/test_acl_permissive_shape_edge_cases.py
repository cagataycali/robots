"""Edge-case coverage for ``_is_permissive_acl_shape`` non-permissive verdicts.

The permissive-shape detector (``strands_robots.mesh._acl_config``) gates
``Mesh.start``: a True verdict means the wire-effective ACL grants any
CA-signed peer publish/subscribe on any key, and the mesh refuses to start.
A False verdict means the posture is genuinely scoped. These tests pin the
False (safe) verdicts for the malformed / non-matching shapes the detector
must NOT flag as permissive, so a future refactor that accidentally widens
the wildcard match (and lets a scoped ACL be treated as permissive, or vice
versa) fails here.

Behavior under test, not implementation: each case asserts the boolean
verdict for a hand-rolled resolved-ACL dict.
"""

from __future__ import annotations

from strands_robots.mesh import _acl_config


def _deny_base() -> dict:
    """A minimal, genuinely-scoped deny-default ACL skeleton."""
    return {
        "enabled": True,
        "default_permission": "deny",
        "rules": [],
        "subjects": [],
        "policies": [],
    }


def test_non_dict_input_is_not_permissive() -> None:
    """A non-dict resolved ACL (e.g. a list slipped through) is never
    treated as permissive -- the detector fails safe to False."""
    assert _acl_config._is_permissive_acl_shape([]) is False  # type: ignore[arg-type]
    assert _acl_config._is_permissive_acl_shape("nope") is False  # type: ignore[arg-type]


def test_allow_default_with_explicit_rules_is_not_pattern1() -> None:
    """``default_permission: allow`` only matches pattern 1 when every
    explicit collection is empty. With a populated rule it is NOT the
    built-in permissive shape, and (being neither 'deny') falls through
    to a False verdict."""
    data = _deny_base()
    data["default_permission"] = "allow"
    data["rules"] = [
        {"id": "r1", "key_exprs": ["scoped/**"], "messages": ["put"], "flows": ["egress"], "permission": "allow"}
    ]
    assert _acl_config._is_permissive_acl_shape(data) is False


def test_unknown_default_permission_is_not_permissive() -> None:
    """An unrecognised ``default_permission`` literal is neither pattern 1
    (allow+empty) nor pattern 2 (deny), so the verdict is False."""
    data = _deny_base()
    data["default_permission"] = "audit-only"
    assert _acl_config._is_permissive_acl_shape(data) is False


def test_deny_with_non_list_rules_is_not_permissive() -> None:
    """Pattern 2 requires list-typed rules/subjects; a malformed non-list
    rules field fails safe to False rather than raising."""
    data = _deny_base()
    data["rules"] = {"oops": "not-a-list"}
    assert _acl_config._is_permissive_acl_shape(data) is False


def test_deny_with_non_list_policies_is_not_permissive() -> None:
    """Pattern 2 also requires list-typed policies; a malformed non-list
    policies field fails safe to False."""
    data = _deny_base()
    data["policies"] = {"oops": "not-a-list"}
    assert _acl_config._is_permissive_acl_shape(data) is False


def test_deny_wildcard_rule_without_wildcard_subject_is_scoped() -> None:
    """A deny default with a ``**``/allow rule but every subject pinned to
    a cert common name is genuinely scoped -- only named peers match, so
    the verdict is False."""
    data = _deny_base()
    data["rules"] = [
        {"id": "open", "key_exprs": ["**"], "messages": ["put"], "flows": ["egress"], "permission": "allow"}
    ]
    data["subjects"] = [{"id": "ops", "cert_common_names": ["op-1"]}]
    data["policies"] = [{"rules": ["open"], "subjects": ["ops"]}]
    assert _acl_config._is_permissive_acl_shape(data) is False


def test_deny_wildcard_subject_without_wildcard_rule_is_scoped() -> None:
    """A wildcard subject (no interfaces, no CNs) tied only to a narrow,
    non-``**`` rule is scoped: the rule does not open everything, so the
    verdict is False."""
    data = _deny_base()
    data["rules"] = [
        {"id": "narrow", "key_exprs": ["telemetry/**"], "messages": ["put"], "flows": ["egress"], "permission": "allow"}
    ]
    data["subjects"] = [{"id": "any"}]
    data["policies"] = [{"rules": ["narrow"], "subjects": ["any"]}]
    assert _acl_config._is_permissive_acl_shape(data) is False


def test_deny_wildcard_rule_is_deny_permission_is_scoped() -> None:
    """A ``**`` rule with ``permission: deny`` is a blanket *block*, not an
    open grant, so it is not a wildcard-allow rule and the verdict is
    False even with a wildcard subject."""
    data = _deny_base()
    data["rules"] = [
        {"id": "blockall", "key_exprs": ["**"], "messages": ["put"], "flows": ["egress"], "permission": "deny"}
    ]
    data["subjects"] = [{"id": "any"}]
    data["policies"] = [{"rules": ["blockall"], "subjects": ["any"]}]
    assert _acl_config._is_permissive_acl_shape(data) is False


def test_deny_wildcard_rule_and_subject_unlinked_by_policy_is_scoped() -> None:
    """The dangerous pattern only fires when a SINGLE policy ties a
    wildcard rule to a wildcard subject. If separate policies keep them
    apart (wildcard rule -> named subject; wildcard subject -> narrow
    rule), no policy grants any-peer-everywhere, so the verdict is False."""
    data = _deny_base()
    data["rules"] = [
        {"id": "open", "key_exprs": ["**"], "messages": ["put"], "flows": ["egress"], "permission": "allow"},
        {"id": "narrow", "key_exprs": ["t/**"], "messages": ["put"], "flows": ["egress"], "permission": "allow"},
    ]
    data["subjects"] = [
        {"id": "named", "cert_common_names": ["op-1"]},
        {"id": "any"},
    ]
    data["policies"] = [
        {"rules": ["open"], "subjects": ["named"]},
        {"rules": ["narrow"], "subjects": ["any"]},
    ]
    assert _acl_config._is_permissive_acl_shape(data) is False


def test_deny_non_dict_policy_entry_is_skipped_not_permissive() -> None:
    """A malformed non-dict entry in the policies list is skipped during
    the cross-link scan; with no valid policy linking wildcard rule to
    wildcard subject the verdict stays False."""
    data = _deny_base()
    data["rules"] = [
        {"id": "open", "key_exprs": ["**"], "messages": ["put"], "flows": ["egress"], "permission": "allow"}
    ]
    data["subjects"] = [{"id": "any"}]
    data["policies"] = ["not-a-dict", 42]
    assert _acl_config._is_permissive_acl_shape(data) is False


def test_deny_wildcard_rule_and_subject_linked_by_policy_is_permissive() -> None:
    """Positive control: deny default + ``**``/allow rule + wildcard
    subject tied together by one policy is wire-effectively
    any-peer-everywhere, so the detector flags it True."""
    data = _deny_base()
    data["rules"] = [
        {"id": "open", "key_exprs": ["**"], "messages": ["put"], "flows": ["egress"], "permission": "allow"}
    ]
    data["subjects"] = [{"id": "any"}]
    data["policies"] = [{"rules": ["open"], "subjects": ["any"]}]
    assert _acl_config._is_permissive_acl_shape(data) is True


def test_deny_non_dict_rule_entry_is_skipped_not_permissive() -> None:
    """A malformed non-dict entry in the rules list is not a wildcard rule;
    with no real wildcard-allow rule present the verdict is False."""
    data = _deny_base()
    data["rules"] = ["not-a-dict", 7]
    data["subjects"] = [{"id": "any"}]
    data["policies"] = [{"rules": ["open"], "subjects": ["any"]}]
    assert _acl_config._is_permissive_acl_shape(data) is False


def test_deny_non_dict_subject_entry_is_skipped_not_permissive() -> None:
    """A malformed non-dict entry in the subjects list is not a wildcard
    subject; with a real wildcard rule but no wildcard subject the verdict
    is False."""
    data = _deny_base()
    data["rules"] = [
        {"id": "open", "key_exprs": ["**"], "messages": ["put"], "flows": ["egress"], "permission": "allow"}
    ]
    data["subjects"] = ["not-a-dict", 9]
    data["policies"] = [{"rules": ["open"], "subjects": ["any"]}]
    assert _acl_config._is_permissive_acl_shape(data) is False
