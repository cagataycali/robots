"""Tests for :mod:`strands_robots.mesh._acl_config`.

The ACL semantics validated here against a live Zenoh session live in
``test_redteam_zenoh.py::TestACLEnforcement``. This file covers only
the static shape of the dict the builder emits and the JSON5-lite
loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_robots.mesh import _acl_config as ac


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("STRANDS_MESH_ACL_FILE", raising=False)


# --- default ACL --------------------------------------------------------


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


# --- ACL file loader ----------------------------------------------------


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


# --- JSON5 preprocessor tests ------------------------------------------


class TestJSON5EndToEnd:
    """End-to-end test loading the shipped example file."""

    def test_example_file_loads_and_parses(self):
        # Load the canonical template `examples/mesh_acl_example.json5`
        # and verify all JSON5 features (comments, trailing commas,
        # unquoted keys) are correctly preprocessed.
        example_path = Path(__file__).resolve().parents[2] / "examples" / "mesh_acl_example.json5"
        assert example_path.is_file(), f"Example file not found at {example_path}"
        raw = example_path.read_text(encoding="utf-8")
        processed = ac._json5_to_json(raw)
        parsed = json.loads(processed)

        # Verify top-level structure
        assert isinstance(parsed, dict)
        assert parsed["enabled"] is True
        assert parsed["default_permission"] == "deny"
        assert "rules" in parsed
        assert "subjects" in parsed
        assert "policies" in parsed

        # Verify rules were parsed (not just comments)
        assert len(parsed["rules"]) > 0
        rule_ids = {r["id"] for r in parsed["rules"]}
        assert "robot_publish_telemetry" in rule_ids
        assert "operator_publish_cmds" in rule_ids
        assert "any_subscribe" in rule_ids

        # Verify subjects were parsed
        assert len(parsed["subjects"]) > 0
        subject_ids = {s["id"] for s in parsed["subjects"]}
        assert "robot_peer" in subject_ids
        assert "operator_peer" in subject_ids

        # Verify nested arrays with trailing commas parsed correctly
        robot_rule = next(r for r in parsed["rules"] if r["id"] == "robot_publish_telemetry")
        assert "**/presence" in robot_rule["key_exprs"]
        assert "**/response/**" in robot_rule["key_exprs"]

        # Verify inline comments didn't break cert_common_names
        robot_subj = next(s for s in parsed["subjects"] if s["id"] == "robot_peer")
        assert "robot-a" in robot_subj["cert_common_names"]
        assert "robot-b" in robot_subj["cert_common_names"]


class TestStripJSON5Comments:
    """Parametrised tests for ``_strip_json5_comments`` edge cases."""

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            # `//` inside a double-quoted string must roundtrip unchanged
            ('{"url": "http://example.com"}', '{"url": "http://example.com"}'),
            # `/* */` inside a string is NOT a comment
            ('{"key": "a /* not a comment */ b"}', '{"key": "a /* not a comment */ b"}'),
            # Single-quoted string with `//` must preserve it
            ("{key: 'http://x'}", "{key: 'http://x'}"),
            # Escaped quote inside string must roundtrip
            ('{"k": "he said \\"hi\\""}', '{"k": "he said \\"hi\\""}'),
            # Block comment spanning multiple lines
            ("{\n/* foo\n bar */\nkey: 1}", "{\n\nkey: 1}"),
            # Line comment at end of line
            ('{"a": 1} // comment', '{"a": 1} '),
            # Multiple line comments
            ('{\n// first\n"x": 1,\n// second\n"y": 2\n}', '{\n\n"x": 1,\n\n"y": 2\n}'),
            # Block comment in middle of object
            ('{"a": 1, /* ignored */ "b": 2}', '{"a": 1,  "b": 2}'),
        ],
    )
    def test_strip_comments_edge_cases(self, input_str, expected):
        assert ac._strip_json5_comments(input_str) == expected

    def test_block_comment_spanning_lines_parses_correctly(self):
        # Verify block comment removal + quote_keys + json.loads roundtrip
        raw = "{\n/* foo\n bar */\nkey: 1}"
        stripped = ac._strip_json5_comments(raw)
        quoted = ac._quote_unquoted_keys(stripped)
        parsed = json.loads(quoted)
        assert parsed == {"key": 1}


class TestStripTrailingCommas:
    """Parametrised tests for ``_strip_trailing_commas`` edge cases."""

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            # Array with trailing comma
            ("[1, 2, 3,]", "[1, 2, 3]"),
            # Object with trailing comma
            ('{"a": 1, "b": 2,}', '{"a": 1, "b": 2}'),
            # Comma inside string adjacent to `]` must be preserved
            ('{"x": "foo,"}', '{"x": "foo,"}'),
            # Nested trailing commas
            ("[1,[2,3,],4,]", "[1,[2,3],4]"),
            # Trailing comma before } with whitespace
            ('{"a": 1,  \n }', '{"a": 1  \n }'),
            # No trailing comma - should be unchanged
            ('{"a": 1, "b": 2}', '{"a": 1, "b": 2}'),
            # Empty array/object should be unchanged
            ("[]", "[]"),
            ("{}", "{}"),
            # Comma in string followed by closing bracket outside string
            ('["foo,", 1,]', '["foo,", 1]'),
        ],
    )
    def test_strip_trailing_commas_edge_cases(self, input_str, expected):
        assert ac._strip_trailing_commas(input_str) == expected

    def test_nested_trailing_commas_parse_correctly(self):
        raw = "[1,[2,3,],4,]"
        stripped = ac._strip_trailing_commas(raw)
        parsed = json.loads(stripped)
        assert parsed == [1, [2, 3], 4]


class TestQuoteUnquotedKeys:
    """Parametrised tests for ``_quote_unquoted_keys`` edge cases."""

    @pytest.mark.parametrize(
        "input_str,expected_parsed",
        [
            # Simple unquoted key
            ("{ enabled: true }", {"enabled": True}),
            # Mix of quoted and unquoted keys
            ('{ "already_quoted": 1, unquoted: 2 }', {"already_quoted": 1, "unquoted": 2}),
            # Key with underscore and digits
            ('{ rule_1: "x" }', {"rule_1": "x"}),
            # Multiple unquoted keys
            ("{ a: 1, b: 2, c: 3 }", {"a": 1, "b": 2, "c": 3}),
            # Nested objects with unquoted keys
            ("{ outer: { inner: 42 } }", {"outer": {"inner": 42}}),
            # Unquoted key with array value
            ("{ items: [1, 2, 3] }", {"items": [1, 2, 3]}),
            # Key starting with underscore
            ('{ _private: "yes" }', {"_private": "yes"}),
            # Key with multiple underscores and numbers
            ("{ rule_123_test: true }", {"rule_123_test": True}),
        ],
    )
    def test_quote_unquoted_keys_edge_cases(self, input_str, expected_parsed):
        quoted = ac._quote_unquoted_keys(input_str)
        parsed = json.loads(quoted)
        assert parsed == expected_parsed

    def test_unquoted_keys_in_shipped_example(self):
        # Verify the example file's unquoted keys are correctly handled
        raw = '{ enabled: true, default_permission: "deny", rules: [] }'
        quoted = ac._quote_unquoted_keys(raw)
        parsed = json.loads(quoted)
        assert parsed == {"enabled": True, "default_permission": "deny", "rules": []}


# --- Default ACL permissive-shape verification -------------------------


class TestDefaultACLPermissiveShape:
    """Verifies the default ACL is genuinely permissive end-to-end.

    Reviewer Review comment: "No live-session test for the
    default (permissive) ACL. ... Given the PR's own quirks list,
    the assertion that the default ACL is actually permissive end-to-end
    relies on author-side reasoning rather than a passing test."

    This class covers each documented quirk that could silently break
    the permissive behaviour, short of spinning up a real Zenoh session.
    """

    def test_enabled_is_true(self):
        # Redundant with TestDefaultACL but included for narrative:
        # without ``enabled: true`` Zenoh silently no-ops the entire ACL.
        acl = ac.default_acl("strands")
        assert acl["enabled"] is True

    def test_default_permission_is_deny(self):
        # Zenoh idiom is deny-by-default with explicit allow rules.
        # The permissiveness comes from the ``**`` wildcard rules,
        # NOT from a permissive ``default_permission``.
        acl = ac.default_acl("strands")
        assert acl["default_permission"] == "deny"

    def test_every_subject_has_non_empty_interfaces(self):
        # Zenoh 1.x quirk #3: subjects with empty or missing
        # ``interfaces`` match nothing. The default ACL must populate
        # the list for every subject.
        acl = ac.default_acl("strands")
        assert "subjects" in acl
        assert len(acl["subjects"]) > 0, "Default ACL must define at least one subject"

        for subj in acl["subjects"]:
            assert "interfaces" in subj, f"Subject {subj['id']} missing 'interfaces' field"
            assert isinstance(subj["interfaces"], list), f"Subject {subj['id']} interfaces is not a list"
            assert len(subj["interfaces"]) > 0, (
                f"Subject {subj['id']} has empty interfaces list -- "
                "this causes the subject to match nothing in Zenoh 1.x"
            )

    def test_at_least_one_subject_omits_cert_common_names(self):
        # Zenoh 1.x quirk #2: ``cert_common_names`` matches LITERAL CNs
        # only; globs/regexes match nothing. The permissive default must
        # have at least one subject that does NOT specify the field
        # (omitting it means match-any-CN). Verify the field is genuinely
        # absent, not present-but-empty (an empty list means match-none).
        acl = ac.default_acl("strands")
        found_permissive = False

        for subj in acl["subjects"]:
            if "cert_common_names" not in subj:
                found_permissive = True
                break
            # If present but empty, that's match-none -- silently denies everything
            if "cert_common_names" in subj and not subj["cert_common_names"]:
                pytest.fail(
                    f"Subject {subj['id']} has empty cert_common_names list -- "
                    "this matches no peers and silently denies everything. "
                    "The field must be ABSENT for match-any-CN, not present-but-empty."
                )

        assert found_permissive, (
            "No subject omits 'cert_common_names'. A permissive default must "
            "have at least one subject without cert_common_names (meaning match-any-CN)."
        )

    def test_policies_reference_valid_subjects(self):
        # Cross-check: every subject ID in ``policies[*].subjects`` must
        # reference an actual subject from ``subjects[*].id``. Dangling
        # references silently fail in Zenoh.
        acl = ac.default_acl("strands")
        subject_ids = {s["id"] for s in acl["subjects"]}

        for policy in acl["policies"]:
            for subj_ref in policy["subjects"]:
                assert subj_ref in subject_ids, (
                    f"Policy references unknown subject '{subj_ref}'. Valid subjects: {subject_ids}"
                )

    def test_policies_reference_valid_rules(self):
        # Cross-check: every rule ID in ``policies[*].rules`` must
        # reference an actual rule from ``rules[*].id``.
        acl = ac.default_acl("strands")
        rule_ids = {r["id"] for r in acl["rules"]}

        for policy in acl["policies"]:
            for rule_ref in policy["rules"]:
                assert rule_ref in rule_ids, f"Policy references unknown rule '{rule_ref}'. Valid rules: {rule_ids}"

    def test_policies_include_any_subscribe_rule(self):
        # Verify the default allows subscribing (declare_subscriber).
        acl = ac.default_acl("strands")
        all_rule_refs = [rule_ref for policy in acl["policies"] for rule_ref in policy["rules"]]
        assert "any_subscribe" in all_rule_refs, (
            "Default ACL policies must reference 'any_subscribe' rule for subscribers to work"
        )

    def test_policies_include_any_publish_rule(self):
        # Verify the default allows publishing (put).
        acl = ac.default_acl("strands")
        all_rule_refs = [rule_ref for policy in acl["policies"] for rule_ref in policy["rules"]]
        assert "any_publish" in all_rule_refs, (
            "Default ACL policies must reference 'any_publish' rule for publishers to work"
        )

    def test_load_acl_file_round_trip_with_example(self, tmp_path):
        # Another review concern: "the loader never being tested against the
        # shipped example." Load ``examples/mesh_acl_example.json5`` and
        # verify ``_load_acl_file`` parses it without raising.
        example_src = Path(__file__).resolve().parents[2] / "examples" / "mesh_acl_example.json5"
        assert example_src.is_file(), f"Example file not found at {example_src}"

        # Copy to tmp_path so we can use _load_acl_file (which requires a Path)
        tmp_example = tmp_path / "mesh_acl_example.json5"
        tmp_example.write_text(example_src.read_text(encoding="utf-8"), encoding="utf-8")

        # Should not raise
        loaded = ac._load_acl_file(tmp_example)
        assert loaded["enabled"] is True
        assert loaded["default_permission"] == "deny"
        assert len(loaded["rules"]) > 0
        assert len(loaded["subjects"]) > 0
        assert len(loaded["policies"]) > 0


class TestIsDefaultACLInUse:
    """Review feedback: operators forgetting STRANDS_MESH_ACL_FILE should
    get a runtime signal. is_default_acl_in_use() is the predicate the
    session-open WARNING calls."""

    def test_unset_returns_true(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_ACL_FILE", raising=False)
        from strands_robots.mesh import _acl_config as ac

        assert ac.is_default_acl_in_use() is True

    def test_empty_returns_true(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", "")
        from strands_robots.mesh import _acl_config as ac

        assert ac.is_default_acl_in_use() is True

    def test_whitespace_only_returns_true(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", "   ")
        from strands_robots.mesh import _acl_config as ac

        assert ac.is_default_acl_in_use() is True

    def test_set_to_path_returns_false(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", "/etc/mesh-acl.json5")
        from strands_robots.mesh import _acl_config as ac

        assert ac.is_default_acl_in_use() is False
