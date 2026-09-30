"""Regression tests: a glob over the command or safety plane is as permissive as ``**``.

``_is_permissive_acl_shape`` (the input to ``Mesh.start``'s refuse-to-start
gate) recognised a wide-open ``allow`` rule only by the literal string ``**``
in its ``key_exprs``. The module's own docstring and the shipped template tell
operators to write globs instead, ``**/cmd``, ``**/broadcast``,
``**/safety/**``, which together cover every actuation-relevant topic. Bound
to a subject with no certificate constraint, such a rule let any CA-signed
peer command and stop every robot, and the gate stayed silent.

Now an ``allow`` rule that can ``put`` on any key expression reaching the
command, broadcast or safety plane (``**``, ``*``, ``$*`` and their
combinations) counts as a wide-open rule for the detector, and the loader
warns about such a rule bound to a subject that lacks ``cert_common_names``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from strands_robots.mesh import _acl_config
from strands_robots.mesh._acl_config import _is_permissive_acl_shape, _key_expr_reaches_actuation_plane


def _acl(rule_key_exprs: list[str], subject: dict[str, Any], *, messages: list[str] | None = ["put"]) -> dict[str, Any]:
    rule: dict[str, Any] = {"id": "r", "key_exprs": rule_key_exprs, "flows": ["ingress"], "permission": "allow"}
    if messages is not None:
        rule["messages"] = messages
    return {
        "enabled": True,
        "default_permission": "deny",
        "rules": [rule],
        "subjects": [subject],
        "policies": [{"rules": ["r"], "subjects": [subject["id"]]}],
    }


_ANY = {"id": "any"}
_NAMED = {"id": "ops", "cert_common_names": ["operator-1"]}


class TestGlobsOverTheActuationPlaneArePermissive:
    @pytest.mark.parametrize(
        "key_exprs",
        [
            ["**/cmd", "**/broadcast", "**/safety/**"],
            ["**/cmd"],
            ["*/cmd"],
            ["**/broadcast"],
            ["**/safety/**"],
            ["safety/*"],
            ["**/safety/estop"],
            ["$*/cmd"],
            ["**/c$*"],
            ["presence", "**/cmd"],
        ],
    )
    def test_bound_to_an_unconstrained_subject(self, key_exprs: list[str]) -> None:
        assert _is_permissive_acl_shape(_acl(key_exprs, _ANY)) is True

    def test_a_rule_with_no_messages_restriction_counts(self) -> None:
        assert _is_permissive_acl_shape(_acl(["**/cmd"], _ANY, messages=None)) is True

    @pytest.mark.parametrize("key_exprs", [["**/cmd", "**/broadcast", "**/safety/**"], ["**"]])
    def test_the_template_shape_bound_to_named_operators_is_scoped(self, key_exprs: list[str]) -> None:
        assert _is_permissive_acl_shape(_acl(key_exprs, _NAMED)) is False

    @pytest.mark.parametrize(
        "key_exprs",
        [["**/presence", "**/state/**", "**/health", "**/response/**"], ["**/lidar/**"], ["**/camera/**"]],
    )
    def test_telemetry_globs_are_not_the_actuation_plane(self, key_exprs: list[str]) -> None:
        assert _is_permissive_acl_shape(_acl(key_exprs, _ANY)) is False

    def test_a_subscribe_only_glob_is_not_a_command(self) -> None:
        """Reading the command plane is observation; writing it is actuation."""
        assert _is_permissive_acl_shape(_acl(["**/cmd"], _ANY, messages=["declare_subscriber"])) is False


class TestTheKeyExpressionMatcher:
    @pytest.mark.parametrize(
        "ke", ["**", "**/cmd", "*/cmd", "$*/cmd", "**/broadcast", "broadcast", "safety/**", "**/safety/*"]
    )
    def test_reaches(self, ke: str) -> None:
        assert _key_expr_reaches_actuation_plane(ke) is True

    @pytest.mark.parametrize("ke", ["**/presence", "arm-1/state", "**/response/**", "cmd/**", "safety", ""])
    def test_does_not_reach(self, ke: str) -> None:
        assert _key_expr_reaches_actuation_plane(ke) is False


class TestTheLoaderWarns:
    def test_an_actuation_glob_bound_to_a_subject_without_cns_is_named(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        data = _acl(["**/cmd", "**/safety/**"], {"id": "lab", "interfaces": ["eth0"]})
        path = tmp_path / "acl.json5"
        path.write_text(json.dumps(data), encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            _acl_config._load_acl_file(path)

        hits = [r.message for r in caplog.records if "cert_common_names" in r.message and "lab" in r.message]
        assert hits, [r.message for r in caplog.records]

    def test_the_template_shape_does_not_warn(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        data = _acl(["**/cmd", "**/broadcast", "**/safety/**"], _NAMED)
        path = tmp_path / "acl.json5"
        path.write_text(json.dumps(data), encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            _acl_config._load_acl_file(path)

        assert not [r for r in caplog.records if "cert_common_names" in r.message]
