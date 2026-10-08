"""Regression tests: a glob over the command or safety plane is as permissive as ``**``.

``_is_permissive_acl_shape`` (the input to ``Mesh.start``'s refuse-to-start
gate) recognised a wide-open ``allow`` rule only by the literal string ``**``
in its ``key_exprs``. The module's own docstring and the shipped template tell
operators to write globs instead, ``**/cmd``, ``**/broadcast``,
``**/safety/**``, which together cover every actuation-relevant topic. Bound
to a subject with no certificate constraint, such a rule let any CA-signed
peer command and stop every robot, and the gate stayed silent.

Now an ``allow`` rule that can ``put`` on any key expression reaching a
peer's command topic, the broadcast, the safety commands or a teleop input
stream counts as writing the actuation plane, decided structurally for any
peer name, glob or namespace (``neon/cmd``, ``scout-$*/cmd``,
``fleet/*/cmd``, ``**/input/**``) rather than against one sample key. Bound to
a subject without ``cert_common_names`` (an ``interfaces`` list scopes a link,
not a peer) it makes the gate refuse to start, and the loader names it.
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
            ["neon/cmd", "scout-01/cmd", "**/input/**"],
            ["**/neon/cmd"],
            ["**/input/**"],
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

    @pytest.mark.parametrize("subject", [{"id": "lab", "interfaces": ["eth0"]}, {"id": "any", "interfaces": ["*"]}])
    @pytest.mark.parametrize("key_exprs", [["neon/cmd", "scout-01/cmd", "**/input/**"], ["**"], ["**/safety/*"]])
    def test_an_interface_only_subject_is_refused(self, key_exprs: list[str], subject: dict[str, Any]) -> None:
        """An ``interfaces`` list admits every CA-signed peer on that link."""
        assert _is_permissive_acl_shape(_acl(key_exprs, subject)) is True

    @pytest.mark.parametrize("key_exprs", [[""], ["a//cmd"], [7]])
    def test_a_key_expression_the_check_cannot_read_fails_closed(self, key_exprs: list[Any]) -> None:
        assert _is_permissive_acl_shape(_acl(key_exprs, _ANY)) is True  # type: ignore[arg-type]

    def test_a_subscribe_only_glob_is_not_a_command(self) -> None:
        """Reading the command plane is observation; writing it is actuation."""
        assert _is_permissive_acl_shape(_acl(["**/cmd"], _ANY, messages=["declare_subscriber"])) is False


class TestTheKeyExpressionMatcher:
    @pytest.mark.parametrize(
        "ke",
        [
            "**",
            "**/cmd",
            "*/cmd",
            "$*/cmd",
            "**/broadcast",
            "broadcast",
            "safety/**",
            "**/safety/*",
            "neon/cmd",
            "scout-01/cmd",
            "scout-$*/cmd",
            "fleet/*/cmd",
            "**/neon/cmd",
            "strands/neon/cmd",
            "strands/strands/neon/cmd",
            "**/input/**",
            "neon/input/*",
            "strands/broadcast",
            "",
        ],
    )
    def test_reaches(self, ke: str) -> None:
        assert _key_expr_reaches_actuation_plane(ke) is True

    @pytest.mark.parametrize(
        "ke",
        [
            "**/presence",
            "arm-1/state",
            "**/response/**",
            "cmd/**",
            "safety",
            "**/state/**",
            "**/robot-a/lidar/**",
            "**/response/robot-a/*",
            "**/robot-a/response/**",
            "input/**",
        ],
    )
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


class TestTheDetectorAgreesWithTheLiveMatcher:
    """Every spelling a live Zenoh ACL lets a ``put`` through on is one the detector flags.

    The mesh publishes ``strands/<peer>/cmd`` and ``strands/<peer>/input/<device>``,
    and the matcher sees those keys behind the configured namespace (here
    ``lab``): ``**/neon/cmd`` and ``lab/strands/neon/cmd`` admit the command,
    ``neon/cmd`` and ``strands/neon/cmd`` admit nothing. The detector flags the
    inert spellings too (fail closed), so an ACL is never judged safe because of
    how a command grant happened to be spelled.
    """

    @pytest.mark.parametrize(
        ("port", "key_expr", "topic", "delivered", "flagged"),
        [
            (28741, "**/cmd", "strands/neon/cmd", True, True),
            (28742, "**/neon/cmd", "strands/neon/cmd", True, True),
            (28743, "lab/strands/neon/cmd", "strands/neon/cmd", True, True),
            (28744, "neon/cmd", "strands/neon/cmd", False, True),
            (28745, "strands/neon/cmd", "strands/neon/cmd", False, True),
            (28746, "**/input/**", "strands/neon/input/gamepad", True, True),
            (28747, "**/presence", "strands/neon/cmd", False, False),
        ],
    )
    def test_spelling(self, port: int, key_expr: str, topic: str, delivered: bool, flagged: bool) -> None:
        zenoh = pytest.importorskip("zenoh")
        import time

        acl = _acl([key_expr], {"id": "lo", "interfaces": ["lo"]})
        acl["rules"][0]["flows"] = ["ingress", "egress"]
        acl["rules"].append(
            {
                "id": "sub",
                "key_exprs": ["**"],
                "messages": ["declare_subscriber"],
                "flows": ["ingress", "egress"],
                "permission": "allow",
            }
        )
        acl["rules"].append(
            {
                "id": "canary",
                "key_exprs": ["lab/canary"],
                "messages": ["put"],
                "flows": ["ingress", "egress"],
                "permission": "allow",
            }
        )
        acl["policies"][0]["rules"] += ["sub", "canary"]

        def config(side: str) -> Any:
            cfg = zenoh.Config()
            cfg.insert_json5("mode", '"peer"')
            cfg.insert_json5("scouting/multicast/enabled", "false")
            cfg.insert_json5("namespace", '"lab"')
            cfg.insert_json5(f"{side}/endpoints", json.dumps([f"tcp/127.0.0.1:{port}"]))
            cfg.insert_json5("access_control", json.dumps(acl))
            return cfg

        # A put the ACL always admits brackets the one under test: its first arrival
        # says the link and the subscription are up, and one sent after the topic
        # arrives after it (one session, one priority), so a denied put is known to
        # be dropped as soon as the next canary lands - no fixed sleep either way.
        got: list[str] = []
        receiver = zenoh.open(config("listen"))
        try:
            receiver.declare_subscriber("**", lambda sample: got.append(str(sample.key_expr)))
            sender = zenoh.open(config("connect"))
            try:

                def canary() -> None:
                    seen = got.count("canary")
                    deadline = time.monotonic() + 10.0
                    while got.count("canary") == seen and time.monotonic() < deadline:
                        sender.put("canary", b"{}")
                        time.sleep(0.02)
                    assert got.count("canary") > seen, "premise: the admitted canary put is delivered"

                canary()
                sender.put(topic, b"{}")
                canary()
            finally:
                sender.close()
        finally:
            receiver.close()

        topics = [key for key in got if key != "canary"]
        assert bool(topics) is delivered, got
        assert _key_expr_reaches_actuation_plane(key_expr) is flagged
