"""Red-team adversarial tests against live Zenoh sessions.

Each test boots one or more real ``zenoh.open()`` peers in process
and exercises a vector from PENTEST.md. They skip cleanly when
``eclipse-zenoh`` is unavailable.

Vector IDs (from .autonomous/PLAN.md):

* Z3 — downsampling caps cmd publish rate.
* Z4 — low_pass_filter caps cmd payload bytes.
* NS — namespace isolates two fleets on the same TCP listener.
* L2 — validate_command rejects an attacker-controlled policy_host
  even if the wire reached the dispatcher (defence in depth).

Tests intentionally do NOT spin up a full :class:`Mesh` — the goal is
to verify Zenoh's transport-layer behaviour under the configs
emitted by :mod:`strands_robots.mesh._zenoh_config`. End-to-end Mesh
tests live elsewhere (`test_deep_mesh.py`).
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

zenoh = pytest.importorskip("zenoh")

from strands_robots.mesh import _zenoh_config as zc  # noqa: E402

# Helpers ---------------------------------------------------------------


def _new_config(
    *,
    namespace: str = "strands",
    listen_port: int | None = None,
    connect: list[str] | None = None,
    extra_blocks: list[tuple[str, str]] | None = None,
) -> Any:
    """Minimal Zenoh config, no auth (auth_mode tested separately)."""
    cfg = zenoh.Config()
    cfg.insert_json5("mode", '"peer"')
    cfg.insert_json5("scouting/multicast/enabled", "false")
    cfg.insert_json5("scouting/gossip/enabled", "true")
    cfg.insert_json5("namespace", json.dumps(namespace))
    if listen_port is not None:
        cfg.insert_json5("listen/endpoints", json.dumps([f"tcp/127.0.0.1:{listen_port}"]))
    if connect:
        cfg.insert_json5("connect/endpoints", json.dumps(connect))
    for path, value in extra_blocks or []:
        cfg.insert_json5(path, value)
    return cfg


def _wait_settle(seconds: float = 0.4) -> None:
    """Brief sleep to let gossip + subscriber declarations settle."""
    time.sleep(seconds)


# NS — Namespace isolation ---------------------------------------------


class TestNamespaceIsolation:
    """Two fleets on the same TCP listener cannot exchange messages
    when they have different ``namespace`` configs.
    """

    def test_distinct_namespaces_do_not_route(self):
        s_a = zenoh.open(_new_config(namespace="fleet_a", listen_port=27001))
        s_b = zenoh.open(_new_config(namespace="fleet_b", connect=["tcp/127.0.0.1:27001"]))
        try:
            got_a: list[str] = []
            got_b: list[str] = []
            s_a.declare_subscriber("robot/cmd", lambda s: got_a.append(s.payload.to_bytes().decode()))
            s_b.declare_subscriber("robot/cmd", lambda s: got_b.append(s.payload.to_bytes().decode()))
            _wait_settle()

            s_b.put("robot/cmd", b'"from_fleet_b"')
            _wait_settle(0.3)

            assert got_a == [], f"fleet_a saw fleet_b traffic: {got_a}"
            assert got_b == ['"from_fleet_b"']
        finally:
            s_b.close()
            s_a.close()

    def test_same_namespace_routes(self):
        s_a = zenoh.open(_new_config(namespace="same_fleet", listen_port=27002))
        s_b = zenoh.open(_new_config(namespace="same_fleet", connect=["tcp/127.0.0.1:27002"]))
        try:
            got: list[str] = []
            s_a.declare_subscriber("robot/cmd", lambda s: got.append(s.payload.to_bytes().decode()))
            _wait_settle()

            s_b.put("robot/cmd", b'"hello"')
            _wait_settle(0.3)

            assert got == ['"hello"']
        finally:
            s_b.close()
            s_a.close()


# Z4 — low_pass_filter byte cap ---------------------------------------


class TestLowPassFilterByteCap:
    """A jumbo cmd payload is dropped at the transport ingress filter
    before the subscriber's callback runs.

    Important: Zenoh's ``low_pass_filter`` requires a non-empty
    ``interfaces`` list — an empty / missing field silently no-ops
    the cap. The block built by ``_zenoh_config.low_pass_filter_block``
    enumerates every local interface so the cap applies regardless of
    which NIC the link rides. The test below uses real interface
    names.
    """

    def test_oversized_cmd_dropped_at_transport(self):
        cap = 256
        # Use a broad iface allowlist so the filter binds to the link
        # the test peers actually use (lo0 on macOS, lo on Linux,
        # eth0/en0 if testing across NICs).
        ifaces = ["lo", "lo0", "eth0", "en0", "en1", "wlan0"]
        lpf_block = (
            "low_pass_filter",
            json.dumps(
                [
                    {
                        "id": "cmd_size_cap",
                        "interfaces": ifaces,
                        "messages": ["put"],
                        "flows": ["ingress"],
                        "key_exprs": ["**/cmd"],
                        "size_limit": cap,
                    }
                ]
            ),
        )
        s_listen = zenoh.open(_new_config(namespace="strands", listen_port=27010, extra_blocks=[lpf_block]))
        s_pub = zenoh.open(_new_config(namespace="strands", connect=["tcp/127.0.0.1:27010"]))
        try:
            got: list[bytes] = []
            s_listen.declare_subscriber("strands/*/cmd", lambda s: got.append(s.payload.to_bytes()))
            _wait_settle()

            small = b'{"action":"status"}'
            big = b"x" * (cap + 64)

            s_pub.put("strands/r1/cmd", small)
            s_pub.put("strands/r1/cmd", big)
            _wait_settle(0.4)

            assert small in got, "small payload should have been delivered"
            assert big not in got, (
                f"oversized payload ({len(big)}B > cap {cap}B) was delivered — "
                "low_pass_filter is not enforcing (check interfaces field)"
            )
        finally:
            s_pub.close()
            s_listen.close()


# Z3 — downsampling rate cap ------------------------------------------


class TestDownsamplingRateCap:
    """A burst at 200 Hz is throttled at the transport. The receiver
    sees a small fraction of the published samples.
    """

    def test_high_rate_publish_is_throttled(self):
        freq_hz = 5.0  # cap
        s_listen = zenoh.open(
            _new_config(
                namespace="strands",
                listen_port=27020,
                extra_blocks=[
                    (
                        "downsampling",
                        json.dumps(
                            [
                                {
                                    "id": "cmd_rate_cap",
                                    "messages": ["put"],
                                    "flows": ["ingress"],
                                    "rules": [{"key_expr": "**/cmd", "freq": freq_hz}],
                                }
                            ]
                        ),
                    )
                ],
            )
        )
        s_pub = zenoh.open(_new_config(namespace="strands", connect=["tcp/127.0.0.1:27020"]))
        try:
            received: list[float] = []
            s_listen.declare_subscriber("strands/*/cmd", lambda s: received.append(time.time()))
            _wait_settle()

            t0 = time.time()
            for i in range(200):
                s_pub.put("strands/r1/cmd", f'{{"i":{i}}}'.encode())
            _wait_settle(1.5)
            t1 = time.time()

            duration = t1 - t0
            # At 5 Hz over ~1.5 s we expect roughly 5-15 samples through.
            # Allow a wide band; just assert the throttle is functioning
            # (much less than the 200 sent).
            assert len(received) < 100, (
                f"downsampling did not throttle: got {len(received)} of 200 in {duration:.2f}s (cap was {freq_hz} Hz)"
            )
            assert len(received) >= 1, "downsampling threw away every sample"
        finally:
            s_pub.close()
            s_listen.close()


# Round-trip: _zenoh_config emits configs Zenoh accepts -----------------


class TestZenohConfigRoundtrip:
    """The blocks emitted by :mod:`_zenoh_config` parse cleanly and
    actually take effect in a live session.
    """

    def test_production_blocks_open_a_live_session(self):
        """Smoke test: a config built only from production blocks
        (namespace + scouting + transport caps + downsampling +
        low_pass_filter + adminspace) opens a live Zenoh session.

        If any builder emits an invalid block, ``zenoh.open()`` raises
        and we catch the regression here.
        """
        ns = zc.resolve_namespace()
        cfg = zenoh.Config()
        cfg.insert_json5("mode", '"peer"')
        cfg.insert_json5("listen/endpoints", json.dumps(["tcp/127.0.0.1:27030"]))
        for path, value in (
            zc.namespace_block(),
            *zc.scouting_block(),
            *zc.transport_caps_block(),
            zc.adminspace_block(),
            zc.downsampling_block(ns),
            zc.low_pass_filter_block(ns),
        ):
            cfg.insert_json5(path, value)
        s = zenoh.open(cfg)
        try:
            assert s.zid() is not None
        finally:
            s.close()

    def test_production_low_pass_filter_actually_drops_oversized(self):
        """The production-builder ``low_pass_filter`` block actually
        drops oversized cmd payloads in a live session.

        This is the regression test for the red-team finding that
        Zenoh's filter requires a non-empty ``interfaces`` list — an
        empty/missing field silently no-ops the cap.
        """
        ns = "strands"
        # Tight cap to keep the test fast.
        import os

        old_cap = os.environ.get("STRANDS_MESH_MAX_CMD_BYTES")
        os.environ["STRANDS_MESH_MAX_CMD_BYTES"] = "256"
        try:
            lpf = zc.low_pass_filter_block(ns)
            cfg_l = _new_config(namespace=ns, listen_port=27031, extra_blocks=[lpf])
            cfg_p = _new_config(namespace=ns, connect=["tcp/127.0.0.1:27031"], extra_blocks=[lpf])

            s_l = zenoh.open(cfg_l)
            s_p = zenoh.open(cfg_p)
            try:
                got: list[int] = []
                s_l.declare_subscriber(f"{ns}/**", lambda s: got.append(len(s.payload.to_bytes())))
                _wait_settle()

                small = b'{"action":"status"}'
                big = b"x" * 1024  # 4x cap
                s_p.put(f"{ns}/r1/cmd", small)
                s_p.put(f"{ns}/r1/cmd", big)
                _wait_settle(0.5)

                assert len(small) in got, f"small payload not delivered: {got}"
                assert 1024 not in got, (
                    f"oversized payload delivered through production filter: {got}. "
                    "low_pass_filter is no-opping; check that interfaces are enumerated."
                )
            finally:
                s_p.close()
                s_l.close()
        finally:
            if old_cap is None:
                os.environ.pop("STRANDS_MESH_MAX_CMD_BYTES", None)
            else:
                os.environ["STRANDS_MESH_MAX_CMD_BYTES"] = old_cap
