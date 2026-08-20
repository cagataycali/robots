"""What address and what patience a single-node elastic launch gets.

Pure tests: they never construct a store, so they run in milliseconds and are safe to
keep in a sweep — unlike the launch tests in test_inproc.py, which drive a real torch
rendezvous and are the subject of BUGS.md Q37.
"""

from __future__ import annotations

import pytest

from strands_robots.training._inproc import (
    DEFAULT_RDZV_TIMEOUT_S,
    LOCAL_ADDR_ENV,
    RDZV_TIMEOUT_ENV,
    launch_local_addr,
    looks_like_reverse_dns,
    free_local_port,
    rdzv_timeout_s,
    rendezvous_endpoint,
)


class TestRendezvousEndpoint:
    def test_an_explicit_endpoint_always_wins(self) -> None:
        assert rendezvous_endpoint("head-node:29500", 4) == "head-node:29500"
        assert rendezvous_endpoint("head-node:29500", 1) == "head-node:29500"

    def test_a_single_node_launch_gets_a_concrete_loopback_port(self) -> None:
        assert rendezvous_endpoint("", 1, port_picker=lambda: 45678) == "127.0.0.1:45678"

    def test_never_port_zero_and_never_the_ambiguous_localhost(self) -> None:
        # port 0 is not an address a client can dial, and `localhost` resolves to both
        # ::1 and 127.0.0.1 on macOS - either one lets the store and its client miss
        # each other inside libtorch, where no Python timeout can help (Q37).
        chosen = rendezvous_endpoint("", 1)
        host, _, port = chosen.rpartition(":")
        assert host == "127.0.0.1"
        assert int(port) > 0

    def test_a_multi_node_launch_without_an_endpoint_is_refused_with_the_reason(self) -> None:
        with pytest.raises(ValueError, match="rdzv_endpoint"):
            rendezvous_endpoint("", 3)

    def test_the_picked_port_is_actually_free(self) -> None:
        import socket

        port = free_local_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))  # would raise if something held it


class TestRdzvTimeout:
    def test_default_when_unset(self) -> None:
        assert rdzv_timeout_s({}) == DEFAULT_RDZV_TIMEOUT_S

    def test_an_operator_can_raise_it_for_a_slow_cluster(self) -> None:
        assert rdzv_timeout_s({RDZV_TIMEOUT_ENV: "900"}) == 900

    @pytest.mark.parametrize("junk", ["", "soon", "0", "-5", "None"])
    def test_junk_and_non_positive_fall_back_rather_than_disabling_the_bound(self, junk: str) -> None:
        # an unparseable or zero value must not be able to restore the unbounded wait
        assert rdzv_timeout_s({RDZV_TIMEOUT_ENV: junk}) == DEFAULT_RDZV_TIMEOUT_S


class TestLocalAddr:
    """MASTER_ADDR — the actual Q37 root cause.

    Left to torch, this address is ``socket.getfqdn()``, which on this Mac returns the
    reverse-DNS PTR name of ``::1`` (``1.0.0.0...ip6.arpa``). Nothing can resolve that
    forwards, so the worker store's client dials it forever inside libtorch's C++
    retry loop — a run parked on "Rendezvous'ing worker group" with no error and no
    timeout that can reach it.
    """

    def test_a_single_node_launch_is_pinned_to_loopback(self) -> None:
        assert launch_local_addr(1, env={}, fqdn=lambda: "1.0.0.0.ip6.arpa") == "127.0.0.1"

    def test_a_broken_fqdn_cannot_reach_a_single_node_launch(self) -> None:
        def _boom() -> str:
            raise OSError("no resolution here")

        assert launch_local_addr(1, env={}, fqdn=_boom) == "127.0.0.1"

    def test_an_explicit_address_wins_everywhere(self) -> None:
        assert launch_local_addr(1, "10.0.0.7", env={}) == "10.0.0.7"
        assert launch_local_addr(4, "10.0.0.7", env={}) == "10.0.0.7"

    def test_the_operator_can_override_by_env(self) -> None:
        assert launch_local_addr(4, env={LOCAL_ADDR_ENV: " 10.0.0.9 "}) == "10.0.0.9"

    def test_a_multi_node_launch_keeps_torchs_own_resolution(self) -> None:
        # the address must be reachable from the OTHER nodes; guessing one here would
        # be worse than letting torch resolve it
        assert launch_local_addr(4, env={}, fqdn=lambda: "head.cluster.local") is None

    def test_a_multi_node_launch_with_a_reverse_dns_fqdn_is_warned_about(self, caplog) -> None:
        with caplog.at_level("WARNING"):
            assert launch_local_addr(4, env={}, fqdn=lambda: "1.0.0.0.0.ip6.arpa") is None
        assert LOCAL_ADDR_ENV in caplog.text
        assert "hang" in caplog.text, "a silent hang deserves to be named as one"

    @pytest.mark.parametrize(
        "name,is_reverse",
        [
            ("1.0.0.0.0.0.0.0.ip6.arpa", True),
            ("1.0.0.0.0.0.0.0.ip6.arpa.", True),
            ("4.3.2.1.in-addr.arpa", True),
            # the zone apex itself, in any case: not a hostname anyone can dial either
            ("IP6.ARPA", True),
            ("head.cluster.local", False),
            ("127.0.0.1", False),
            ("arpa-node.example.com", False),
        ],
    )
    def test_reverse_dns_names_are_recognised(self, name: str, is_reverse: bool) -> None:
        assert looks_like_reverse_dns(name) is is_reverse
