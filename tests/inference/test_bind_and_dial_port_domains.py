"""One ``port`` parameter, two accepted domains: the server binds, the client dials.

:class:`~strands_robots.inference.PolicyServer` and
:class:`~strands_robots.inference.RemotePolicy` both take a ``port``, and both
used to store whatever they were handed. The server's field went straight onto
``self.port``; the client's went verbatim into ``f"ws://{host}:{port}"``, so
``ws://127.0.0.1:nan`` and ``ws://127.0.0.1:[8765]`` were constructed and kept.
A WebSocket target is only resolved on first use, so neither was refused by the
transport - each surfaced much later as an unreachable server, implicating the
service the caller was trying to reach rather than the port.

The two halves cannot share one domain, which is what this module pins:

* the client **dials**, so it needs a port it can address -
  :func:`~strands_robots.utils.tcp_port_error`'s ``[1, 65535]`` unchanged;
* the server **binds**, so it may ask the kernel for an ephemeral port -
  ``PolicyServer`` documents ``0`` as exactly that and reads the assigned port
  back onto ``.port``, so its domain is the same range with the floor at 0.

The asymmetry is pinned deliberately: a later change that "harmonises" the two
fails here rather than silently removing the ephemeral bind or admitting a
client that cannot dial what it was pointed at.

These are constructor-level contracts, so nothing here opens a socket except
the one test that asserts the documented ephemeral bind still works end to end.
"""

from typing import Any

import pytest

from strands_robots.inference import PolicyServer, RemotePolicy
from strands_robots.inference import server as server_mod
from strands_robots.policies import MockPolicy

#: Values that name no port on either side. Each is refused by both surfaces.
#: ``2.7`` and ``'8765'`` matter because neither half applies ``int()``, so a
#: non-integer reached the URI and the bind unchanged; ``True``/``False``
#: because ``bool`` is an ``int`` subclass, so a bare range test admits ``True``
#: as a silent privileged port 1 and a bare zero test reads ``False`` as the
#: ephemeral request.
UNUSABLE_PORTS: list[Any] = [-1, 70000, 2.7, True, False, float("nan"), float("inf"), "8765", [8765], None]

#: Accepted by both halves - the boundaries of the addressable range, so a guard
#: that is off by one at either end fails here rather than in the field.
DIALABLE_PORTS: list[Any] = [1, 8765, 65535]


def _server(**kwargs: Any) -> PolicyServer:
    """Construct a ``PolicyServer``, splatted so off-type ports reach the guard.

    The ``port`` parameter is annotated ``int``; several cases here deliberately
    pass something else to prove the runtime refuses it, which a direct call
    would make a type error rather than a test.
    """
    return PolicyServer(policy=MockPolicy(), **kwargs)


def _client(**kwargs: Any) -> RemotePolicy:
    """Construct a ``RemotePolicy``, splatted for the same reason as ``_server``."""
    return RemotePolicy(**kwargs)


class TestNeitherHalfAcceptsAnUnusablePort:
    """A value that names no port is refused by the bind and the dial surface."""

    @pytest.mark.parametrize("port", UNUSABLE_PORTS)
    def test_the_server_refuses_it(self, port: Any) -> None:
        """``PolicyServer`` refuses it, naming the parameter, value and range."""
        with pytest.raises(ValueError) as exc:
            _server(port=port)
        text = str(exc.value)
        assert "PolicyServer" in text
        assert "invalid port" in text
        assert "1-65535" in text

    @pytest.mark.parametrize("port", UNUSABLE_PORTS)
    def test_the_client_refuses_it(self, port: Any) -> None:
        """``RemotePolicy`` refuses it the same way."""
        with pytest.raises(ValueError) as exc:
            _client(port=port)
        text = str(exc.value)
        assert "RemotePolicy" in text
        assert "invalid port" in text
        assert "1-65535" in text

    @pytest.mark.parametrize("port", UNUSABLE_PORTS)
    def test_no_client_uri_is_built_from_it(self, port: Any) -> None:
        """The refusal precedes ``uri``, so no unusable endpoint is ever stored.

        Asserts the object does not exist rather than that a later connect
        fails: ``uri`` is the client's whole addressing state, and a stored
        ``ws://127.0.0.1:nan`` is what turned a bad port into an apparently
        unreachable server.
        """
        with pytest.raises(ValueError):
            client = _client(port=port)
            pytest.fail(f"constructed a client addressing {client.uri!r}")

    @pytest.mark.parametrize("port", UNUSABLE_PORTS)
    def test_the_server_refuses_it_before_building_a_policy(self, port: Any) -> None:
        """The refusal precedes ``create_policy``, which can download a checkpoint.

        Uses the ``policy_provider`` spelling, where accepting the port first
        would resolve and build a policy for a server that can never bind.
        """
        calls: list[str] = []

        class _Boom(Exception):
            pass

        def _fail(provider: str, **_kwargs: Any) -> Any:
            calls.append(provider)
            raise _Boom("create_policy must not be reached")

        import strands_robots.policies as policies_pkg

        original = policies_pkg.create_policy
        policies_pkg.create_policy = _fail  # type: ignore[assignment]
        try:
            with pytest.raises(ValueError, match="invalid port"):
                PolicyServer(policy_provider="mock", port=port)
        finally:
            policies_pkg.create_policy = original  # type: ignore[assignment]
        assert calls == [], "the port was accepted long enough to build a policy"


class TestAnAddressablePortIsAcceptedByBoth:
    """The over-reach controls: the guards refuse only what they must."""

    @pytest.mark.parametrize("port", DIALABLE_PORTS)
    def test_the_server_accepts_it(self, port: int) -> None:
        """A port in range is stored for the bind."""
        assert _server(port=port).port == port

    @pytest.mark.parametrize("port", DIALABLE_PORTS)
    def test_the_client_dials_it(self, port: int) -> None:
        """A port in range is interpolated into the endpoint."""
        assert _client(port=port).uri == f"ws://127.0.0.1:{port}"


class TestOnlyTheServerMayAskForAnEphemeralPort:
    """``0`` is the documented ephemeral bind, and only a binder can make it."""

    def test_the_server_accepts_zero(self) -> None:
        """``PolicyServer(port=0)`` is the documented "any free port" request."""
        assert _server(port=0).port == 0

    def test_the_client_refuses_zero(self) -> None:
        """``RemotePolicy`` cannot dial "any free port", so ``0`` is refused.

        This is the asymmetry that makes the two domains distinct rather than a
        single shared rule. Harmonising them in either direction breaks one of
        these two tests.
        """
        with pytest.raises(ValueError, match="invalid port"):
            _client(port=0)

    def test_zero_still_binds_and_reports_the_assigned_port(self) -> None:
        """The ephemeral path works end to end: ``start()`` reports the real port.

        The one behaviour that must not regress - accepting ``0`` is only
        meaningful because the OS-assigned port is readable afterwards.
        """
        server = _server(port=0).start()
        try:
            assert server.port > 0
            assert server.port != 0
        finally:
            server.stop()

    def test_false_is_not_read_as_the_ephemeral_request(self) -> None:
        """``False == 0``, so the zero test must check the type first.

        Otherwise a boolean would be accepted as "any free port", and ``True``
        would bind privileged port 1.
        """
        with pytest.raises(ValueError, match="invalid port"):
            _server(port=False)
        with pytest.raises(ValueError, match="invalid port"):
            _server(port=True)


class TestTheZeroExemptionIsTheOnlyDifferenceFromTheSharedRule:
    """``_bind_port_error`` defers everything but the floor to the shared domain."""

    @pytest.mark.parametrize("port", [*UNUSABLE_PORTS, *DIALABLE_PORTS])
    def test_it_agrees_with_the_shared_domain_away_from_zero(self, port: Any) -> None:
        """Away from ``0``, bind and dial return byte-identical verdicts."""
        from strands_robots.utils import tcp_port_error

        assert server_mod._bind_port_error(port, "port", "Ctx") == tcp_port_error(port, "port", "Ctx")

    def test_it_differs_from_the_shared_domain_at_zero(self) -> None:
        """At ``0`` the wrapper accepts and the shared domain refuses."""
        from strands_robots.utils import tcp_port_error

        assert server_mod._bind_port_error(0, "port", "Ctx") is None
        assert tcp_port_error(0, "port", "Ctx") is not None


class TestAnExplicitEndpointSupersedesThePort:
    """``endpoint`` wins over ``host``/``port``, so only the effective knob is checked."""

    def test_a_port_is_not_refused_when_an_endpoint_is_given(self) -> None:
        """The port is unread on this path, so refusing it would be a false rejection."""
        client = _client(endpoint="ws://gpu-box:9999", port=0)
        assert client.uri == "ws://gpu-box:9999"

    def test_the_port_is_checked_when_it_is_the_effective_spelling(self) -> None:
        """An empty endpoint falls back to ``host``/``port``, so the port is read."""
        with pytest.raises(ValueError, match="invalid port"):
            _client(endpoint="", port=0)


class TestTheCliBindsOnTheSameDomainAsTheClassItConstructs:
    """The CLI cannot refuse a port ``PolicyServer`` accepts, or vice versa."""

    def test_it_accepts_the_ephemeral_bind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--port 0`` reaches ``serve()``.

        It previously exited 2 on an inline ``1 <= port <= 65535`` range, so the
        CLI refused the ephemeral bind its own class documents as first-class.
        """
        served: dict[str, object] = {}

        def fake_serve(self: PolicyServer) -> None:
            served["port"] = self.port

        monkeypatch.setattr(PolicyServer, "serve", fake_serve)
        server_mod.main(["--provider", "mock", "--port", "0"])

        assert served == {"port": 0}

    @pytest.mark.parametrize("port", ["-1", "70000"])
    def test_it_refuses_an_out_of_range_port(self, port: str) -> None:
        """A port outside the range still exits 2, with the shared reason."""
        with pytest.raises(SystemExit) as exc:
            server_mod.main(["--provider", "mock", "--port", port])
        assert exc.value.code == 2
