"""``timeout_ms`` is a wait budget, and only a positive whole number is one.

Three public constructors write a caller-supplied ``timeout_ms`` into the same
ZMQ socket option - :class:`~strands_robots.policies.groot.client.Gr00tInferenceClient`,
:class:`~strands_robots.policies.moveit2.client.MoveIt2InferenceClient`, and the
:class:`~strands_robots.policies.moveit2.MoveIt2Policy` that forwards it (also
reachable as ``create_policy("moveit2", timeout_ms=...)``). All three set
``RCVTIMEO``/``SNDTIMEO`` from it, so a value that cannot be a budget cannot be
honored by any of them.

Measured on pyzmq 27.1.0 against a sidecar that answers every request in ~5 ms,
before this domain existed - the three surfaces agreed on every row, which is
why one shared rule covers them:

===============  ==========  ============================================
``timeout_ms``   verdict     effect on a healthy sidecar
===============  ==========  ============================================
``15000``        accepted    answered in 5.5 ms
``0``            accepted    ``zmq.Again`` after 0.0 ms
``True``         accepted    ``zmq.Again`` after 0.5 ms (a 1 ms budget)
``-1``           accepted    unbounded receive (ZMQ's "block forever")
``-5000``        raised      ``ZMQError: Invalid argument``
``2.7``          raised      ``TypeError: expected int, got: 2.7``
``"15000"``      raised      ``TypeError: unicode not allowed``
===============  ==========  ============================================

The first three are the silent half: the call reports success and the socket is
live, so a *healthy* service is reported unreachable on every request. ``-1`` is
the opposite failure - against a sidecar that never answers it does not return
at all, which is the unbounded block the ``LINGER, 0`` two lines below the same
``setsockopt`` was added to remove.

So the domain is :func:`~strands_robots.utils.positive_count_error`: the option
is a C ``int``, so an integral float is not coerced but refused by ``setsockopt``
itself, exactly as for the ``range()`` bounds and framebuffer dimensions that
rule already covers.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
from typing import Any

import pytest

from strands_robots.policies.groot.client import Gr00tInferenceClient
from strands_robots.policies.moveit2 import MoveIt2Policy
from strands_robots.policies.moveit2.client import MoveIt2InferenceClient
from strands_robots.utils import positive_count_error

# A port nothing listens on. ZMQ ``connect`` is asynchronous, so construction
# neither blocks nor needs a server - only the socket option is under test.
DEAD_PORT = 57419

#: Values that cannot be a wait budget, and what each one would have done.
UNUSABLE_BUDGETS: list[Any] = [
    0,  # ZMQ: return immediately -> a healthy sidecar reads as unreachable
    -1,  # ZMQ: block forever -> an unbounded receive with no recovery
    -5000,  # refused by ZMQ itself, as a bare ZMQError
    True,  # int subclass -> a silent 1 ms budget
    False,  # int subclass -> the zero budget by another spelling
    2.7,  # setsockopt refuses an integral or fractional float
    15000.0,
    math.nan,
    math.inf,
    "15000",
    None,
    [15000],
]

#: Budgets that are usable and must keep working unchanged.
USABLE_BUDGETS: list[int] = [1, 250, 15000, 120_000]

SURFACES = ("Gr00tInferenceClient", "MoveIt2InferenceClient", "MoveIt2Policy")


def _build(surface: str, **kwargs: Any) -> Any:
    """Construct one surface with ``kwargs`` splatted.

    One funnel for every construction in this module: the probe values are
    deliberately outside the declared ``int`` annotation, and a ``**dict[str,
    Any]`` splat is what keeps that intent in the test rather than spread over
    a per-call suppression at each site.
    """
    if surface == "Gr00tInferenceClient":
        return Gr00tInferenceClient(host="127.0.0.1", port=DEAD_PORT, **kwargs)
    if surface == "MoveIt2InferenceClient":
        return MoveIt2InferenceClient(host="127.0.0.1", port=DEAD_PORT, **kwargs)
    return MoveIt2Policy(host="127.0.0.1", port=DEAD_PORT, **kwargs)


def _client_of(obj: Any) -> Any:
    """The ZMQ-owning client, whether ``obj`` is one or forwards to one."""
    return getattr(obj, "_client", obj)


def _close(obj: Any) -> None:
    """Release a constructed client's socket and context."""
    client = _client_of(obj)
    try:
        client.socket.close(linger=0)
        client.context.term()
    except Exception:  # noqa: BLE001 - teardown of a probe, never the assertion
        pass


class TestAnUnusableBudgetIsRefused:
    """Every surface refuses a value that cannot be a wait budget."""

    @pytest.mark.parametrize("surface", SURFACES)
    @pytest.mark.parametrize("value", UNUSABLE_BUDGETS)
    def test_the_refusal_names_the_parameter_and_the_surface(self, surface: str, value: Any) -> None:
        with pytest.raises(ValueError) as excinfo:
            _build(surface, timeout_ms=value)
        message = str(excinfo.value)
        assert "timeout_ms" in message
        assert surface in message

    @pytest.mark.parametrize("surface", SURFACES)
    def test_the_zero_budget_is_refused_rather_than_read_as_non_blocking(self, surface: str) -> None:
        """``0`` is ZMQ's "return immediately", not a budget.

        Accepted, it made every request fail with ``zmq.Again`` before a healthy
        sidecar could answer - a service reported unreachable by the caller's
        own configuration.
        """
        with pytest.raises(ValueError, match="timeout_ms"):
            _build(surface, timeout_ms=0)

    @pytest.mark.parametrize("surface", SURFACES)
    def test_the_infinite_budget_is_refused_rather_than_read_as_no_timeout(self, surface: str) -> None:
        """``-1`` is ZMQ's "block forever", which is the absence of a budget."""
        with pytest.raises(ValueError, match="timeout_ms"):
            _build(surface, timeout_ms=-1)

    @pytest.mark.parametrize("surface", SURFACES)
    @pytest.mark.parametrize("value", [True, False])
    def test_a_bool_is_refused_rather_than_read_as_one_millisecond(self, surface: str, value: bool) -> None:
        """``True`` reached the socket as ``RCVTIMEO=1`` - a budget nobody chose."""
        with pytest.raises(ValueError, match="timeout_ms"):
            _build(surface, timeout_ms=value)


class TestAUsableBudgetIsHonored:
    """The accepted domain is unchanged, and the value still reaches the socket."""

    @pytest.mark.parametrize("surface", SURFACES)
    @pytest.mark.parametrize("value", USABLE_BUDGETS)
    def test_a_positive_whole_millisecond_budget_reaches_the_socket(self, surface: str, value: int) -> None:
        zmq = pytest.importorskip("zmq")
        obj = _build(surface, timeout_ms=value)
        try:
            client = _client_of(obj)
            assert client.timeout_ms == value
            assert client.socket.getsockopt(zmq.RCVTIMEO) == value
            assert client.socket.getsockopt(zmq.SNDTIMEO) == value
        finally:
            _close(obj)

    @pytest.mark.parametrize("surface", SURFACES)
    def test_the_default_budget_is_accepted(self, surface: str) -> None:
        """No caller who omitted the parameter is affected."""
        pytest.importorskip("zmq")
        obj = _build(surface)
        try:
            assert _client_of(obj).timeout_ms == 15000
        finally:
            _close(obj)

    def test_the_factory_path_honors_a_usable_budget(self) -> None:
        """``create_policy`` is how an agent reaches ``MoveIt2Policy``."""
        pytest.importorskip("zmq")
        from strands_robots.policies import create_policy

        policy = create_policy("moveit2", host="127.0.0.1", port=DEAD_PORT, timeout_ms=250)
        try:
            assert policy._client.timeout_ms == 250  # type: ignore[attr-defined]
        finally:
            _close(policy)

    @pytest.mark.parametrize("value", UNUSABLE_BUDGETS)
    def test_the_factory_path_refuses_an_unusable_budget(self, value: Any) -> None:
        from strands_robots.policies import create_policy

        with pytest.raises(ValueError, match="timeout_ms"):
            create_policy("moveit2", host="127.0.0.1", port=DEAD_PORT, timeout_ms=value)


class TestTheRefusalPrecedesTheSocket:
    """A refused budget must not create, configure or dial a socket."""

    @pytest.mark.parametrize(
        ("surface", "module_path"),
        [
            ("Gr00tInferenceClient", "strands_robots.policies.groot.client"),
            ("MoveIt2InferenceClient", "strands_robots.policies.moveit2.client"),
            ("MoveIt2Policy", "strands_robots.policies.moveit2.client"),
        ],
    )
    @pytest.mark.parametrize("value", [0, -1, True, 2.7, None])
    def test_no_zmq_context_is_built_for_a_refused_budget(
        self, surface: str, module_path: str, value: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Also the reason these refusals need no ZMQ installed to hold."""
        import importlib

        module = importlib.import_module(module_path)

        def fatal() -> Any:
            raise AssertionError("a refused budget reached the ZMQ loader")

        monkeypatch.setattr(module, "_load_zmq", fatal)
        with pytest.raises(ValueError, match="timeout_ms"):
            _build(surface, timeout_ms=value)


class TestTheDomainIsTheSharedRule:
    """No surface may diverge from :func:`positive_count_error`."""

    @pytest.mark.parametrize("surface", SURFACES)
    @pytest.mark.parametrize("value", [*UNUSABLE_BUDGETS, *USABLE_BUDGETS])
    def test_each_surface_refuses_exactly_what_the_shared_rule_refuses(self, surface: str, value: Any) -> None:
        pytest.importorskip("zmq")
        expected_refusal = positive_count_error(value, "timeout_ms", surface) is not None
        try:
            obj = _build(surface, timeout_ms=value)
        except ValueError:
            refused = True
        else:
            refused = False
            _close(obj)
        assert refused == expected_refusal, f"{surface} disagrees with the shared domain for {value!r}"

    @pytest.mark.parametrize("surface", SURFACES)
    def test_the_message_is_the_shared_rule_verbatim(self, surface: str) -> None:
        with pytest.raises(ValueError) as excinfo:
            _build(surface, timeout_ms=0)
        assert str(excinfo.value) == positive_count_error(0, "timeout_ms", surface)


class TestZmqReadsTheRefusedSentinelsAsModes:
    """The premise, measured rather than asserted: ``0``/``-1`` are not budgets.

    Pins the two dependency behaviours the domain rests on. If a future pyzmq
    stopped reading them as modes, this fails here rather than leaving the
    refusal justified by a stale claim.
    """

    def test_zero_returns_immediately_instead_of_waiting(self) -> None:
        zmq = pytest.importorskip("zmq")
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        try:
            socket.setsockopt(zmq.RCVTIMEO, 0)
            assert socket.getsockopt(zmq.RCVTIMEO) == 0
            socket.connect(f"tcp://127.0.0.1:{DEAD_PORT}")
            socket.send(b"ping")
            with pytest.raises(zmq.Again):
                socket.recv()
        finally:
            socket.close(linger=0)
            context.term()

    def test_minus_one_is_the_infinite_receive(self) -> None:
        zmq = pytest.importorskip("zmq")
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        try:
            socket.setsockopt(zmq.RCVTIMEO, -1)
            # ZMQ's own documented sentinel for "no timeout"; a recv here would
            # not return, which is why it is refused rather than exercised.
            assert socket.getsockopt(zmq.RCVTIMEO) == -1
        finally:
            socket.close(linger=0)
            context.term()

    def test_an_integral_float_is_refused_by_setsockopt_rather_than_coerced(self) -> None:
        zmq = pytest.importorskip("zmq")
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        try:
            with pytest.raises(TypeError):
                socket.setsockopt(zmq.RCVTIMEO, 15000.0)
        finally:
            socket.close(linger=0)
            context.term()


def _policies_root() -> pathlib.Path:
    """The policies package, derived from a symbol rather than a path literal."""
    return pathlib.Path(inspect.getfile(Gr00tInferenceClient)).resolve().parent.parent


def _budget_taking_constructors(source: str) -> dict[str, bool]:
    """Map public class -> whether its ``__init__`` takes and guards ``timeout_ms``."""
    found: dict[str, bool] = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
            continue
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.FunctionDef) or child.name != "__init__":
                continue
            args = [a.arg for a in child.args.args + child.args.kwonlyargs]
            if "timeout_ms" not in args:
                continue
            guarded = any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "positive_count_error"
                for call in ast.walk(child)
            )
            found[node.name] = guarded
    return found


class TestNoBudgetSurfaceDrifts:
    """Every public constructor taking ``timeout_ms`` routes through the rule.

    A fourth sidecar client added later fails here until it does, which is the
    guard the three agreeing-but-unvalidated copies did not have.
    """

    def _scan(self) -> dict[str, bool]:
        found: dict[str, bool] = {}
        for path in sorted(_policies_root().rglob("*.py")):
            found.update(_budget_taking_constructors(path.read_text(encoding="utf-8")))
        return found

    def test_the_known_surfaces_are_the_ones_found(self) -> None:
        """Non-vacuity: a scan rooted elsewhere cannot report a clean sweep."""
        assert set(self._scan()) == {"Gr00tInferenceClient", "MoveIt2InferenceClient", "MoveIt2Policy"}

    def test_every_found_surface_guards_the_budget(self) -> None:
        adrift = sorted(name for name, guarded in self._scan().items() if not guarded)
        assert adrift == [], f"these constructors take timeout_ms without the shared domain: {adrift}"

    def test_the_scanner_detects_an_unguarded_constructor(self) -> None:
        """Meta: an empty result must mean clean sources, not a blind scanner."""
        planted = "class Sidecar:\n    def __init__(self, timeout_ms: int = 1) -> None:\n        self.t = timeout_ms\n"
        assert _budget_taking_constructors(planted) == {"Sidecar": False}
