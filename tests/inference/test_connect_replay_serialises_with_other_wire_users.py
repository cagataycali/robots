"""The connect sequence is a wire user, so it holds the lock the others hold.

``RemotePolicy._request`` documents its precondition in the section header
above it: "call while holding ``self._lock``". Four of its call sites did;
the three inside :meth:`~strands_robots.inference.client.RemotePolicy._connect`
did not, and those three run *after* ``self._ws`` is live - the handshake read
plus the replay of any config set before the connection existed. A second
thread reaching the wire in that window overlapped the read, and ``websockets``
refused it with a ``ConcurrencyError`` naming its own internals.

Two threads on one policy is the ordinary case: every policy coroutine resolves
through :mod:`strands_robots._async_utils`' reused worker thread, and the
async-RTC path in :mod:`strands_robots.simulation.policy_runner` submits
prefetch inference to its own ``rtc-prefetch`` worker while the rollout thread
carries on. The pre-existing suite could not see it - its threads run the
*server*, and the client is driven from one thread throughout.

The precondition is graded at runtime rather than by reading the source: a
witness lock records whether it was held when ``_request`` ran, which follows
the coverage ``_connect`` gets from its caller. A lexical scan would report
``_connect``'s calls as unlocked either way.
"""

import ast
import inspect
import textwrap
import threading
from typing import Any

import pytest

from strands_robots.inference import PolicyServer, RemotePolicy
from strands_robots.policies import MockPolicy

pytest.importorskip("websockets")


class ParkingPolicy(MockPolicy):  # type: ignore[misc]
    """Server-side policy that parks inside ``set_robot_state_keys``.

    Parking there holds the client's connect replay open at a known point -
    after ``self._ws`` is live, mid-``_request`` - so a second thread can be
    pointed at the same connection deterministically instead of by sleeping.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.entered = threading.Event()
        self.release = threading.Event()

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.entered.set()
        self.release.wait(timeout=10.0)
        super().set_robot_state_keys(robot_state_keys)


class WitnessLock:
    """A lock that records whether it is currently held.

    Stands in for ``RemotePolicy._lock`` so a test can assert the wire
    helper's documented precondition at the moment it runs, wherever the
    coverage came from.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.held = False

    def __enter__(self) -> "WitnessLock":
        self._lock.acquire()
        self.held = True
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.held = False
        self._lock.release()

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        return self._lock.acquire(*args, **kwargs)

    def release(self) -> None:
        self._lock.release()


@pytest.fixture
def parked() -> Any:
    """A live server whose policy parks on the connect replay, plus its client.

    The client is handed config *before* connecting, so the replay has
    something to send and the window is a real one rather than a bare
    handshake.
    """
    policy = ParkingPolicy()
    server = PolicyServer(policy=policy, host="127.0.0.1", port=0).start()
    client = RemotePolicy(host="127.0.0.1", port=server.port, connect_timeout=10.0, request_timeout=10.0)
    client.set_robot_state_keys(["shoulder_pan", "elbow_flex"])
    client.set_control_frequency(50.0)
    try:
        yield policy, server, client
    finally:
        policy.release.set()
        client.close()
        server.stop()


def _open_the_window(policy: ParkingPolicy, client: RemotePolicy) -> threading.Thread:
    """Start a connect in its own thread and return once it is parked mid-replay."""
    thread = threading.Thread(target=lambda: client.requires_images, daemon=True)
    thread.start()
    assert policy.entered.wait(timeout=10.0), "the server never reached the parked replay"
    return thread


class TestTheConnectReplayIsAWireUser:
    """Premise: there is something in the connect sequence to serialise."""

    def test_the_connect_sequence_sends_requests_of_its_own(self) -> None:
        source = inspect.getsource(RemotePolicy._connect)
        calls = [
            node
            for node in ast.walk(ast.parse(textwrap.dedent(source)))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_request"
        ]
        assert calls, "_connect no longer replays config; this contract needs re-deriving"

    def test_the_connection_is_live_while_the_replay_runs(self, parked: Any) -> None:
        policy, _server, client = parked
        thread = _open_the_window(policy, client)
        assert client._ws is not None, "the replay runs before the connection is usable"
        policy.release.set()
        thread.join(timeout=10.0)


class TestAConcurrentWireUserWaitsForTheConnectReplay:
    """Regression: a second thread is serialised, not refused."""

    @pytest.mark.parametrize("caller", ["reset", "set_robot_state_keys", "get_actions"])
    def test_a_second_thread_is_not_refused_mid_replay(self, parked: Any, caller: str) -> None:
        policy, _server, client = parked
        connecting = _open_the_window(policy, client)

        outcome: dict[str, Any] = {}

        def drive() -> None:
            try:
                if caller == "reset":
                    client.reset(seed=7)
                elif caller == "set_robot_state_keys":
                    client.set_robot_state_keys(["j0", "j1"])
                else:
                    client.get_actions_sync({"observation.state": [0.0]}, "")
                outcome["result"] = "returned"
            except BaseException as exc:  # noqa: BLE001 - the failure is the subject
                outcome["result"] = f"{type(exc).__name__}: {exc}"

        second = threading.Thread(target=drive, daemon=True)
        second.start()
        # The second caller must be waiting, not racing the replay onto the wire.
        second.join(timeout=0.5)
        assert second.is_alive(), "the second wire user did not wait for the replay"

        policy.release.set()
        second.join(timeout=10.0)
        connecting.join(timeout=10.0)
        assert outcome["result"] == "returned", outcome["result"]

    def test_the_refusal_that_used_to_surface_names_nothing_the_caller_passed(self, parked: Any) -> None:
        """The old failure was a ``websockets`` concurrency error, not a domain refusal.

        Pinned so a future change cannot re-route this window into an error
        whose text is about the transport rather than about the call.
        """
        policy, _server, client = parked
        connecting = _open_the_window(policy, client)
        errors: list[str] = []

        def drive() -> None:
            try:
                client.reset(seed=1)
            except BaseException as exc:  # noqa: BLE001 - the failure is the subject
                errors.append(f"{type(exc).__name__}: {exc}")

        second = threading.Thread(target=drive, daemon=True)
        second.start()
        policy.release.set()
        second.join(timeout=10.0)
        connecting.join(timeout=10.0)
        assert errors == [], errors


class TestTwoRacingFirstCallersOpenOneConnection:
    """Regression: the double check under the lock, not just the serialising."""

    def test_one_connection_is_opened_for_two_racing_first_callers(self, parked: Any) -> None:
        policy, _server, client = parked
        policy.release.set()
        opened: list[object] = []
        real_connect = client._connect

        def counting() -> None:
            real_connect()
            opened.append(client._ws)

        client._connect = counting  # type: ignore[method-assign]
        # A barrier so every caller reaches the guard together. The window is
        # the connect handshake itself, so staggered starts let the GIL
        # serialise them by luck and the race goes unobserved.
        racers = 8
        gate = threading.Barrier(racers)

        def race() -> None:
            gate.wait(timeout=10.0)
            client.requires_images  # noqa: B018 - the property is what connects

        threads = [threading.Thread(target=race, daemon=True) for _ in range(racers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
        assert len(opened) == 1, f"opened {len(opened)} connections for one policy"


class TestEveryWireRequestRunsUnderTheLock:
    """The documented precondition, graded at runtime over every wire path."""

    def test_no_request_reaches_the_wire_without_the_lock(self, parked: Any) -> None:
        policy, _server, client = parked
        policy.release.set()  # this one drives the paths, it does not need the window

        witness = WitnessLock()
        client._lock = witness  # type: ignore[assignment]
        seen: list[tuple[str, bool]] = []
        real_request = client._request

        def recording(message: dict[str, Any]) -> dict[str, Any]:
            seen.append((str(message.get("type")), witness.held))
            return real_request(message)

        client._request = recording  # type: ignore[method-assign]

        # Drive every public path that reaches the wire, connect included.
        assert client.requires_images in (True, False)
        client.set_robot_state_keys(["j0", "j1"])
        client.set_control_frequency(25.0)
        client.reset(seed=3)
        client.get_actions_sync({"observation.state": [0.0]}, "")

        assert seen, "no request reached the wire; this contract would pass vacuously"
        unlocked = [message_type for message_type, held in seen if not held]
        assert unlocked == [], f"reached the wire without the lock: {unlocked}"

    def test_the_connect_replay_is_among_the_paths_graded(self, parked: Any) -> None:
        """Non-vacuity: the replay's own requests are in the graded set.

        Without this, narrowing the drive above to the already-connected paths
        would still report a clean result.
        """
        policy, _server, client = parked
        policy.release.set()

        seen: list[str] = []
        real_request = client._request

        def recording(message: dict[str, Any]) -> dict[str, Any]:
            seen.append(str(message.get("type")))
            return real_request(message)

        client._request = recording  # type: ignore[method-assign]
        assert client.requires_images in (True, False)  # connect + replay only
        assert "set_state_keys" in seen, seen
        assert "set_control_frequency" in seen, seen


class TestWhatIsUnchanged:
    """Serialising the connect must not cost the behaviour around it."""

    def test_the_pending_config_still_reaches_the_server(self, parked: Any) -> None:
        policy, _server, client = parked
        policy.release.set()
        assert client.requires_images in (True, False)
        assert policy.robot_state_keys == ["shoulder_pan", "elbow_flex"]

    def test_close_still_returns_after_a_connect(self, parked: Any) -> None:
        policy, _server, client = parked
        policy.release.set()
        assert client.requires_images in (True, False)
        client.close()
        assert client._ws is None
        client.close()  # idempotent
        assert client._ws is None
