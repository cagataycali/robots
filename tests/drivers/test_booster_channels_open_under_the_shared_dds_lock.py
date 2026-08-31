"""Opening the Booster T1's channels holds the shared DDS lock.

``_g1_common``'s module docstring states the contract graded here: "A
``ChannelSubscriber`` and a ``ChannelPublisher`` cannot be constructed
concurrently: the CycloneDDS bindings segfault. ``_DDS_INIT_LOCK`` is the
*shared* lock the driver and the tools both hold while creating readers or
writers. One lock; two consumers."

:meth:`~strands_robots.drivers.booster.BoosterDriver.connect_eagerly` builds six
endpoints in a row - the channel factory, an RPC client, and four
subscriber/publisher channels - while
:class:`~strands_robots.tools.g1._dds_engine.DDSSubscriberSet` constructs
subscribers under that same lock on other threads in the same process
(streaming, a policy rollout, mesh telemetry). A shared lock is only a guarantee
where every caller takes it, and the caller that loses the race is the one
holding nothing. The loss is a native segfault, which no ``except`` boundary can
turn into an error envelope: the process dies, possibly while a 1.2 m biped is
standing under its own controller.

These cells grade the *behaviour* - was the shared lock held while each endpoint
was built - rather than the source, and that is what this driver specifically
needs. The package-wide source rule in
``tests/tools/g1/test_every_dds_endpoint_is_created_under_the_shared_lock.py``
derives its vocabulary of endpoint-creating operations from the Unitree
infrastructure modules, so it recognises ``Init`` and nothing else here: the
Booster SDK spells the identical operations ``InitWithName``, ``InitChannel``,
``InitChannelWithName`` and ``B1LowStateSubscriber`` / ``B1LowCmdPublisher`` /
``B1BatteryStateSubscriber`` / ``B1FallDownStateSubscriber``. Eight of this
driver's ten construction sites are therefore invisible to that rule, and a
behavioural pin cannot be evaded by a vendor's choice of name.

:func:`TestTheLockIsTheSharedOneAndNotAPrivateOne.test_the_open_waits_for_a_competing_endpoint_construction`
is the cell that makes the pin specific: a driver-private lock would satisfy
"some lock was held" while excluding nothing the engine does.

No cell here needs a T1, a DDS bus or ``booster_robotics_sdk_python``: the SDK is
a recorder staged on :mod:`sys.modules`, which is the driver's lazy-import path.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

import pytest

from strands_robots.drivers.booster import BoosterDriver
from strands_robots.tools.g1._g1_common import _DDS_INIT_LOCK

#: How long a cell waits for something that should already have happened, and
#: how long it waits to conclude that something is correctly still blocked.
_TIMEOUT_S = 5.0
_BLOCKED_S = 0.2

#: Every endpoint-creating operation the driver performs, by the name the vendor
#: gives it. ``Init`` is the only one the package-wide source rule recognises.
_UNNAMED_ROBOT_OPERATIONS = (
    "ChannelFactory.Init",
    "B1LocoClient",
    "B1LocoClient.Init",
    "B1LowStateSubscriber",
    "B1LowCmdPublisher",
    "B1BatteryStateSubscriber",
    "B1FallDownStateSubscriber",
) + ("InitChannel",) * 4
_NAMED_ROBOT_OPERATIONS = (
    "ChannelFactory.Init",
    "B1LocoClient",
    "B1LocoClient.InitWithName",
    "B1LowStateSubscriber",
    "B1LowCmdPublisher",
    "B1BatteryStateSubscriber",
    "B1FallDownStateSubscriber",
) + ("InitChannelWithName",) * 4


class _Recorder:
    """Collects ``(operation, was the shared lock held)`` in call order."""

    def __init__(self) -> None:
        self.observed: list[tuple[str, bool]] = []
        self.refuse: str | None = None

    def note(self, operation: str) -> None:
        self.observed.append((operation, _DDS_INIT_LOCK.locked()))
        if self.refuse == operation:
            raise RuntimeError(f"{operation}: channel busy")

    def held(self, operation: str) -> list[bool]:
        return [was_held for name, was_held in self.observed if name == operation]


class _FakeChannel:
    """A subscriber or publisher, recording when it was opened and closed."""

    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder

    def InitChannel(self) -> None:  # noqa: N802 - the SDK's own spelling
        self._recorder.note("InitChannel")

    def InitChannelWithName(self, robot_name: str) -> None:  # noqa: N802
        self._recorder.note("InitChannelWithName")

    def CloseChannel(self) -> None:  # noqa: N802
        self._recorder.note("CloseChannel")


class _FakeLocoClient:
    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder

    def Init(self) -> None:  # noqa: N802
        self._recorder.note("B1LocoClient.Init")

    def InitWithName(self, robot_name: str) -> None:  # noqa: N802
        self._recorder.note("B1LocoClient.InitWithName")


class _FakeSdk:
    """The construction surface of ``booster_robotics_sdk_python``, recording."""

    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder
        sdk = self

        class _Factory:
            @staticmethod
            def Instance() -> Any:  # noqa: N802
                return _Factory()

            def Init(self, domain_id: int, ip: str) -> None:  # noqa: N802
                sdk._recorder.note("ChannelFactory.Init")

        self.ChannelFactory = _Factory

    def B1LocoClient(self) -> _FakeLocoClient:  # noqa: N802
        self._recorder.note("B1LocoClient")
        return _FakeLocoClient(self._recorder)

    def B1LowStateSubscriber(self, handler: Any) -> _FakeChannel:  # noqa: N802
        self._recorder.note("B1LowStateSubscriber")
        return _FakeChannel(self._recorder)

    def B1LowCmdPublisher(self) -> _FakeChannel:  # noqa: N802
        self._recorder.note("B1LowCmdPublisher")
        return _FakeChannel(self._recorder)

    def B1BatteryStateSubscriber(self, handler: Any) -> _FakeChannel:  # noqa: N802
        self._recorder.note("B1BatteryStateSubscriber")
        return _FakeChannel(self._recorder)

    def B1FallDownStateSubscriber(self, handler: Any) -> _FakeChannel:  # noqa: N802
        self._recorder.note("B1FallDownStateSubscriber")
        return _FakeChannel(self._recorder)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Stage the recording SDK under the name the driver imports."""
    rec = _Recorder()
    monkeypatch.setitem(sys.modules, "booster_robotics_sdk_python", _FakeSdk(rec))
    return rec


class TestEveryEndpointIsBuiltUnderTheSharedLock:
    """The whole channel set is one critical section, not just the graded call."""

    @pytest.mark.parametrize(
        ("robot_name", "expected"),
        [(None, _UNNAMED_ROBOT_OPERATIONS), ("t1_left", _NAMED_ROBOT_OPERATIONS)],
    )
    def test_no_endpoint_is_constructed_with_the_lock_free(
        self, recorder: _Recorder, robot_name: str | None, expected: tuple[str, ...]
    ) -> None:
        """Both spellings of the open are graded: a named T1 takes the other branch."""
        assert BoosterDriver(robot_name=robot_name).connect_eagerly() is None

        assert [name for name, _ in recorder.observed] == list(expected), (
            "the recorder did not see the construction sequence it grades, so "
            "the assertion below would pass over an open that never happened"
        )
        unlocked = sorted({name for name, was_held in recorder.observed if not was_held})
        assert not unlocked, (
            f"these endpoints were constructed with the shared {_DDS_INIT_LOCK!r} "
            f"free, so they race every other endpoint construction in the "
            f"process and the loss is a native segfault: {unlocked}"
        )

    def test_the_lock_is_released_once_the_channels_are_open(self, recorder: _Recorder) -> None:
        """The lock is a critical section, not something the driver keeps."""
        assert BoosterDriver().connect_eagerly() is None
        assert not _DDS_INIT_LOCK.locked(), (
            "the shared lock is still held after the open returned, which would "
            "stall every later endpoint construction in the process"
        )

    def test_a_refused_channel_releases_the_lock_on_the_way_out(self, recorder: _Recorder) -> None:
        """The failure path is the one that leaks, and a leak deadlocks the process."""
        recorder.refuse = "InitChannel"
        reason = BoosterDriver().connect_eagerly()
        assert reason is not None and "did not open" in reason
        assert not _DDS_INIT_LOCK.locked(), (
            "a channel that refused to open left the shared lock held, so every "
            "later endpoint construction in the process would block forever"
        )

    def test_the_release_path_does_not_hold_the_lock(self, recorder: _Recorder) -> None:
        """Closing creates no endpoint, so it owes no serialisation.

        Pinned because the alternative is worse than redundant: a close that
        blocks would stall every endpoint construction in the process while the
        driver tidied up a set it is about to discard.
        """
        recorder.refuse = "InitChannel"
        assert BoosterDriver().connect_eagerly() is not None
        closes = recorder.held("CloseChannel")
        assert closes, "no channel was released, so this cell grades nothing"
        assert not any(closes), f"the partial channel set was released while holding the shared lock: {closes}"


class TestTheLockIsTheSharedOneAndNotAPrivateOne:
    """A driver-private lock would exclude nothing the engine does."""

    def test_the_open_waits_for_a_competing_endpoint_construction(self, recorder: _Recorder) -> None:
        opened = threading.Event()
        release = threading.Event()

        def hold_the_lock() -> None:
            with _DDS_INIT_LOCK:
                release.wait(_TIMEOUT_S)

        holder = threading.Thread(target=hold_the_lock, daemon=True)
        holder.start()
        try:
            # Wait for the competing holder to actually own the lock, so the
            # assertion below cannot pass just because the race was not run.
            while not _DDS_INIT_LOCK.locked():
                threading.Event().wait(0.01)

            def open_the_channels() -> None:
                if BoosterDriver().connect_eagerly() is None:
                    opened.set()

            threading.Thread(target=open_the_channels, daemon=True).start()
            assert not opened.wait(_BLOCKED_S), (
                "the channels opened while another thread held the shared DDS "
                "lock, so they are not serialised against the engine's endpoint "
                "construction - a private lock excludes nothing"
            )
            assert not recorder.observed, (
                "an endpoint was constructed before the competing holder let go, "
                "so the lock the driver takes is not the one being held here"
            )
        finally:
            release.set()
            holder.join(_TIMEOUT_S)
        assert opened.wait(_TIMEOUT_S), (
            "the open never completed after the shared lock was released, so it "
            "is blocked on something other than that lock"
        )
