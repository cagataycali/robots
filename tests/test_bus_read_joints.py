"""A broken camera must not be able to hide an arm's joints.

Measured on real hardware: an arm had published ZERO joints for eleven hours while its
mesh presence showed it connected and non-stale. The only trace was one line five
seconds after it started - `state probe 'hw_joints' failed, that section of the snapshot is
omitted (further failures logged at debug): RuntimeError('OpenCVCamera(1) read failed')` -
and then silence by design.

Cause, read in lerobot's source: SOFollower.get_observation() sync-reads the motors FIRST,
then loops the cameras calling read_latest(). A camera that raises throws away the joint
positions already in hand. Everything joints-shaped went through that one call - the fleet
snapshot, the history traces, motion detection and teleop's publisher - so one dead USB
camera silently disarmed the whole arm.
"""

from __future__ import annotations

import threading

import pytest

from strands_robots import bus_access
from strands_robots.bus_access import bus_lock, read_joints


class _Bus:
    def __init__(self, positions: dict[str, float], *, accepts_retry: bool = True) -> None:
        self.positions = positions
        self.accepts_retry = accepts_retry
        self.calls: list[dict] = []
        self.locked_during_read: bool | None = None
        self.owner: object | None = None

    def sync_read(self, register: str, num_retry: int | None = None):
        if num_retry is not None and not self.accepts_retry:
            raise TypeError("sync_read() got an unexpected keyword argument 'num_retry'")
        self.calls.append({"register": register, "num_retry": num_retry})
        if self.owner is not None:
            # The lock must be HELD while the wire is in use, or this reintroduces
            # the multi-reader collision bus_access exists to prevent. Probed from
            # ANOTHER thread, because the lock is an RLock: the reading thread can
            # always re-acquire it, so asking here would prove nothing.
            lock = bus_lock(self.owner)
            result: list[bool] = []

            def probe() -> None:
                got = lock.acquire(blocking=False)
                result.append(got)
                if got:
                    lock.release()

            t = threading.Thread(target=probe)
            t.start()
            t.join(2)
            self.locked_during_read = result == [False]
        return dict(self.positions)


class _Cfg:
    def __init__(self, retries: int | None = 3) -> None:
        if retries is not None:
            self.num_read_retries = retries


class _Arm:
    """An arm whose camera is dead - exactly arm-1's state for eleven hours."""

    def __init__(self, bus: _Bus | None = None, *, retries: int | None = 3) -> None:
        self.bus = bus
        self.config = _Cfg(retries)
        self.is_connected = True
        self.observation_calls = 0

    def get_observation(self):
        self.observation_calls += 1
        raise RuntimeError("OpenCVCamera(1) read failed (status=False).")


POSITIONS = {
    "shoulder_pan": 1.0,
    "shoulder_lift": -46.0,
    "elbow_flex": 92.0,
    "wrist_flex": 3.0,
    "wrist_roll": 170.0,
    "gripper": 2.0,
}


class TestJointsSurviveADeadCamera:
    def test_the_incident_joints_are_returned_while_get_observation_raises(self) -> None:
        arm = _Arm(_Bus(POSITIONS))
        joints = read_joints(arm)
        assert joints == {f"{m}.pos": v for m, v in POSITIONS.items()}
        assert arm.observation_calls == 0, "the camera path must not be touched at all"

    def test_the_old_path_really_did_lose_them(self) -> None:
        # Pins the bug itself: through get_observation there are no joints to have.
        from strands_robots.bus_access import read_observation

        with pytest.raises(RuntimeError, match="OpenCVCamera"):
            read_observation(_Arm(_Bus(POSITIONS)))

    def test_the_shape_matches_get_observations_joint_half(self) -> None:
        # Callers (positions_from_observation, the hw_joints probe) parse ".pos".
        joints = read_joints(_Arm(_Bus(POSITIONS)))
        assert all(k.endswith(".pos") for k in joints)
        assert "wrist_roll.pos" in joints


class TestItStillSharesTheBusLock:
    def test_the_read_happens_under_the_device_lock(self) -> None:
        bus = _Bus(POSITIONS)
        arm = _Arm(bus)
        bus.owner = arm
        read_joints(arm)
        assert bus.locked_during_read is True

    def test_a_thread_holding_the_lock_blocks_the_read(self) -> None:
        arm = _Arm(_Bus(POSITIONS))
        started = threading.Event()
        done = threading.Event()

        def reader() -> None:
            started.set()
            read_joints(arm)
            done.set()

        with bus_lock(arm):
            t = threading.Thread(target=reader, daemon=True)
            t.start()
            started.wait(1)
            assert not done.wait(0.2), "read_joints must wait its turn on the wire"
        assert done.wait(2)


class TestDriverVariations:
    def test_the_configured_retry_count_is_passed_through(self) -> None:
        bus = _Bus(POSITIONS)
        read_joints(_Arm(bus, retries=5))
        assert bus.calls[-1] == {"register": "Present_Position", "num_retry": 5}

    def test_a_bus_without_the_retry_keyword_is_still_read(self) -> None:
        bus = _Bus(POSITIONS, accepts_retry=False)
        joints = read_joints(_Arm(bus))
        assert joints["gripper.pos"] == 2.0
        assert bus.calls[-1]["num_retry"] is None

    def test_a_driver_with_no_num_read_retries_is_read_plainly(self) -> None:
        bus = _Bus(POSITIONS)
        read_joints(_Arm(bus, retries=None))
        assert bus.calls[-1]["num_retry"] is None

    def test_a_driver_with_no_bus_falls_back_to_the_full_observation(self) -> None:
        # A driver whose only reader is get_observation cannot be asked for less;
        # returning nothing would be worse than returning frames we ignore.
        class _NoBus:
            def __init__(self) -> None:
                self.reads = 0

            def get_observation(self):
                self.reads += 1
                return {"shoulder_pan.pos": 1.0, "top": object()}

        dev = _NoBus()
        obs = read_joints(dev)
        assert dev.reads == 1
        assert obs["shoulder_pan.pos"] == 1.0

    def test_a_bus_answering_with_a_non_mapping_takes_the_observation_fallback(self) -> None:
        """A ``bus`` that is not a motor bus is detected, not trusted.

        Handing the non-mapping straight back was worse than useless: this
        function documents a mapping and every joints consumer iterates it, so
        the caller failed on ``.items()`` a frame later with nothing naming the
        cause. Measured on the state probe, which swallowed the resulting
        ``TypeError`` and published no state message at all -- so the fleet saw
        a peer that had gone quiet rather than one with an unreadable bus.
        ``bus`` is an attribute name a wrapper or a proxy can also use, so this
        is reachable without a mock.
        """

        class _Weird(_Bus):
            def sync_read(self, register: str, num_retry: int | None = None):
                return ["not", "a", "mapping"]

        class _WeirdBusArm:
            def __init__(self) -> None:
                self.bus = _Weird({})
                self.config = _Cfg(3)
                self.is_connected = True
                self.observation_calls = 0

            def get_observation(self):
                self.observation_calls += 1
                return {"shoulder_pan.pos": 1.0, "top": object()}

        dev = _WeirdBusArm()
        out = read_joints(dev)

        assert isinstance(out, dict), "a non-mapping bus answer must not reach the caller"
        assert out["shoulder_pan.pos"] == 1.0
        assert dev.observation_calls == 1, "the fallback observation is what produced the reading"

    def test_the_fallback_reports_the_drivers_own_error_rather_than_a_non_mapping(self) -> None:
        """The fallback is a route to the reading, not a way to swallow a failure.

        A device with an unreadable bus AND a raising ``get_observation()`` has
        nothing to give, and the honest answer is the driver's own exception --
        which is what a caller can act on. Returning the non-mapping instead
        reported success and failed later somewhere else.
        """

        class _Weird(_Bus):
            def sync_read(self, register: str, num_retry: int | None = None):
                return ["not", "a", "mapping"]

        arm = _Arm(_Weird({}))  # its get_observation() raises the dead-camera error

        with pytest.raises(RuntimeError, match="OpenCVCamera"):
            read_joints(arm)


# ---------------------------------------------------------------------------
# A port left marked in-use by an exchange that never finished
# ---------------------------------------------------------------------------

_BUSY = "Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6] after 3 tries. [TxRxResult] Port is in use!"


class _Handler:
    def __init__(self, is_using: bool = True) -> None:
        self.is_using = is_using


class _StuckBus:
    """A bus whose first read fails port-busy, and whose later reads depend on the flag."""

    def __init__(self, handler: object | None, heals: bool = True) -> None:
        self.port_handler = handler
        self.port = "/dev/cu.usbmodemTEST"
        self.heals = heals
        self.calls = 0

    def sync_read(self, register, num_retry=0):  # noqa: ANN001, ANN201
        self.calls += 1
        if self.calls == 1 or not self.heals:
            raise ConnectionError(_BUSY)
        return {"shoulder_pan": 1.0, "wrist_roll": 170.0}


class _Device:
    def __init__(self, bus: object) -> None:
        self.bus = bus
        self.name = "so101-follower"


def test_a_stale_in_use_flag_is_cleared_once_and_the_joints_come_back() -> None:
    """The measured case: the flag outlives the exchange that set it, so nothing else recovers.

    The vendored SDK sets ``port.is_using = True`` inside ``txPacket`` and clears it with a bare
    assignment AFTER the call, with no try/finally, so an exception in between strands the flag for
    the life of the process and every later read fails before the port is even touched. Both arms
    on a real rig sat mute for hours that way with healthy presence.
    """
    handler = _Handler(is_using=True)
    bus = _StuckBus(handler)
    out = read_joints(_Device(bus))
    assert out == {"shoulder_pan.pos": 1.0, "wrist_roll.pos": 170.0}
    assert handler.is_using is False, "the stale flag must be cleared, not worked around"
    assert bus.calls == 2, "exactly one retry - recovery is not a retry loop"


def test_a_port_busy_again_right_after_clearing_is_reported_as_a_REAL_owner() -> None:
    """The case that must never be smoothed over.

    We only clear the flag because holding :func:`bus_lock` proves no reader of ours is mid-exchange.
    If the very next read is busy again, that proof has failed: something outside this module owns the
    port - another process, or a reader that skips the lock. Clearing repeatedly would interleave two
    conversations on one UART and hand back positions that were never measured, so the error names the
    situation and the command that identifies the holder.
    """
    bus = _StuckBus(_Handler(is_using=True), heals=False)
    with pytest.raises(ConnectionError) as caught:
        read_joints(_Device(bus))
    msg = str(caught.value)
    assert "REAL owner" in msg
    assert "lsof /dev/cu.usbmodemTEST" in msg, "name the port, so the operator can ask the OS"
    assert bus.calls == 2, "one clear, one retry, then the truth - never a clearing loop"


def test_an_unrelated_read_failure_never_touches_the_flag() -> None:
    """Only the port-busy signature is recoverable; everything else propagates untouched."""

    class _Broken(_StuckBus):
        def sync_read(self, register, num_retry=0):  # noqa: ANN001, ANN201
            self.calls += 1
            raise ConnectionError("Failed to sync read 'Present_Position' ... after 3 tries.")

    handler = _Handler(is_using=True)
    bus = _Broken(handler)
    with pytest.raises(ConnectionError):
        read_joints(_Device(bus))
    assert handler.is_using is True, "a no-response failure is not a stale flag - leave it alone"
    assert bus.calls == 1, "no retry for a failure we cannot explain"


def test_a_bus_with_no_port_handler_reports_the_original_error() -> None:
    """A stub, a sim or a future driver must not see an invented repair."""
    bus = _StuckBus(None, heals=False)
    with pytest.raises(ConnectionError) as caught:
        read_joints(_Device(bus))
    assert "Port is in use!" in str(caught.value), "the driver's own error, unchanged"
    assert "REAL owner" not in str(caught.value), "we cannot claim a proof we never made"


def test_clear_stale_port_busy_does_not_touch_a_flag_that_is_already_clear() -> None:
    handler = _Handler(is_using=False)
    assert bus_access.clear_stale_port_busy(_StuckBus(handler)) is False
    assert bus_access.clear_stale_port_busy(_StuckBus(None)) is False


class _ObsDevice:
    """A driver whose only reader is get_observation (no direct bus read)."""

    def __init__(self, bus: object, heals: bool = True) -> None:
        self.bus = bus
        self.name = "so101-leader"
        self.heals = heals
        self.calls = 0

    def get_observation(self):  # noqa: ANN201
        self.calls += 1
        if self.calls == 1 or not self.heals:
            raise ConnectionError(_BUSY)
        return {"shoulder_pan.pos": 2.0}

    def send_action(self, action):  # noqa: ANN001, ANN201
        raise ConnectionError(_BUSY)


def test_the_other_read_path_recovers_too_so_a_stale_flag_cannot_mute_one_caller_only() -> None:
    """read_observation is the path a driver with no exposed bus must use - it gets the same cure.

    A stranded flag is a property of the PORT, not of one call site, so recovering it in read_joints
    alone would leave every get_observation-only driver mute for the life of the process.
    """
    handler = _Handler(is_using=True)
    dev = _ObsDevice(_StuckBus(handler))
    observed = bus_access.read_observation(dev)
    assert observed == {"shoulder_pan.pos": 2.0}
    assert handler.is_using is False
    assert dev.calls == 2


def test_a_write_REFUSES_and_never_clears_the_flag_itself() -> None:
    """The asymmetry is deliberate: a read may clear it, a write may not.

    The exchange that stranded the flag may itself have been a write, so the arm's commanded target is
    unknown - and the answer to an unknown commanded state is to READ the arm, never to send it
    another target. Re-sending is motion, and a stale teleop frame replayed as a fresh command is
    exactly how an arm jumps. So the write explains itself and points at the read that recovers the
    port (the state probe does one every cycle, so telemetry alone fixes it within about a second).
    """
    handler = _Handler(is_using=True)
    dev = _ObsDevice(_StuckBus(handler))
    with pytest.raises(ConnectionError) as caught:
        bus_access.write_action(dev, {"shoulder_pan.pos": 5.0})
    msg = str(caught.value)
    assert handler.is_using is True, "a write must NOT clear the flag - only a read may"
    assert "refusing to re-send this action" in msg
    assert "READ clears that flag" in msg, "the refusal must name the way out"
    assert "commanded position is unknown" in msg, "say why re-sending is not safe"
    assert "Port is in use!" in msg, "the driver's original error stays visible"


def test_write_refusal_names_the_device_and_keeps_the_original_error() -> None:
    text = bus_access.write_refusal(_ObsDevice(_StuckBus(_Handler())), ConnectionError(_BUSY))
    assert text.startswith("so101-leader:")
    assert "Present_Position" in text


def test_a_silent_recovery_is_counted_per_port_so_a_failing_cable_cannot_hide() -> None:
    """The cure is silent, and that is the danger: a flag strands when an exchange dies mid-word.

    The usual reasons are physical - a marginal USB cable, a hub browning out, a connector working
    loose as the arm moves - so an arm that heals itself every cycle would hide a degrading rig behind
    healthy-looking joints until the recovery itself failed. Once is a hiccup; dozens is hardware to
    replace, and only a count can tell those apart.
    """
    handler = _Handler(is_using=True)
    bus = _StuckBus(handler)
    bus.port = "/dev/cu.usbmodemCOUNT1"
    dev = _Device(bus)
    before = bus_access.recovery_count(dev)
    read_joints(dev)
    assert bus_access.recovery_count(dev) == before + 1

    # A second stranding on the SAME port accumulates - that rising number is the evidence.
    handler.is_using = True
    bus.calls = 0
    read_joints(dev)
    assert bus_access.recovery_count(dev) == before + 2

    # A different port keeps its own score: one bad cable must not smear across a healthy arm.
    other_bus = _StuckBus(_Handler(is_using=True))
    other_bus.port = "/dev/cu.usbmodemCOUNT2"
    other = _Device(other_bus)
    assert bus_access.recovery_count(other) == 0
    read_joints(other)
    assert bus_access.recovery_count(other) == 1
    assert bus_access.recovery_count(dev) == before + 2, "the other port's count is untouched"


def test_a_read_that_never_stranded_reports_zero_recoveries() -> None:
    """Zero must mean zero: the count only rises on a flag we actually cleared."""

    class _Fine:
        bus = None
        port = "/dev/cu.usbmodemFINE"
        name = "healthy"

        def get_observation(self):  # noqa: ANN201
            return {"shoulder_pan.pos": 0.0}

    dev = _Fine()
    bus_access.read_observation(dev)
    assert bus_access.recovery_count(dev) == 0
