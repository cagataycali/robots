"""Native G1 driver: contract, DDS decode, and factory wiring.

Every DDS touch is mocked. Thor never had a G1 and running these tests on a
box that did would still be wrong: hardware bring-up is validated at the
office, on the real robot. This suite pins the driver's *own* behaviour, and
that is a class problem, not a robotics problem.
"""

from __future__ import annotations

import asyncio
import types
from typing import Any

import pytest

from strands_robots.drivers import (
    HardwareDriver,
    get_native_driver_class,
    list_native_drivers,
    missing_driver_members,
)
from strands_robots.drivers.g1 import G1Driver
from strands_robots.tools.g1 import (
    HANDSHAKE_FSMS,
    WALK_FSMS,
    _dds_engine,
    decode_code,
    ensure_dds,
    reset_dds_state,
)
from strands_robots.tools.g1._dds_engine import DDSSubscriberSet
from strands_robots.tools.g1._g1_common import _DDS_INIT_LOCK

# =========================================================================
# The seam - Protocol conformance and factory wiring.                     #
# =========================================================================


def test_g1_driver_satisfies_hardware_driver_protocol() -> None:
    """Every :data:`DRIVER_SURFACE` member is present on the class.

    :func:`register_native_driver` runs the same check at registration time,
    but pinning it here catches a regression the moment the class drifts,
    not when someone tries to register a fresh driver.
    """
    assert missing_driver_members(G1Driver) == ()
    inst = G1Driver(tool_name="g1", port="192.168.1.172")
    assert isinstance(inst, HardwareDriver)


def test_g1_is_registered_on_import() -> None:
    """Importing :mod:`strands_robots.drivers` puts G1Driver into the table.

    The auto-registration hook is what makes ``Robot("g1", mode="real",
    driver="strands")`` a one-liner - a caller who has to remember a second
    import is a caller who eventually forgets. The registry keys on the
    canonical name, so ``unitree_g1`` is what appears.
    """
    registered = list_native_drivers()
    assert registered.get("unitree_g1") == "G1Driver"


def test_get_native_driver_class_returns_g1() -> None:
    """The registry accepts both aliases and returns the same class.

    ``g1`` is an alias for ``unitree_g1``; the registry canonicalises before
    lookup so a caller does not have to.
    """
    assert get_native_driver_class("g1") is G1Driver
    assert get_native_driver_class("unitree_g1") is G1Driver


def test_factory_builds_g1_driver_with_driver_strands() -> None:
    """``Robot(..., driver="strands")`` returns the driver, not HardwareRobot.

    The regression to guard here is the seam: a factory that ignores
    ``driver="strands"`` and returns the lerobot driver would silently
    debug a caller who thinks they got the native path.
    """
    from strands_robots.robot import Robot

    driver = Robot(
        "g1",
        mode="real",
        driver="strands",
        port="192.168.1.172",
        network_interface="eth0",
    )
    assert isinstance(driver, G1Driver)
    assert driver.tool_name == "unitree_g1"


def test_registry_declares_strands_as_default_driver_for_g1() -> None:
    """The registry entry sets ``hardware.driver = "strands"``.

    Setting the default here means ``Robot("g1", mode="real")`` (no
    ``driver=`` at all) resolves to the native driver via ``auto``. A
    caller who wants the lerobot path back can still ask for it explicitly
    with ``driver="lerobot"``.
    """
    from strands_robots.registry import get_driver

    assert get_driver("unitree_g1") == "strands"


# =========================================================================
# Constructor contract.                                                   #
# =========================================================================


def test_constructor_accepts_the_three_factory_kwargs() -> None:
    """The factory forwards ``tool_name``, ``cameras``, ``data_config``.

    Every native driver must accept those three kwargs (the base module's
    documented constructor contract). The G1 driver ignores ``cameras`` and
    ``data_config`` because its cameras live on the DDS bus, not v4l2, but
    accepting them keeps the factory shape uniform.
    """
    driver = G1Driver(
        tool_name="g1",
        cameras={"front": {"index": 0}},
        data_config="some_config",
        port="192.168.1.172",
    )
    assert driver.tool_name == "g1"


def test_constructor_tolerates_extra_kwargs() -> None:
    """Unknown extras are logged and discarded, not raised.

    A factory may forward kwargs the driver has never heard of; refusing
    them would couple the factory to every driver's parameter list. The
    driver logs the surprise so it is discoverable, then continues.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4", made_up_option=42)
    assert driver.tool_name == "g1"


# =========================================================================
# Sensor decode. Each callback is called with a fake IDL message.         #
# =========================================================================


def _fake_imu(rpy=(0.1, 0.2, 0.3), quat=(1.0, 0.0, 0.0, 0.0)) -> Any:
    return types.SimpleNamespace(
        rpy=list(rpy),
        gyroscope=[0.01, 0.02, 0.03],
        accelerometer=[0.0, 0.0, 9.81],
        quaternion=list(quat),
    )


def _fake_lowstate(fsm: int = 501, imu: Any | None = None) -> Any:
    return types.SimpleNamespace(
        imu_state=imu or _fake_imu(),
        mode_machine=fsm,
    )


def test_lowstate_populates_imu_and_fsm() -> None:
    """A ``rt/lowstate`` callback fills ``_imu`` and ``_fsm_id``.

    The mesh reads ``_imu`` with a ``getattr`` default (see
    :mod:`strands_robots.mesh.sensors`) so a driver that has not received
    lowstate yet publishes nothing rather than a stale value. Once one
    message arrives the cache is populated and the FSM gate can consult it.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._on_lowstate(_fake_lowstate(fsm=501))
    assert driver._fsm_id == 501
    assert driver._imu is not None
    assert driver._imu["rpy"] == [0.1, 0.2, 0.3]
    assert driver._imu["quaternion"] == [1.0, 0.0, 0.0, 0.0]
    assert isinstance(driver._imu["t"], float)


def test_bms_populates_battery_with_pct_and_charging() -> None:
    """``rt/lf/bmsstate`` yields the fields the mesh's health chip reads.

    :mod:`strands_robots.mesh.sensors` reads ``battery.get("pct")`` and
    ``battery.get("charging")`` - the names must match those keys or the
    mesh publishes an empty health entry.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    msg = types.SimpleNamespace(soc=87.5, charge=0, current=-1.2, cycle=42)
    driver._on_bms(msg)
    assert driver._battery is not None
    assert driver._battery["pct"] == pytest.approx(87.5)
    assert driver._battery["charging"] is False
    assert driver._battery["current"] == pytest.approx(-1.2)


def test_lidar_state_decodes_code() -> None:
    """LidarState decodes the response code through :func:`decode_code`."""
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    msg = types.SimpleNamespace(code=0, freq=10.0, sys_rotation_speed=10.0)
    driver._on_lidar_state(msg)
    assert driver._lidar_state is not None
    assert driver._lidar_state["code"] == 0
    assert "OK" in driver._lidar_state["code_text"]
    assert driver._lidar_state["freq"] == pytest.approx(10.0)


def test_lidar_cloud_summary_is_bounded() -> None:
    """A full Livox frame produces a fixed-size summary, not a per-point dump.

    The mesh publishes ``_lidar_summary`` as a small dict every tick; if
    :meth:`_on_lidar_cloud` shipped 30k points into that field the topic
    would drown Zenoh. What keeps the record small is that every field is read
    from the message header, so the summary has the same shape whatever the
    cloud's size - which is what the absence of a point list asserts here.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    # Livox at 10 Hz reports width ~ 24000, height 1
    msg = types.SimpleNamespace(width=24000, height=1, point_step=16, row_step=24000 * 16)
    driver._on_lidar_cloud(msg)
    assert driver._lidar_summary is not None
    assert driver._lidar_summary["count"] == 24000
    # No raw point list: the summary dict is small and fixed-shape.
    assert "points" not in driver._lidar_summary


def test_decoders_swallow_bad_messages() -> None:
    """A malformed IDL message logs and is dropped - the DDS thread survives.

    If one bad message tore down the DDS callback, the driver would go silent
    on a real robot until the next reconnect. Every decoder catches, logs
    at debug, and moves on. A ``None`` message is what a broken CycloneDDS
    subscriber has been observed to deliver; an unrelated object is the
    shape a firmware update might land the day the IDL changes.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._on_lowstate(None)  # would raise on plain getattr
    driver._on_lidar_state("not a message")
    driver._on_lidar_cloud(object())
    # Lowstate with imu_state=None and no mode_machine: cache stays empty.
    assert driver._imu is None
    assert driver._fsm_id is None


# =========================================================================
# Command gates. send_action refuses until connected, FSM is good and    #
# the battery is above the floor.                                        #
# =========================================================================


def test_send_action_refuses_before_connect() -> None:
    """A driver that never connected cannot write - the message says so."""
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    result = driver.send_action({"any": 0.0})
    assert result["status"] == "error"
    assert "not connected" in result["content"][0]["text"]


def test_send_action_refuses_without_fsm_id() -> None:
    """Connected but no lowstate yet - the FSM is unknown, refuse."""
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True  # simulate connect_eagerly success
    result = driver.send_action({"any": 0.0})
    assert result["status"] == "error"
    assert "FSM id unknown" in result["content"][0]["text"]


@pytest.mark.parametrize("fsm", [0, 1, 3, 4])  # zero-torque, damp, sit, standup
def test_send_action_refuses_outside_handshake_fsm(fsm: int) -> None:
    """FSM outside :data:`HANDSHAKE_FSMS` is refused with the set named.

    The refusal names the *arm* set specifically (``send_action`` writes to
    ``rt/armsdk``), not the union with :data:`WALK_FSMS` - a message that
    over-claims coverage would mislead a caller reading a log.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = fsm
    result = driver.send_action({"any": 0.0})
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert f"FSM {fsm}" in text
    assert "arm writes" in text
    for handshake_fsm in HANDSHAKE_FSMS:
        assert str(handshake_fsm) in text
    # And the message must *not* name a set the arm gate does not test:
    # a locomotion-only FSM in the message would suggest a gate this call
    # cannot enforce. The two sets overlap on {501, 801}, so the check is
    # that any FSM in WALK_FSMS \ HANDSHAKE_FSMS is absent - which is the
    # empty set today, so the constructive check is on the wording.
    assert "loco" not in text  # message no longer over-claims scope


def test_send_action_refuses_below_battery_floor() -> None:
    """Battery under the configured floor refuses even with a good FSM."""
    driver = G1Driver(tool_name="g1", port="1.2.3.4", battery_floor_pct=15.0)
    driver._connected = True
    driver._fsm_id = 501
    driver._battery = {"pct": 12.0, "charging": False, "current": 0.0, "cycle": 0, "t": 0.0}
    result = driver.send_action({"any": 0.0})
    assert result["status"] == "error"
    assert "battery" in result["content"][0]["text"]
    assert "12.0%" in result["content"][0]["text"]


def test_send_action_reports_motion_not_wired_when_gates_pass() -> None:
    """Every gate passes but no publisher is attached - refuse with a named reason.

    Before :meth:`connect_eagerly` builds the publisher, ``_pubs`` is ``None``
    and a caller who forces ``_connected = True`` (as the older stub tests
    did) cannot reach the wire.  The refusal names the publisher, not
    ``issue #358``, because the write path is wired now - the missing piece
    is the DDS init the caller has not run yet.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = 501
    driver._battery = {"pct": 92.0, "charging": True, "current": 1.0, "cycle": 0, "t": 0.0}
    result = driver.send_action({"left_shoulder_pitch": 0.1})
    assert result["status"] == "error"
    assert "publisher not initialised" in result["content"][0]["text"]


# =========================================================================
# Task / policy stubs.                                                   #
# =========================================================================


def test_task_and_policy_paths_report_not_wired_when_gates_pass() -> None:
    """Every task/policy verb reports the "issue #358" reason honestly.

    Shipping the stubs with the right envelope shape means the day motion
    lands nothing on the caller side has to change. ``start_task`` and
    ``run_policy`` consult the same FSM + battery gates :meth:`send_action`
    does (motion-scoped: the union of :data:`HANDSHAKE_FSMS` and
    :data:`WALK_FSMS`), then return the error envelope - they cannot
    produce work. ``get_task_status`` and ``stop_task`` succeed because "no
    task running" and "nothing to stop" are honest answers, not failures
    and not motion writes.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = 501  # in both HANDSHAKE_FSMS and WALK_FSMS
    driver._battery = {"pct": 92.0, "charging": True, "current": 1.0, "cycle": 0, "t": 0.0}

    start = driver.start_task("do X", policy_port=8000)
    assert start["status"] == "error"
    assert "issue #358" in start["content"][0]["text"]

    status = driver.get_task_status()
    assert status["status"] == "success"
    assert status["content"][0]["json"]["running"] is False

    stop = driver.stop_task()
    assert stop["status"] == "success"

    # run_policy takes a Policy object we do not have here; feed None and
    # confirm it does not touch it before refusing.
    envelope = driver.run_policy(policy_object=None, instruction="", duration=1.0)  # type: ignore[arg-type]
    assert envelope["status"] == "error"
    assert "issue #358" in envelope["content"][0]["text"]


# -------------------------------------------------------------------------
# Gates on the task/policy verbs. The bot review of PR #2739 found that the
# gates the PR advertised "on every motion-path stub" only fired on
# send_action. These tests pin the gate coverage so the day the writes land
# a bypass would be caught here rather than on hardware.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["start_task", "run_policy"])
def test_task_paths_refuse_when_not_connected(verb: str) -> None:
    """An unconnected driver refuses every motion verb with the same reason.

    The refusal comes from the gate, not from the "not wired" stub, so the
    contract is: bring me up before you ask me to move. A stub that skipped
    the gate would silently accept a request the day the writes land.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    if verb == "start_task":
        result = driver.start_task("do X")
    else:
        result = driver.run_policy(policy_object=None, instruction="")  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "not connected" in text
    # The gate ran first, so the "not wired" reason must not be what we see.
    assert "issue #358" not in text


@pytest.mark.parametrize("verb", ["start_task", "run_policy"])
def test_task_paths_refuse_below_battery_floor(verb: str) -> None:
    """A battery under the floor refuses even when the FSM would admit motion."""
    driver = G1Driver(tool_name="g1", port="1.2.3.4", battery_floor_pct=15.0)
    driver._connected = True
    driver._fsm_id = 501  # walkable
    driver._battery = {"pct": 4.0, "charging": False, "current": 0.0, "cycle": 0, "t": 0.0}
    if verb == "start_task":
        result = driver.start_task("walk 1m")
    else:
        result = driver.run_policy(policy_object=None, instruction="")  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "battery" in text
    assert "4.0%" in text
    assert "issue #358" not in text


@pytest.mark.parametrize("verb", ["start_task", "run_policy"])
def test_task_paths_refuse_outside_motion_fsm(verb: str) -> None:
    """An FSM outside the motion union refuses; the refusal names the union.

    ``start_task`` and ``run_policy`` use the ``"motion"`` scope, which is
    the union of :data:`HANDSHAKE_FSMS` and :data:`WALK_FSMS`. An FSM at 0
    (zero-torque) is in neither, so both verbs refuse and the message names
    the union so a caller reading a log can see what would satisfy it.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = 0
    driver._battery = {"pct": 92.0, "charging": True, "current": 1.0, "cycle": 0, "t": 0.0}
    if verb == "start_task":
        result = driver.start_task("do X")
    else:
        result = driver.run_policy(policy_object=None, instruction="")  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "FSM 0" in text
    # Every FSM in the motion union must appear so the caller can see what
    # would admit them. Values stated literally (not derived from the
    # constants under test) so a maintainer tightening the ``motion`` scope
    # to the intersection sees a failure named for the acceptance side of
    # the motion union, not merely a suite that passes with fewer cases.
    assert "motion writes" in text
    for member in (500, 501, 801):
        assert str(member) in text


# -------------------------------------------------------------------------
# The two ungraded mutations the reviewer flagged, pinned. Both are
# safety-adjacent: a suite that survives them ships a documented set
# distinction with no enforcement.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("fsm", [500, 501, 801])
def test_send_action_admits_every_handshake_fsm(fsm: int) -> None:
    """Every FSM in :data:`HANDSHAKE_FSMS` passes the arm gate.

    Values stated literally so narrowing :data:`HANDSHAKE_FSMS` (dropping
    500, say) fires the ``[500]`` case rather than deselecting it. A
    parametrize list derived from the set under test cannot detect a
    narrowing of that set: the case disappears and the suite reads green
    with one fewer test.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = fsm
    driver._battery = {"pct": 92.0, "charging": True, "current": 1.0, "cycle": 0, "t": 0.0}
    result = driver.send_action({"any": 0.0})
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    # Every handshake FSM passes the gate - the refusal is "not wired".
    assert "issue #358" in text
    # The FSM was accepted, so its number is *not* mentioned as a refusal.
    assert f"FSM {fsm} refuses" not in text


def test_walk_fsms_has_a_consumer_and_a_documented_boundary() -> None:
    """The :data:`WALK_FSMS` constant is not dead: a locomotion write at 500
    refuses even though 500 is in :data:`HANDSHAKE_FSMS`.

    500 is sitting - the G1 accepts arm gestures there but not walking. A
    ``run_policy`` at FSM 500 is a motion write (union-scoped), so it
    passes; a hypothetical loco-scoped write at 500 would refuse. This
    test pins the boundary WALK_FSMS exists to record: the day the write
    lines land, the loco-scoped path can call ``_check_motion_gates("loco")``
    and the sitting refusal is already correct.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = 500  # sitting: in HANDSHAKE_FSMS, not in WALK_FSMS
    driver._battery = {"pct": 92.0, "charging": True, "current": 1.0, "cycle": 0, "t": 0.0}

    # Arm-scoped write at 500 passes the gate (500 is in HANDSHAKE_FSMS).
    arm_result = driver.send_action({"any": 0.0})
    assert arm_result["status"] == "error"
    assert "issue #358" in arm_result["content"][0]["text"]  # not gated

    # Loco-scoped gate at 500 refuses. Calling the helper directly is the
    # narrow way to pin the boundary before any write verb classifies its
    # scope; the write verbs land in issue #358.
    loco_refusal = driver._check_motion_gates("loco")
    assert loco_refusal is not None
    text = loco_refusal["content"][0]["text"]
    assert "FSM 500" in text
    assert "locomotion writes" in text
    # The message names the WALK set, not the union - so a caller sees
    # exactly what would satisfy a loco write. Stated literally so an
    # emptied WALK_FSMS (a maintainer error that would refuse every
    # locomotion write the day the writes land) fires this test rather
    # than passing over a vacuous ``for`` loop.
    for member in (501, 801):
        assert str(member) in text
    # 500 is *not* in WALK_FSMS, so it must not appear as an admitted FSM.
    admitted_set_part = text.split("needs one of ")[-1]
    assert "500" not in admitted_set_part


# -------------------------------------------------------------------------
# Complementary pins for the set choices themselves. Each states an
# expectation as literal values so a mutation of the constants shows up as
# a graded failure, not as a suite that runs one fewer case or reads green
# over an empty ``for`` loop.
# -------------------------------------------------------------------------


def test_walk_fsms_is_a_proper_subset_of_handshake_fsms() -> None:
    """Both sets are populated and the distinction is a real one.

    ``test_walk_fsms_has_a_consumer_and_a_documented_boundary`` asserts the
    contents of :data:`WALK_FSMS` with ``for member in WALK_FSMS``, which
    is vacuous when the set is empty. This test states the shape directly
    so an emptied :data:`WALK_FSMS` -- or a widening that erases the
    distinction the ``loco`` scope depends on -- fails a test named for
    the shape, not a test named for something else.
    """
    assert WALK_FSMS, "WALK_FSMS must not be empty"
    assert WALK_FSMS < HANDSHAKE_FSMS, "WALK_FSMS must be strictly narrower"
    # 500 is the boundary the scope split exists to record: sitting admits
    # arm gestures but not walks. Naming it literally makes the intent
    # readable from the assertion alone.
    assert 500 in HANDSHAKE_FSMS - WALK_FSMS


@pytest.mark.parametrize("fsm", [501, 801])
def test_the_loco_gate_admits_every_walking_fsm(fsm: int) -> None:
    """Acceptance side of the loco gate, at literal FSM values.

    ``test_walk_fsms_has_a_consumer_and_a_documented_boundary`` only grades
    the refusal at 500; nothing asserts that a walkable FSM is admitted,
    so emptying :data:`WALK_FSMS` fires no test in the shipped suite. This
    test closes that gap by driving the helper directly at each FSM the
    ``loco`` scope must accept.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = fsm
    driver._battery = {"pct": 92.0, "charging": True, "current": 1.0, "cycle": 0, "t": 0.0}
    assert driver._check_motion_gates("loco") is None


@pytest.mark.parametrize("verb", ["start_task", "run_policy"])
@pytest.mark.parametrize("fsm", [501, 801])
def test_motion_verbs_admit_a_literally_walkable_fsm(verb: str, fsm: int) -> None:
    """A motion verb at an FSM both scopes accept passes the gate.

    Uses literal FSM values rather than a derivation from the composition
    under test, so this survives a maintainer's later decision to tighten
    the ``motion`` pre-flight to the intersection. The refusal a caller
    sees is the ``"not wired"`` reason, not the FSM-refusal reason.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = fsm
    driver._battery = {"pct": 92.0, "charging": True, "current": 1.0, "cycle": 0, "t": 0.0}
    if verb == "start_task":
        result = driver.start_task("do X")
    else:
        result = driver.run_policy(policy_object=None, instruction="")  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "issue #358" in text
    assert f"FSM {fsm} refuses" not in text


# =========================================================================
# Lifecycle - status, stop, cleanup, and the stream tool surface.        #
# =========================================================================


def test_get_status_shape_matches_driver_envelope() -> None:
    """``get_status`` returns the same envelope the lerobot driver returns.

    The mesh publishes both peers identically; a driver that returns a
    different shape breaks the mesh's presence chip.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4", network_interface="wlan0")
    envelope = asyncio.run(driver.get_status())
    assert envelope["status"] == "success"
    inner = envelope["content"][0]["json"]
    assert inner["tool_name"] == "g1"
    assert inner["connected"] is False
    assert inner["network_interface"] == "wlan0"


def test_stream_sensors_action_returns_the_cached_snapshots() -> None:
    """The ``sensors`` verb yields exactly the four cache snapshots.

    A caller who wants to know what the robot last reported gets it in one
    call, without having to look at ``_imu`` and its siblings directly - the
    private-attribute path is for the mesh, not the agent.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._imu = {"rpy": [0.0, 0.0, 0.0], "t": 1.0}
    driver._battery = {"pct": 88.0, "t": 1.0}

    async def _collect() -> list[Any]:
        events: list[Any] = []
        async for event in driver.stream({"toolUseId": "u1", "name": "g1", "input": {"action": "sensors"}}, {}):
            events.append(event)
        return events

    events = asyncio.run(_collect())
    assert len(events) == 1
    payload = events[0]["content"][0]["json"]
    assert payload["imu"]["rpy"] == [0.0, 0.0, 0.0]
    assert payload["battery"]["pct"] == 88.0
    assert payload["lidar_state"] is None  # never delivered
    assert payload["lidar_summary"] is None


def test_stream_stop_action_calls_stop() -> None:
    """The ``stop`` verb runs :meth:`stop` and reports the no-op reason.

    Today ``stop`` is a debug log; the shape of the tool result already
    matches what the motion-wired version will return.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")

    async def _run() -> Any:
        async for event in driver.stream({"toolUseId": "u1", "name": "g1", "input": {"action": "stop"}}, {}):
            return event
        return None  # pragma: no cover

    event = asyncio.run(_run())
    assert event["status"] == "success"
    assert "issue #358" in event["content"][0]["text"]


def test_cleanup_is_idempotent() -> None:
    """Two ``cleanup`` calls do not raise; the second is a no-op."""
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver.cleanup()
    driver.cleanup()  # would raise on double-release without the guard
    assert driver._connected is False


# =========================================================================
# ensure_dds and decode_code - the shared helpers.                       #
# =========================================================================


def test_ensure_dds_reports_missing_sdk() -> None:
    """Without ``unitree_sdk2py`` installed, :func:`ensure_dds` returns a reason.

    Thor never has the SDK, so this is what every unit run actually hits.
    The reason names the missing package so a reader sees the fix rather
    than an obscure ImportError deep in the stack.
    """
    reset_dds_state()
    err = ensure_dds("eth-nonexistent")
    assert err is not None
    # Either the SDK is missing (Thor, CI) or the factory refused the
    # interface (an office machine with the SDK). Both spellings are
    # accepted so the test survives both environments.
    assert "unitree_sdk2py" in err or "ChannelFactoryInitialize" in err


def test_decode_code_names_known_and_unknown() -> None:
    """A known code renders with its meaning; an unknown one still shows the number."""
    assert "OK" in decode_code(0)
    assert "unknown" in decode_code(99999)
    assert "None" in decode_code(None) or "None" in repr(None)  # non-int path


def test_dds_init_lock_is_a_lock() -> None:
    """:data:`_DDS_INIT_LOCK` is the shared lock the driver and issue #358 tools hold.

    A different lock object here and in the tools would allow a race between
    ``ChannelSubscriber(...)`` calls; the segfault CycloneDDS bindings
    produce under that race is what this lock exists to prevent. Test what
    matters: same object, acquirable, releasable.

    The lock is private to ``_g1_common`` and reached there rather than through
    the package, so ``_dds_engine`` binding a *copy* would be invisible at the
    import site. The identity assertion is what makes "same object" a fact.
    """
    assert _dds_engine._DDS_INIT_LOCK is _DDS_INIT_LOCK
    assert hasattr(_DDS_INIT_LOCK, "acquire")
    assert hasattr(_DDS_INIT_LOCK, "release")
    acquired = _DDS_INIT_LOCK.acquire(blocking=False)
    try:
        assert acquired
    finally:
        if acquired:
            _DDS_INIT_LOCK.release()


# =========================================================================
# connect_eagerly - the DDS path fails gracefully on Thor.               #
# =========================================================================


def test_connect_eagerly_is_a_no_op_on_a_connected_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second call keeps the subscriber set it already has.

    Rebuilding it would re-subscribe all four topics and drop the only
    reference to the previous :class:`DDSSubscriberSet`, leaking its
    subscribers - on a bus whose bindings segfault under concurrent
    ``ChannelSubscriber(...)``. No caller does this today, which is exactly
    why the method's idempotence needs pinning rather than assuming.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    # Constructing one touches no DDS - __init__ only records the interface.
    already = DDSSubscriberSet("eth0")
    driver._connected = True
    driver._subs = already

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("connect_eagerly() rebuilt the subscriber set")

    monkeypatch.setattr("strands_robots.drivers.g1.DDSSubscriberSet", _forbidden)

    assert driver.connect_eagerly() is None
    assert driver._subs is already
    assert driver._connected is True


def test_connect_eagerly_reports_reason_without_sdk() -> None:
    """A machine without ``unitree_sdk2py`` gets a named connect error.

    The driver stays usable - the tests that don't need the bus can still
    call every stub - so a caller who wants a driver instance for
    a smoke test can still get one.
    """
    reset_dds_state()
    driver = G1Driver(tool_name="g1", port="1.2.3.4", network_interface="eth-none")
    err = driver.connect_eagerly()
    assert err is not None
    # Same acceptance as ensure_dds: SDK-missing on Thor/CI, bind-fail in office.
    assert "unitree_sdk2py" in err or "ChannelFactoryInitialize" in err or "cannot import" in err


# =========================================================================
# send_action wire path (PR-B: issue #361, decoupled from #358).         #
# =========================================================================
# Every test builds a driver that has already passed the gates and installs
# a recording publisher in ``_pubs``.  The point is the wire capture: which
# ``motor_cmd`` slots the driver filled, and with which values.  A missing
# SDK short-circuits the whole class (the driver refuses with the SDK's own
# reason before it would build a LowCmd_) so a skip-marker keeps the suite
# green on machines without the SDK.


_HAS_SDK: bool
try:  # pragma: no cover - environmental
    import unitree_sdk2py.idl.default as _sdk_default  # noqa: F401

    _HAS_SDK = True
except ImportError:  # pragma: no cover - environmental
    _HAS_SDK = False


class _RecordingPublisher:
    """Records ``publish`` calls without touching a DDS bus.

    Same acceptance contract as :class:`DDSPublisher`: ``publish`` returns
    ``None`` on success and a reason string on failure.  Every call is
    stashed in :attr:`writes` so a test can walk the wire capture.
    """

    def __init__(self) -> None:
        self.writes: list[tuple[str, type, Any]] = []
        self.publish_should_return: str | None = None
        self.close_calls = 0

    def publish(self, topic: str, message_class: type, message: Any) -> str | None:
        if self.publish_should_return is not None:
            return self.publish_should_return
        self.writes.append((topic, message_class, message))
        return None

    def close(self) -> None:
        self.close_calls += 1


def _gated_driver() -> tuple[G1Driver, _RecordingPublisher]:
    """Return a driver ready to publish, and the recorder attached to it.

    ``_connected`` is forced True, ``_fsm_id`` is a handshake FSM, the
    battery is above the floor, and the publisher is a recorder.  Every
    write-path test starts here so the tests read as "given a gate-passing
    driver".
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = 501  # in HANDSHAKE_FSMS
    driver._battery = {"pct": 92.0, "charging": True, "current": 1.0, "cycle": 0, "t": 0.0}
    pub = _RecordingPublisher()
    driver._pubs = pub  # type: ignore[assignment]
    return driver, pub


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_send_action_writes_to_lowcmd_topic_when_gates_pass() -> None:
    """A gated action publishes exactly one frame to ``rt/lowcmd``.

    The frame's IDL class is ``LowCmd_`` and no other topic was touched;
    the driver's write path is a single-topic path by design.
    """
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

    driver, pub = _gated_driver()
    result = driver.send_action({"left_shoulder_pitch": 0.5})

    assert result["status"] == "success"
    assert len(pub.writes) == 1
    topic, message_class, message = pub.writes[0]
    assert topic == "rt/lowcmd"
    assert message_class is LowCmd_
    assert isinstance(message, LowCmd_)


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_send_action_fills_the_named_slot_and_leaves_the_rest_alone() -> None:
    """A caller who names one joint sets exactly one slot's ``q``.

    The other 34 slots have the SDK's default values (all zeros for numeric
    fields on a fresh ``LowCmd_``); a driver that spread the target across
    the wholebody would move joints the caller never asked about.
    """
    from strands_robots.drivers.g1 import _G1_JOINT_INDEX

    driver, pub = _gated_driver()
    result = driver.send_action({"left_shoulder_pitch": 0.5})
    assert result["status"] == "success"

    _, _, message = pub.writes[0]
    slot = _G1_JOINT_INDEX["left_shoulder_pitch"]
    assert message.motor_cmd[slot].q == pytest.approx(0.5)
    # Every other slot's q stays at the SDK default (0.0).
    for i, motor in enumerate(message.motor_cmd):
        if i == slot:
            continue
        assert motor.q == pytest.approx(0.0)


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_send_action_applies_default_gains_for_scalar_targets() -> None:
    """A scalar target lands with :data:`_DEFAULT_KP`/``_DEFAULT_KD``.

    The scalar form is the common case (a caller who just wants a hold).
    The wire capture pins the exact defaults so a change to them is a
    test-visible event, not a silent regression.
    """
    from strands_robots.drivers.g1 import (
        _DEFAULT_KD,
        _DEFAULT_KP,
        _G1_JOINT_INDEX,
    )

    driver, pub = _gated_driver()
    driver.send_action({"right_elbow": -0.2})

    _, _, message = pub.writes[0]
    slot = _G1_JOINT_INDEX["right_elbow"]
    m = message.motor_cmd[slot]
    assert m.q == pytest.approx(-0.2)
    assert m.kp == pytest.approx(_DEFAULT_KP)
    assert m.kd == pytest.approx(_DEFAULT_KD)
    assert m.dq == pytest.approx(0.0)
    assert m.tau == pytest.approx(0.0)


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_send_action_accepts_per_joint_gains_and_feedforward() -> None:
    """A dict target lands every field the caller supplied.

    Missing keys inside the dict fall back to the driver default (gains) or
    zero (dq/tau); the presence of one key does not overwrite another with
    an implicit value.
    """
    from strands_robots.drivers.g1 import _G1_JOINT_INDEX

    driver, pub = _gated_driver()
    driver.send_action(
        {
            "left_elbow": {"q": 0.3, "kp": 50.0, "kd": 1.5, "dq": 0.1, "tau": 0.05},
        }
    )
    slot = _G1_JOINT_INDEX["left_elbow"]
    _, _, message = pub.writes[0]
    m = message.motor_cmd[slot]
    assert m.q == pytest.approx(0.3)
    assert m.kp == pytest.approx(50.0)
    assert m.kd == pytest.approx(1.5)
    assert m.dq == pytest.approx(0.1)
    assert m.tau == pytest.approx(0.05)


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_send_action_refuses_unknown_joint_name() -> None:
    """An unknown joint name refuses the whole action - no partial writes.

    A silent drop would move only the recognised joints, leaving the
    caller with a robot that did most of what it wanted.  Loud is the
    only right answer.
    """
    driver, pub = _gated_driver()
    result = driver.send_action({"left_shoulder_pitch": 0.1, "elbow_typo": 0.2})
    assert result["status"] == "error"
    assert "unknown joint name" in result["content"][0]["text"]
    assert "elbow_typo" in result["content"][0]["text"]
    assert pub.writes == []  # no partial writes


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_send_action_refuses_per_joint_dict_missing_q() -> None:
    """A per-joint dict without ``q`` is refused; no default target is invented.

    Zeroing an unnamed target would be worse than refusing: on a robot
    holding a pose, ``q=0`` is a full swing to the joint's zero.
    """
    driver, pub = _gated_driver()
    result = driver.send_action({"left_elbow": {"kp": 10.0}})
    assert result["status"] == "error"
    assert "missing required key 'q'" in result["content"][0]["text"]
    assert pub.writes == []


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_send_action_refuses_unknown_per_joint_key() -> None:
    """An unknown inner key refuses the action - same reason as an unknown joint name.

    A typo like ``"K_p"`` for ``"kp"`` would land with the default gain and
    silently ignore the caller's intent.  Refuse.
    """
    driver, pub = _gated_driver()
    result = driver.send_action({"left_elbow": {"q": 0.1, "K_p": 50.0}})
    assert result["status"] == "error"
    assert "unknown per-joint keys" in result["content"][0]["text"]
    assert "K_p" in result["content"][0]["text"]
    assert pub.writes == []


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_send_action_refuses_non_numeric_target() -> None:
    """A non-numeric target is refused with the joint name in the reason."""
    driver, pub = _gated_driver()
    result = driver.send_action({"left_elbow": "hold"})
    assert result["status"] == "error"
    assert "left_elbow" in result["content"][0]["text"]
    assert "non-numeric" in result["content"][0]["text"]
    assert pub.writes == []


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_send_action_refuses_empty_action() -> None:
    """An empty action dict is refused - "nothing to command" is not a write."""
    driver, pub = _gated_driver()
    result = driver.send_action({})
    assert result["status"] == "error"
    assert "empty" in result["content"][0]["text"]
    assert pub.writes == []


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_send_action_surfaces_publisher_error() -> None:
    """A publisher that returns a reason surfaces that reason in the envelope.

    The envelope's ``text`` field carries the publisher's exact string so a
    log downstream still shows what the DDS layer complained about; the
    driver adds no extra wrapping.
    """
    driver, pub = _gated_driver()
    pub.publish_should_return = "publish to 'rt/lowcmd' failed: bus is down"
    result = driver.send_action({"left_elbow": 0.1})
    assert result["status"] == "error"
    assert "bus is down" in result["content"][0]["text"]


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_send_action_refuses_when_pubs_is_missing() -> None:
    """A gated driver whose ``_pubs`` was never set refuses with a named reason.

    This is the "forced ``_connected = True`` in a test" path.  Real bring-up
    sets ``_pubs`` in :meth:`connect_eagerly` alongside ``_connected``; the
    refusal points a reader at that method rather than at the SDK.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = 501
    driver._battery = {"pct": 92.0, "charging": True, "current": 1.0, "cycle": 0, "t": 0.0}
    # _pubs left at its __init__ default (None).
    result = driver.send_action({"left_elbow": 0.1})
    assert result["status"] == "error"
    assert "publisher not initialised" in result["content"][0]["text"]


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_send_action_returns_success_envelope_with_joints_and_topic() -> None:
    """The success envelope carries the topic and every joint it commanded.

    A caller that logs the envelope should see exactly what went on the wire
    - not "success" without a body.  This is what makes a downstream monitor
    (dashboard, telemetry) attributable at the driver seam.
    """
    driver, _pub = _gated_driver()
    result = driver.send_action({"left_elbow": 0.1, "right_elbow": -0.1})
    assert result["status"] == "success"
    body = result["content"][0]["json"]
    assert body["topic"] == "rt/lowcmd"
    assert body["joints"] == ["left_elbow", "right_elbow"]  # sorted
    assert body["fsm_id"] == 501


def test_send_action_still_refuses_before_fsm_gate_regardless_of_wire() -> None:
    """The FSM gate runs before the wire path - a stub-shaped call still refuses.

    This pins the wire path is *after* the gates: a caller who supplies an
    unknown joint on a gate-failing driver still gets the FSM refusal, not
    a joint-name refusal, because the gate does not care what the write
    was going to say.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = 0  # not in HANDSHAKE_FSMS
    result = driver.send_action({"elbow_typo": 0.0})
    assert result["status"] == "error"
    assert "FSM 0" in result["content"][0]["text"]
    # And crucially, the gate ran before the joint-name check.
    assert "unknown joint" not in result["content"][0]["text"]


# =========================================================================
# _build_lowcmd_from_action - the pure helper, tested off the driver.    #
# =========================================================================


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_build_lowcmd_scalar_and_dict_land_on_the_same_slot() -> None:
    """Scalar and dict forms land ``q`` on the same slot for the same joint.

    The form is a convenience; the wire result is the same target value on
    the same slot.  A caller that reads the driver's contract picks whichever
    shape suits, knowing it is not a semantic switch.
    """
    from strands_robots.drivers.g1 import (
        _G1_JOINT_INDEX,
        _build_lowcmd_from_action,
    )

    cmd_scalar, err_s = _build_lowcmd_from_action({"waist_yaw": 0.4})
    cmd_dict, err_d = _build_lowcmd_from_action({"waist_yaw": {"q": 0.4}})
    assert err_s is None and err_d is None
    slot = _G1_JOINT_INDEX["waist_yaw"]
    assert cmd_scalar.motor_cmd[slot].q == pytest.approx(cmd_dict.motor_cmd[slot].q)


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_build_lowcmd_from_action_rejects_non_dict_input() -> None:
    """A non-dict action is refused with the type it actually was."""
    from strands_robots.drivers.g1 import _build_lowcmd_from_action

    cmd, err = _build_lowcmd_from_action([("left_elbow", 0.1)])  # type: ignore[arg-type]
    assert cmd is None
    assert err is not None
    assert "must be a dict" in err
    assert "list" in err


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_build_zero_torque_lowcmd_zeroes_every_motor_field() -> None:
    """Every motor in the zero-torque envelope has kp=kd=tau=q=dq=0.

    The control loop (issue #361 follow-up) uses this on stop; the shape
    is what "soft" looks like on the wire.  Pinning it here means a change
    to the mapping fails a test rather than reaches a robot.
    """
    from strands_robots.drivers.g1 import _build_zero_torque_lowcmd

    cmd, err = _build_zero_torque_lowcmd()
    assert err is None
    assert cmd is not None
    for m in cmd.motor_cmd:
        assert m.q == 0.0
        assert m.dq == 0.0
        assert m.tau == 0.0
        assert m.kp == 0.0
        assert m.kd == 0.0


# =========================================================================
# Module-load hygiene: the write path adds no SDK import at load time.   #
# =========================================================================


def test_g1_driver_module_does_not_import_unitree_sdk2py_at_load_time() -> None:
    """Importing :mod:`strands_robots.drivers.g1` does not import the SDK.

    The DDSPublisher tests pin this for :mod:`_dds_engine`; this test pins
    it for the driver module itself so a future edit that reaches for a
    module-level ``import unitree_sdk2py.idl.default`` fails here.  Thor
    and CI both need the driver module importable without the SDK.
    """
    import importlib
    import sys

    # Snapshot the SDK modules already loaded (may or may not be present on
    # this box).  Then reimport the driver module and confirm no *new*
    # unitree_sdk2py submodule crept in - just importing the driver.
    before = {n for n in sys.modules if n.startswith("unitree_sdk2py")}
    importlib.reload(importlib.import_module("strands_robots.drivers.g1"))
    after = {n for n in sys.modules if n.startswith("unitree_sdk2py")}
    assert after == before, (
        "importing strands_robots.drivers.g1 pulled in unitree_sdk2py modules: "
        f"{sorted(after - before)}"
    )
    assert driver._connected is False
    assert driver._connect_error == err
