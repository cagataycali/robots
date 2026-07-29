# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Only one rollout at a time may drive the motors bus.

``start_task`` / ``run_policy`` / the agent-tool ``execute`` action all guarded
concurrency with ``if self._task_state.status == TaskStatus.RUNNING``. But
``_execute_task_async`` writes ``CONNECTING`` first and only reaches ``RUNNING``
after ``_connect_robot()`` and the policy build - a motors-bus handshake plus
per-camera warmup, seconds on a real arm. Every caller arriving inside that
window read a non-running status and was admitted, so two control loops drove
one ``FeetechMotorsBus`` at once.

Measured against the real loop with a 1 s ``connect()`` and two policies tagging
their own commands::

    status during 2nd call        : connecting
    2nd run_policy verdict        : success        <- admitted
    connect() calls on one bus    : 2
    commands on the wire          : BBBB...BABABABABABABABABAB
    peak concurrent transactions  : 2              <- two loops mid-transaction
    per-tag counts                : {'B': 108, 'A': 38}

The bus is half-duplex and not thread-safe: interleaved ``sync_read`` /
``sync_write`` transactions give framing errors, or a ``Goal_Position`` write
from one policy landing between the other's read and write - two different
intents applied to one physical arm. And because both rollouts share the single
``_task_state`` slot, they overwrite each other's step count and terminal
status, so both calls report success while only one was really driving.

The fix is an admission claim taken before the bring-up window opens and
released when the rollout ends, with the check-and-claim under a lock so two
callers racing at the same instant cannot both be admitted.

No serial port is opened and no arm is commanded: a gated in-memory double
stands in for the driver, and the bring-up window is opened by the test rather
than slept through, so every assertion here is deterministic.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState, TaskStatus
from strands_robots.policies.base import Policy

#: Upper bound on any wait, so a broken contract fails instead of hanging.
DEADLINE = 10.0

#: ``gripper.pos`` value -> the policy that commanded it. Rollouts are attributed
#: by the value they write on a declared joint, so the double never receives a
#: key a real driver would reject.
TAGS: dict[float, str] = {0.11: "A", 0.22: "B"}


class GatedBus:
    """Motors-bus double whose ``connect()`` blocks until the test opens it.

    Records which rollout's commands reached the wire and the peak number of
    ``send_action`` calls in flight at once - anything above 1 means two loops
    were mid-transaction on one half-duplex port.
    """

    def __init__(self) -> None:
        self.name = self.robot_type = "gated_arm"
        self.is_calibrated = True
        self.config = type("Cfg", (), {"cameras": {}})()
        self._connected = False
        self.connect_calls = 0
        self.connect_entered = threading.Event()
        self.connect_gate = threading.Event()
        self.writers: list[str] = []
        self.max_concurrent = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = False) -> None:
        self.connect_calls += 1
        self.connect_entered.set()
        if not self.connect_gate.wait(DEADLINE):  # pragma: no cover - deadline
            raise TimeoutError("bring-up gate never opened")
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_observation(self) -> dict[str, Any]:
        return {"gripper.pos": 0.0}

    def send_action(self, action: dict[str, Any]) -> None:
        with self._lock:
            self._in_flight += 1
            self.max_concurrent = max(self.max_concurrent, self._in_flight)
            self.writers.append(TAGS[action["gripper.pos"]])
            self._in_flight -= 1


class TaggedPolicy(Policy):
    """Commands one identifying ``gripper.pos`` value every step."""

    def __init__(self, tag: str) -> None:
        self.value = next(v for v, t in TAGS.items() if t == tag)

    @property
    def provider_name(self) -> str:
        return "tagged"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        pass

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{"gripper.pos": self.value}] * 4


def make_robot(bus: GatedBus | None = None) -> HwRobot:
    """Build a Robot bypassing hardware init (the pattern used across tests/)."""
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "gated_arm"
    hw.action_horizon = 4
    hw.data_config = None
    hw.control_frequency = 500.0
    hw.action_sleep_time = 1.0 / 500.0
    hw._task_state = RobotTaskState()
    hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gated_arm_executor")
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_admission = threading.Lock()
    hw._task_claimed = False
    hw.mesh = None
    hw.peer_id = None
    hw.robot = bus if bus is not None else GatedBus()
    return hw


@pytest.fixture
def bus() -> GatedBus:
    return GatedBus()


@pytest.fixture
def hw(bus: GatedBus) -> Any:
    robot = make_robot(bus)
    yield robot
    bus.connect_gate.set()
    robot.cleanup()


def _rollout_in_bringup(hw: HwRobot, bus: GatedBus, tag: str, steps: int) -> tuple[threading.Thread, dict]:
    """Start a rollout on a thread and return once it is inside ``connect()``."""
    out: dict[str, Any] = {}
    thread = threading.Thread(
        target=lambda: out.__setitem__(
            "result",
            hw.run_policy(policy_object=TaggedPolicy(tag), instruction=f"rollout {tag}", n_steps=steps),
        ),
        daemon=True,
    )
    thread.start()
    assert bus.connect_entered.wait(DEADLINE), "rollout never reached connect()"
    assert hw._task_state.status is TaskStatus.CONNECTING
    return thread, out


class TestBringUpWindowIsExclusive:
    """The window a status-based guard could not see."""

    def test_a_second_rollout_during_bring_up_never_reaches_the_bus(self, hw: HwRobot, bus: GatedBus):
        thread, first = _rollout_in_bringup(hw, bus, "A", steps=6)

        # Bring-up is opened by a third thread rather than after the refusal, so
        # a wrongly admitted second rollout also gets past connect() and its
        # commands reach the wire. What keeps the bus single-writer here is the
        # refusal, not the test holding the gate shut.
        opener = threading.Timer(0.2, bus.connect_gate.set)
        opener.start()
        try:
            second = hw.run_policy(policy_object=TaggedPolicy("B"), instruction="rollout B", n_steps=6)
        finally:
            opener.join(DEADLINE)

        assert second["status"] == "error"
        assert "already running" in second["content"][0]["text"].lower()

        thread.join(DEADLINE)
        assert first["result"]["status"] == "success"

        # Only the admitted rollout drove the arm, it connected once, and no two
        # transactions were ever in flight together.
        assert set(bus.writers) == {"A"}
        assert bus.connect_calls == 1
        assert bus.max_concurrent == 1
        assert hw._task_state.step_count == 6

    def test_start_task_during_bring_up_is_refused_synchronously(self, hw: HwRobot, bus: GatedBus):
        thread, _ = _rollout_in_bringup(hw, bus, "A", steps=4)
        before = hw._task_state.task_future

        result = hw.start_task("rollout B", policy_port=5555)

        # Refused by the caller's own thread: start_task returns before its
        # executor job runs, so a claim taken inside the job would have reported
        # "Task started" and only then turned this caller away.
        assert result["status"] == "error"
        assert "already running" in result["content"][0]["text"].lower()
        assert hw._task_state.task_future is before

        bus.connect_gate.set()
        thread.join(DEADLINE)
        assert set(bus.writers) == {"A"}

    def test_the_execute_chokepoint_during_bring_up_is_refused(self, hw: HwRobot, bus: GatedBus):
        # The agent-tool "execute" action and the mesh execute dispatch reach
        # this method directly, bypassing run_policy/start_task entirely.
        thread, _ = _rollout_in_bringup(hw, bus, "A", steps=4)

        result = hw._execute_task_sync("rollout B", policy_port=5555)

        assert result["status"] == "error"
        assert "already running" in result["content"][0]["text"].lower()

        bus.connect_gate.set()
        thread.join(DEADLINE)
        assert set(bus.writers) == {"A"}
        assert bus.connect_calls == 1

    def test_the_refusal_names_the_rollout_that_holds_the_bus(self, hw: HwRobot, bus: GatedBus):
        hw._task_state.instruction = "a task that already finished"
        thread, _ = _rollout_in_bringup(hw, bus, "A", steps=4)

        text = hw.run_policy(policy_object=TaggedPolicy("B"), instruction="rollout B", n_steps=4)["content"][0]["text"]

        # Named from the claim, not from whatever ran last: the task state's
        # instruction is otherwise only written once bring-up is under way.
        assert "rollout A" in text
        assert "a task that already finished" not in text

        bus.connect_gate.set()
        thread.join(DEADLINE)


class TestTheClaimIsAlwaysReleased:
    """A robot that refused one task must not refuse every later one."""

    def test_a_completed_rollout_hands_the_bus_back(self, hw: HwRobot, bus: GatedBus):
        bus.connect_gate.set()

        first = hw.run_policy(policy_object=TaggedPolicy("A"), instruction="rollout A", n_steps=4)
        second = hw.run_policy(policy_object=TaggedPolicy("B"), instruction="rollout B", n_steps=4)

        assert first["status"] == "success"
        assert second["status"] == "success"
        assert bus.writers == ["A"] * 4 + ["B"] * 4

    def test_a_failed_connect_hands_the_bus_back(self, hw: HwRobot, bus: GatedBus):
        bus.connect_gate.set()
        bus.is_calibrated = False

        failed = hw.run_policy(policy_object=TaggedPolicy("A"), instruction="rollout A", n_steps=4)
        assert failed["status"] == "error"

        bus.is_calibrated = True
        assert hw.run_policy(policy_object=TaggedPolicy("B"), instruction="rollout B", n_steps=4)["status"] == "success"

    def test_a_raising_rollout_hands_the_bus_back(self, hw: HwRobot, bus: GatedBus, monkeypatch):
        bus.connect_gate.set()

        def boom(*_a: Any, **_k: Any) -> None:
            raise RuntimeError("driver exploded")

        monkeypatch.setattr(hw, "_run_control_loop", boom)
        with pytest.raises(RuntimeError, match="driver exploded"):
            hw.run_policy(policy_object=TaggedPolicy("A"), instruction="rollout A", n_steps=4)

        monkeypatch.undo()
        assert hw.run_policy(policy_object=TaggedPolicy("B"), instruction="rollout B", n_steps=4)["status"] == "success"

    def test_a_refused_duration_does_not_take_the_bus(self, hw: HwRobot, bus: GatedBus):
        bus.connect_gate.set()

        refused = hw.run_policy(policy_object=TaggedPolicy("A"), instruction="rollout A", duration=-1.0)
        assert refused["status"] == "error"
        assert "duration" in refused["content"][0]["text"]

        # The budget check is stateless, so rejecting it must not have taken the
        # bus away from a rollout that can still start.
        assert hw._task_claimed is False
        assert hw.run_policy(policy_object=TaggedPolicy("B"), instruction="rollout B", n_steps=4)["status"] == "success"

    def test_a_rollout_stopped_during_bring_up_hands_the_bus_back(self, hw: HwRobot, bus: GatedBus):
        # Where the two bring-up fixes meet. A stop latched during bring-up is
        # honored by ``_execute_task_async`` as soon as connect() returns, which
        # abandons the rollout *before* the arm is commanded - so it never
        # reaches the terminal block, and the release has to come from the
        # ``finally`` rather than from a rollout that ran to completion. Without
        # that, stopping a task during bring-up would cost the bus permanently.
        thread, stopped = _rollout_in_bringup(hw, bus, "A", steps=4)

        assert hw.stop_task()["status"] == "success"
        bus.connect_gate.set()
        thread.join(DEADLINE)

        assert hw._task_state.status is TaskStatus.STOPPED
        assert stopped["result"]["status"] == "error"
        # Abandoned during bring-up, so nothing was ever put on the wire.
        assert bus.writers == []
        assert hw._task_claimed is False

        # And the bus is genuinely usable again, not merely unclaimed.
        second = hw.run_policy(policy_object=TaggedPolicy("B"), instruction="rollout B", n_steps=4)
        assert second["status"] == "success"
        assert set(bus.writers) == {"B"}

    def test_a_failed_submit_does_not_leak_the_claim(self, hw: HwRobot, bus: GatedBus):
        hw._executor.shutdown(wait=True)

        with pytest.raises(RuntimeError):
            hw.start_task("rollout B", policy_port=5555)

        # Nothing ran to release the claim, so start_task has to unwind it
        # itself or the robot refuses every task for the rest of its life.
        assert hw._task_claimed is False


class TestAdmissionIsAtomic:
    def test_simultaneous_callers_admit_exactly_one(self, hw: HwRobot, bus: GatedBus):
        # Every thread calls in at the same instant, so a check that is not
        # atomic with the claim lets more than one through.
        callers = 8
        start = threading.Barrier(callers)
        verdicts: list[str | None] = [None] * callers

        def caller(index: int) -> None:
            start.wait(DEADLINE)
            verdicts[index] = hw._claim_task(f"rollout {index}") is None and "admitted" or "refused"

        threads = [threading.Thread(target=caller, args=(i,), daemon=True) for i in range(callers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(DEADLINE)

        assert verdicts.count("admitted") == 1
        assert verdicts.count("refused") == callers - 1
        hw._release_task()
