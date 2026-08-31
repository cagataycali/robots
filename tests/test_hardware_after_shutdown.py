# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A rollout can neither start nor finish successfully on a shut-down robot.

``_shutdown_event`` is one of the control loop's exit conditions::

    while (
        time.monotonic() - start_mono < duration
        and (n_steps is None or self._task_state.step_count < n_steps)
        and self._task_state.status == TaskStatus.RUNNING
        and not self._stop_requested.is_set()
        and not self._shutdown_event.is_set()   # <- set by cleanup()
    ):

but it was neither an admission check nor a terminal-status discriminator, so
once ``cleanup()`` had set it a rollout ran the whole bring-up, fell out of the
loop on the condition's first evaluation, and reported itself ``completed``.

Two producers reach that state, and they need different fixes - which is why
both are pinned here:

**A task started after ``cleanup()``.** Bring-up is not side-effect-free, so
this is refused at the entry points rather than merely relabelled. Measured on
a two-device arm, calling each entry point once after ``cleanup()``:

    entry point                    verdict on main       connect()  left open
    run_policy                     success                   1        True
    execute (agent tool / mesh)     success                   1        True
    start_task                     raises RuntimeError        0        False

- ``_connect_robot()`` re-opens the motors bus and warms every camera. Because
  ``cleanup()`` does not disconnect the robot and the executor is already shut
  down, those devices stay open for the life of the process - a second
  ``cleanup()`` does not close them either.
- ``Policy.reset()`` is called, clearing the per-episode state of a policy
  object the caller may still be driving (the documented ``policy_object=``
  reuse pattern).
- The policy is never queried and the arm is never commanded, yet two of the
  three entry points returned ``status="success"`` with ``steps: 0`` - a result
  indistinguishable from a rollout that really drove the arm.
- ``start_task`` was the odd one out: it raised
  ``RuntimeError("cannot schedule new futures after shutdown")`` from the
  executor submit, naming a ``concurrent.futures`` internal rather than the
  robot, against this module's contract that a handler returns an error dict.

**A ``cleanup()`` landing during bring-up.** No entry-point guard can cover
this one: ``cleanup()`` sets ``_shutdown_event`` and only then calls
``stop_task()``, gated on ``status == RUNNING``. A task still in ``CONNECTING``
therefore gets no stop latch, finishes bring-up, and exits the loop on its
first evaluation::

    status when cleanup() ran : connecting
    _stop_requested set       : False        <- stop_task() never called
    verdict on main           : success
    task status / steps       : completed / 0

so the terminal block has to treat ``_shutdown_event`` the way it already
treats the stop latch. That fixes what the rollout *reports*. It does not stop
the bring-up: an entry-point guard cannot cover this producer, but the
in-flight stage gate can, and it read only the stop latch. With the status
already correct, the same side effects listed above still ran after the
teardown was latched::

    Policy.reset() on the caller's object : 1   <- 0 for the other producer
    observation reads off bus + cameras   : 1
    policy-server dial / checkpoint load  : 1

Both producers now stop before the next effect, which is why the two are
asserted side by side here.

No serial port is opened and no arm is commanded: an in-memory double stands in
for the driver, and the bring-up window is opened by the test rather than slept
through, so every assertion here is deterministic.
"""

from __future__ import annotations

import ast
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from strands_robots import hardware_robot
from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState, TaskStatus
from strands_robots.policies.base import Policy

#: Upper bound on any wait, so a broken contract fails instead of hanging.
DEADLINE = 10.0


class _Camera:
    """Camera double recording whether bring-up left it open."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._connected = False
        self.connect_calls = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, warmup: bool = True) -> None:
        self.connect_calls += 1
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False


class Bus:
    """Motors-bus double recording every device effect a rollout has.

    ``connect()`` blocks on ``connect_gate`` when the test arms it, which is how
    the ``cleanup()``-during-bring-up window is opened deterministically instead
    of being slept through.
    """

    def __init__(self) -> None:
        self.name = self.robot_type = "arm"
        self.is_calibrated = True
        self.cameras = {"wrist": _Camera("wrist")}
        self.config = type("Cfg", (), {"cameras": {}})()
        self._connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.commands: list[dict[str, Any]] = []
        self.connect_entered = threading.Event()
        self.connect_gate: threading.Event | None = None
        self.observation_reads = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = False) -> None:
        self.connect_calls += 1
        self.connect_entered.set()
        if self.connect_gate is not None and not self.connect_gate.wait(DEADLINE):  # pragma: no cover - deadline
            raise TimeoutError("bring-up gate never opened")
        for cam in self.cameras.values():
            cam.connect()
        self._connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        for cam in self.cameras.values():
            cam.disconnect()
        self._connected = False

    def get_observation(self) -> dict[str, Any]:
        self.observation_reads += 1
        return {"gripper.pos": 0.0}

    def send_action(self, action: dict[str, Any]) -> None:
        self.commands.append(action)


class CountingPolicy(Policy):
    """Records the per-episode reset and query calls bring-up makes."""

    def __init__(self) -> None:
        self.reset_calls = 0
        self.get_actions_calls = 0

    @property
    def provider_name(self) -> str:
        return "counting"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        pass

    def reset(self, seed: int | None = None) -> None:
        self.reset_calls += 1

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.get_actions_calls += 1
        return [{"gripper.pos": 0.11}] * 4


def make_robot(bus: Bus) -> HwRobot:
    """Build a Robot bypassing hardware init (the pattern used across tests/)."""
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "arm"
    hw.action_horizon = 4
    hw.data_config = None
    hw.control_frequency = 500.0
    hw.action_sleep_time = 1.0 / 500.0
    hw._task_state = RobotTaskState()
    hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arm_executor")
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_admission = threading.Lock()
    hw._task_claimed = False
    hw.mesh = None
    hw.peer_id = None
    hw.robot = bus
    return hw


@pytest.fixture
def bus() -> Bus:
    return Bus()


@pytest.fixture
def hw(bus: Bus) -> Any:
    robot = make_robot(bus)
    yield robot
    if bus.connect_gate is not None:
        bus.connect_gate.set()
    robot.cleanup()


def _text(result: dict[str, Any]) -> str:
    """The text of a tool-shaped result."""
    return " ".join(block["text"] for block in result["content"] if "text" in block)


def _shut_down(hw: HwRobot) -> None:
    """Put the robot in the post-``cleanup()`` state the guard refuses."""
    hw.cleanup()
    assert hw._shutdown_event.is_set()


class TestEveryEntryPointRefusesAfterShutdown:
    """All three admission paths refuse in the same tool shape."""

    def test_run_policy_refuses(self, hw: HwRobot, bus: Bus):
        """``run_policy`` errors naming the robot, not a completed rollout."""
        _shut_down(hw)

        result = hw.run_policy(policy_object=CountingPolicy(), instruction="after shutdown", n_steps=6)

        assert result["status"] == "error"
        assert "run_policy" in _text(result)
        assert "shut down" in _text(result)

    def test_execute_action_refuses(self, hw: HwRobot, bus: Bus):
        """The agent-tool ``execute`` / mesh dispatch chokepoint refuses too."""
        _shut_down(hw)

        result = hw._execute_task_sync("after shutdown", policy_object=CountingPolicy(), n_steps=6)

        assert result["status"] == "error"
        assert "execute_task" in _text(result)
        assert "shut down" in _text(result)

    def test_start_task_refuses_instead_of_raising(self, hw: HwRobot, bus: Bus):
        """``start_task`` reports the robot, not a ``concurrent.futures`` internal.

        Pre-fix this raised ``RuntimeError("cannot schedule new futures after
        shutdown")`` out of the executor submit.
        """
        _shut_down(hw)

        result = hw.start_task("after shutdown")

        assert result["status"] == "error"
        assert "start_task" in _text(result)
        assert "shut down" in _text(result)
        assert "futures" not in _text(result)

    def test_the_refusal_does_not_leave_the_claim_held(self, hw: HwRobot, bus: Bus):
        """A refused rollout must not lock the bus out of future ones."""
        _shut_down(hw)

        hw.run_policy(policy_object=CountingPolicy(), instruction="after shutdown", n_steps=6)

        assert hw._task_claimed is False


class TestRefusalHappensBeforeAnyDeviceEffect:
    """The refusal is what makes the leak and the state clobber unreachable."""

    @pytest.mark.parametrize("entry", ["run_policy", "execute_task", "start_task"])
    def test_no_entry_point_re_opens_the_hardware(self, hw: HwRobot, bus: Bus, entry: str):
        """``cleanup()`` does not disconnect, so a re-open would never be closed."""
        _shut_down(hw)

        policy = CountingPolicy()
        if entry == "run_policy":
            hw.run_policy(policy_object=policy, instruction="after shutdown", n_steps=6)
        elif entry == "execute_task":
            hw._execute_task_sync("after shutdown", policy_object=policy, n_steps=6)
        else:
            hw.start_task("after shutdown")

        assert bus.connect_calls == 0
        assert bus.is_connected is False
        assert bus.cameras["wrist"].connect_calls == 0
        assert bus.cameras["wrist"].is_connected is False

    def test_a_reusable_policy_keeps_its_episode_state(self, hw: HwRobot, bus: Bus):
        """``Policy.reset()`` must not fire for a rollout that cannot run.

        A caller may drive one policy object through several tasks, so clearing
        its action-chunk cache / sampler RNG on a refused call would corrupt a
        rollout running elsewhere.
        """
        _shut_down(hw)
        policy = CountingPolicy()

        hw.run_policy(policy_object=policy, instruction="after shutdown", n_steps=6)

        assert policy.reset_calls == 0
        assert policy.get_actions_calls == 0

    def test_the_arm_is_never_commanded(self, hw: HwRobot, bus: Bus):
        _shut_down(hw)

        hw.run_policy(policy_object=CountingPolicy(), instruction="after shutdown", n_steps=6)

        assert bus.commands == []


class TestNoFalseTerminalStatus:
    """A rollout that never ran is never reported as a completed one."""

    def test_the_task_state_is_not_marked_completed(self, hw: HwRobot, bus: Bus):
        """Pre-fix this was ``COMPLETED`` with 0 steps."""
        _shut_down(hw)

        hw.run_policy(policy_object=CountingPolicy(), instruction="after shutdown", n_steps=6)

        assert hw._task_state.status is not TaskStatus.COMPLETED
        assert hw._task_state.step_count == 0

    def test_the_json_payload_does_not_claim_a_rollout(self, hw: HwRobot, bus: Bus):
        """``run_policy``'s refusal carries no ``completed`` status payload."""
        _shut_down(hw)

        result = hw.run_policy(policy_object=CountingPolicy(), instruction="after shutdown", n_steps=6)

        payloads = [block["json"] for block in result["content"] if "json" in block]
        assert all(block.get("status") != "completed" for block in payloads)

    def test_a_shutdown_during_bring_up_reports_stopped_not_completed(self, hw: HwRobot, bus: Bus):
        """The path no entry-point guard can cover.

        ``cleanup()`` sets ``_shutdown_event`` then calls ``stop_task()`` only
        for ``status == RUNNING``, so a task still in ``CONNECTING`` gets no stop
        latch and used to fall out of the loop reporting ``completed`` / 0 steps.
        """
        bus.connect_gate = threading.Event()
        out: dict[str, Any] = {}
        thread = threading.Thread(
            target=lambda: out.__setitem__(
                "result",
                hw.run_policy(policy_object=CountingPolicy(), instruction="interrupted", n_steps=6),
            ),
            daemon=True,
        )
        thread.start()
        assert bus.connect_entered.wait(DEADLINE), "rollout never reached connect()"
        assert hw._task_state.status is TaskStatus.CONNECTING

        # The shutdown lands mid-bring-up, so no stop latch is set for it.
        hw.cleanup()
        assert hw._stop_requested.is_set() is False
        bus.connect_gate.set()
        thread.join(DEADLINE)
        assert not thread.is_alive(), "rollout never finished"

        assert out["result"]["status"] == "error"
        assert hw._task_state.status is TaskStatus.STOPPED
        assert hw._task_state.step_count == 0
        assert bus.commands == []


class TestAHealthyRolloutIsUnaffected:
    """The guard refuses exactly the shut-down case and nothing else."""

    def test_a_rollout_before_cleanup_still_completes(self, hw: HwRobot, bus: Bus):
        policy = CountingPolicy()

        result = hw.run_policy(policy_object=policy, instruction="healthy", n_steps=6)

        assert result["status"] == "success"
        assert hw._task_state.status is TaskStatus.COMPLETED
        assert hw._task_state.step_count == 6
        assert len(bus.commands) == 6
        assert policy.reset_calls == 1
        assert bus.connect_calls == 1

    def test_start_task_before_cleanup_still_submits(self, hw: HwRobot, bus: Bus):
        """A well-formed start still submits; the shutdown guard is the only refusal.

        The port is explicit because ``start_task`` now judges it before the
        submit: an absent ``policy_port`` is refused on its own terms (no policy
        can be built from it), which would mask the property under test here.
        """
        result = hw.start_task("healthy", policy_port=5555)

        assert result["status"] == "success"
        assert "Task started" in _text(result)


def _wait_until(predicate: Any, timeout: float = DEADLINE) -> bool:
    """Wait for ``predicate()`` without racing the thread that satisfies it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return False


def _interrupted_bring_up(
    hw: HwRobot,
    bus: Bus,
    *,
    policy: CountingPolicy | None = None,
    **run_kwargs: Any,
) -> dict[str, Any]:
    """Land ``cleanup()`` while the rollout is still in ``CONNECTING``.

    The bring-up is parked inside ``connect()`` -- the motors-bus handshake plus
    per-camera warmup, seconds on a real arm -- so the teardown lands in the one
    window ``cleanup()`` sets no stop latch for. The gate is opened only after
    ``cleanup()`` has returned, so whatever the rollout does next, it does with
    the shutdown already latched.
    """
    bus.connect_gate = threading.Event()
    out: dict[str, Any] = {}
    kwargs: dict[str, Any] = {"instruction": "interrupted", "n_steps": 6, **run_kwargs}
    if policy is not None:
        kwargs["policy_object"] = policy
    thread = threading.Thread(
        target=lambda: out.__setitem__("result", hw.run_policy(**kwargs)),
        daemon=True,
    )
    thread.start()
    assert bus.connect_entered.wait(DEADLINE), "rollout never reached connect()"
    assert hw._task_state.status is TaskStatus.CONNECTING

    hw.cleanup()
    # The premise the gate exists for: this teardown gated its ``stop_task()``
    # call on ``status == RUNNING``, so the shutdown event is its only record.
    assert hw._stop_requested.is_set() is False
    assert hw._shutdown_event.is_set() is True

    bus.connect_gate.set()
    thread.join(DEADLINE)
    assert not thread.is_alive(), "rollout never finished"
    return out["result"]


class TestAShutdownDuringBringUpStopsBeforeTheNextEffect:
    """The in-flight producer reaches no further than the one started later.

    ``TestRefusalHappensBeforeAnyDeviceEffect`` above pins these properties for
    a task started *after* ``cleanup()``. These pin them for a task already in
    bring-up when the teardown lands: the producer no entry-point guard can
    reach, and the one whose stage gate read only the stop latch.
    """

    def test_a_reusable_policy_keeps_its_episode_state(self, hw: HwRobot, bus: Bus):
        """``Policy.reset()`` must not fire for a rollout being abandoned.

        The mirror of the cell above: a caller may drive one policy object
        through several tasks, so clearing its action-chunk cache / sampler RNG
        while backing out of a rollout this teardown has cancelled would corrupt
        a rollout running elsewhere.
        """
        policy = CountingPolicy()

        _interrupted_bring_up(hw, bus, policy=policy)

        assert policy.reset_calls == 0
        assert policy.get_actions_calls == 0

    def test_nothing_is_read_off_the_devices(self, hw: HwRobot, bus: Bus):
        """No observation is taken from the bus or the cameras.

        ``_initialize_policy`` reads one to derive the policy's state keys. That
        read goes through ``read_observation``, which takes the bus lock, so on
        a real arm it is a serial exchange performed for a rollout the caller
        has already torn the robot down for.
        """
        _interrupted_bring_up(hw, bus, policy=CountingPolicy())

        assert bus.observation_reads == 0

    def test_no_policy_is_built_for_the_abandoned_rollout(self, hw: HwRobot, bus: Bus, monkeypatch: pytest.MonkeyPatch):
        """The most expensive effect: a policy-server dial or a checkpoint load.

        Driven through ``start_task`` and a provider, because that is the path
        where the build is remote work -- and the path where ``cleanup()`` pays
        for it directly, since the rollout is on the executor it joins.
        """
        dials: list[int | None] = []

        async def recording_get_policy(
            policy_port: int | None = None,
            policy_host: str = "localhost",
            policy_provider: str = "cosmos3",
            **kwargs: Any,
        ) -> Any:
            # The real one is called positionally, so the stand-in has to accept
            # the same three: a narrower one raises and the task reports ERROR,
            # which would leave ``dials`` empty for the wrong reason.
            dials.append(policy_port)
            return CountingPolicy()

        monkeypatch.setattr(hw, "_get_policy", recording_get_policy)
        bus.connect_gate = threading.Event()

        started = hw.start_task("interrupted", policy_port=5555, n_steps=6)
        assert started["status"] == "success"
        assert bus.connect_entered.wait(DEADLINE), "rollout never reached connect()"
        assert hw._task_state.status is TaskStatus.CONNECTING

        # ``cleanup()`` joins the executor, so it cannot return while the
        # bring-up is parked. Release the bring-up only once the shutdown latch
        # is on record -- the first thing ``cleanup()`` sets -- so the rollout
        # resumes into a teardown that has already happened.
        teardown = threading.Thread(target=hw.cleanup, daemon=True)
        teardown.start()
        assert _wait_until(hw._shutdown_event.is_set), "cleanup() never latched the shutdown"
        assert hw._stop_requested.is_set() is False
        bus.connect_gate.set()
        teardown.join(DEADLINE)
        assert not teardown.is_alive(), "cleanup() never returned"

        assert dials == []
        assert bus.observation_reads == 0
        # The rollout was abandoned, not derailed: an ERROR here would mean the
        # stand-in raised and the empty ``dials`` proved nothing.
        assert hw._task_state.status is TaskStatus.STOPPED


class TestBothProducersStopBeforeTheSameEffects:
    """One rule for the two ways a shutdown can meet a rollout."""

    @pytest.mark.parametrize("producer", ["started-after-cleanup", "cleanup-during-bring-up"])
    def test_neither_reaches_a_policy_reset_or_a_device_read(self, hw: HwRobot, bus: Bus, producer: str):
        policy = CountingPolicy()

        if producer == "started-after-cleanup":
            _shut_down(hw)
            hw.run_policy(policy_object=policy, instruction="after shutdown", n_steps=6)
        else:
            _interrupted_bring_up(hw, bus, policy=policy)

        assert policy.reset_calls == 0
        assert bus.observation_reads == 0


class TestWhatAnInterruptedBringUpStillReports:
    """Held before the gate read the shutdown latch, and still held after it.

    The terminal-status discriminator already covered the report, which is why
    reading the shutdown latch earlier had to leave every one of these alone:
    the gap was the work the rollout did, not what it said about it.
    """

    def test_it_reports_stopped_with_no_steps_and_no_commands(self, hw: HwRobot, bus: Bus):
        result = _interrupted_bring_up(hw, bus, policy=CountingPolicy())

        assert result["status"] == "error"
        assert hw._task_state.status is TaskStatus.STOPPED
        assert hw._task_state.step_count == 0
        assert bus.commands == []

    def test_the_devices_the_parked_bring_up_opens_are_still_left_open(self, hw: HwRobot, bus: Bus):
        """The boundary this gate does not move, stated so it is not assumed.

        ``connect()`` is already inside its blocking handshake when the teardown
        lands, so it finishes either way -- and ``cleanup()`` disconnects the
        robot before it returns, which for a rollout it did not submit is before
        that handshake completes. The devices are therefore open afterwards, on
        both sides of this change. Closing them needs ``cleanup()`` to wait for
        a rollout running on a caller's thread, which is a different fix.
        """
        _interrupted_bring_up(hw, bus, policy=CountingPolicy())

        assert bus.connect_calls == 1
        assert bus.cameras["wrist"].connect_calls == 1
        assert bus.is_connected is True
        assert bus.disconnect_calls == 0


def _rollout_module_ast() -> ast.Module:
    """Parse the hardware-robot module the three readers live in."""
    source = pathlib.Path(hardware_robot.__file__).read_text(encoding="utf-8")
    return ast.parse(source)


def _function_named(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is no longer defined in {hardware_robot.__name__}")


_PREDICATE = "_rollout_stop_latched"


class TestTheThreeReadersShareOnePredicate:
    """Structural, because equal copies are what the drift looked like before.

    Three places ask whether the rollout in flight must stop, and the answer has
    to be the same in all three. Asserting the values agree cannot see that: two
    copies of the same disjunction agree right up to the moment one of them is
    edited, which is how the bring-up gate came to read one latch while the loop
    condition and the terminal discriminator read both.
    """

    def test_the_disjunction_has_exactly_one_owner(self):
        """No reader re-derives ``stop or shutdown`` for itself."""
        wanted = {"self._stop_requested.is_set()", "self._shutdown_event.is_set()"}
        owners: list[str] = []
        tree = _rollout_module_ast()
        for parent in ast.walk(tree):
            if not isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for node in ast.walk(parent):
                if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
                    if wanted <= {ast.unparse(value) for value in node.values}:
                        owners.append(parent.name)

        assert owners == [_PREDICATE], f"the two latches are OR-ed together in {owners}"

    def test_the_bring_up_gate_reads_it(self):
        gate = _function_named(_rollout_module_ast(), "_honor_stop_request")

        assert _PREDICATE in ast.unparse(gate), "the bring-up gate does not read the shared predicate"

    def test_the_control_loop_exit_condition_reads_it(self):
        rollout = _function_named(_rollout_module_ast(), "_execute_task_async")
        loops = [
            node
            for node in ast.walk(rollout)
            if isinstance(node, ast.While) and "TaskStatus.RUNNING" in ast.unparse(node.test)
        ]

        assert len(loops) == 1, "the rollout's control loop is no longer recognisable"
        assert _PREDICATE in ast.unparse(loops[0].test)

    def test_the_terminal_status_discriminator_reads_it(self):
        rollout = _function_named(_rollout_module_ast(), "_execute_task_async")
        discriminators = [
            node
            for node in ast.walk(rollout)
            if isinstance(node, ast.If) and "TaskStatus.STOPPED" in ast.unparse(node.body)
        ]

        assert discriminators, "nothing in the rollout writes STOPPED as a terminal status"
        assert any(_PREDICATE in ast.unparse(node.test) for node in discriminators)
