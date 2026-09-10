"""Every native ``run_policy`` runs the ``Policy`` its own seam declares.

:meth:`~strands_robots.drivers.base.HardwareDriver.run_policy` types its first
argument :class:`~strands_robots.policies.Policy`, and that class offers exactly
two ways to ask for an action -
:meth:`~strands_robots.policies.Policy.get_actions` and its synchronous wrapper
``get_actions_sync``. It declares no ``step`` and no ``__call__``, and no
subclass in this package adds either.

Two independent failures followed from that, one per admission dialect, and both
are pinned here:

* G1 and Go2 asked for ``step`` or a bare callable, so a built policy was turned
  away at the door with ``policy_object must be callable or expose a .step()
  method`` - the annotation named the one shape the refusal rejected.
* UR resolved ``get_actions_sync`` and so admitted the policy, then refused its
  answer one step later: ``get_actions`` returns a *chunk* (a list whose length
  is the action horizon) and the loop demanded a dict, so the rollout ended at
  step 0 having commanded nothing.

:func:`~strands_robots.drivers.base.policy_step` is the one owner of that
resolution, which is what keeps the set a driver admits and the set its loop can
call from drifting apart again. The untyped shapes keep working: a driver that
already ran a ``step`` object or a bare callable still runs it.
"""

from __future__ import annotations

import ast
import pathlib
import time
import types
from typing import Any

import pytest

import strands_robots.drivers as drivers_pkg
from strands_robots.drivers import g1 as g1_mod
from strands_robots.drivers.base import policy_step
from strands_robots.drivers.g1 import G1Driver
from strands_robots.drivers.go2 import Go2Driver
from strands_robots.drivers.ur import URDriver
from strands_robots.policies.base import Policy
from tests.drivers.test_g1_control_loop import install_unitree_sdk_stub as install_hg_sdk
from tests.drivers.test_go2_driver import install_unitree_sdk_stub as install_go_sdk
from tests.mocks.ur_rtde import MEASURED_Q, FakeRTDE

#: One joint per robot, so a cell can name the value that reached the wire.
G1_JOINT = "left_hip_pitch"
GO2_JOINT = "FL_calf_joint"

UR_HOST = "192.168.1.10"


class RecordingPolicy(Policy):
    """A real :class:`~strands_robots.policies.Policy` and nothing more.

    Implements only the abstract surface, so it is exactly what a driver
    receives from :func:`~strands_robots.policies.create_policy` as far as the
    call contract is concerned: ``get_actions`` plus the ABC's own
    ``get_actions_sync`` wrapper, no ``step``, not callable.

    Records the instruction it was handed on every step, which is how the cells
    below observe that ``run_policy``'s ``instruction`` reached the policy rather
    than being dropped at the driver boundary.
    """

    def __init__(self, chunk: list[dict[str, Any]]) -> None:
        self.chunk = chunk
        self.instructions: list[str] = []

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.instructions.append(instruction)
        return self.chunk

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        return None

    @property
    def provider_name(self) -> str:
        return "recording"


def _g1_rollout(
    monkeypatch: pytest.MonkeyPatch, policy: Any, instruction: str, n_steps: int
) -> tuple[dict[str, Any], list[Any]]:
    """Roll ``policy`` out on a G1 whose publisher is a recorder.

    Returns:
        The terminal task snapshot, and the commanded position of
        :data:`G1_JOINT` in every frame that carried a live gain - the
        zero-torque stop frame the loop always emits is excluded, so the list
        length is the number of steps that actuated.
    """
    install_hg_sdk(monkeypatch)
    slot = g1_mod._G1_JOINT_INDEX[G1_JOINT]
    commanded: list[float] = []

    def publish(_topic: Any, _cls: Any, cmd: Any) -> None:
        motor = cmd.motor_cmd[slot]
        if motor.kp != 0.0:
            commanded.append(motor.q)
        return None

    driver = G1Driver(port="127.0.0.1", network_interface="lo")
    driver._pubs = types.SimpleNamespace(publish=publish)  # type: ignore[assignment]
    driver._connected = True
    driver._mode_machine = 9
    driver._fsm_id = 500
    driver._battery = {"pct": 80.0}
    driver._imu = {"rpy": [0.0, 0.0, 0.0]}
    driver._fsm_read_at = time.monotonic()
    monkeypatch.setattr(
        driver,
        "_refresh_fsm_id",
        lambda: setattr(driver, "_fsm_read_at", time.monotonic()),
    )
    envelope = driver.run_policy(policy, instruction=instruction, n_steps=n_steps)
    if envelope["status"] != "success":
        return envelope, commanded
    _drain(driver)
    return driver.get_task_status()["content"][0]["json"], commanded


def _go2_rollout(
    monkeypatch: pytest.MonkeyPatch, policy: Any, instruction: str, n_steps: int
) -> tuple[dict[str, Any], list[Any]]:
    """Roll ``policy`` out on a gate-passing Go2 whose publisher is a recorder."""
    install_go_sdk(monkeypatch)
    slot = list(g1_mod_go2_index()).index(GO2_JOINT)
    commanded: list[float] = []

    def publish(_topic: Any, _cls: Any, cmd: Any) -> None:
        motor = cmd.motor_cmd[slot]
        if motor.kp != 0.0:
            commanded.append(motor.q)
        return None

    driver = Go2Driver(tool_name="go2", port="192.168.123.161")
    driver._connected = True
    driver._sport_mode_released = True
    driver._battery = {"pct": 88.0, "current": 1.0, "cycle": 3}
    driver._pubs = types.SimpleNamespace(publish=publish)  # type: ignore[assignment]
    envelope = driver.run_policy(policy, instruction=instruction, n_steps=n_steps)
    if envelope["status"] != "success":
        return envelope, commanded
    _drain(driver)
    return driver.get_task_status()["content"][0]["json"], commanded


def _ur_rollout(
    monkeypatch: pytest.MonkeyPatch, policy: Any, instruction: str, n_steps: int
) -> tuple[dict[str, Any], list[Any]]:
    """Roll ``policy`` out on a UR wired to the fake controller."""
    fake = FakeRTDE()
    control = types.ModuleType("rtde_control")
    receive = types.ModuleType("rtde_receive")
    control.RTDEControlInterface = fake.make_control  # type: ignore[attr-defined]
    receive.RTDEReceiveInterface = fake.make_receive  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "rtde_control", control)
    monkeypatch.setitem(__import__("sys").modules, "rtde_receive", receive)

    driver = URDriver(tool_name="ur5e", port=UR_HOST, control_frequency=200.0)
    assert driver.connect_eagerly() is None
    envelope = driver.run_policy(policy, instruction=instruction, n_steps=n_steps)
    if envelope["status"] != "success":
        return envelope, []
    _drain(driver)
    status = driver.get_task_status()["content"][0]["json"]
    servoed = [call[0] for control in fake.controls for call in control.servoj_calls]
    driver.cleanup()
    return status, servoed


def g1_mod_go2_index() -> dict[str, int]:
    """The Go2 joint order, read off the driver rather than restated here."""
    from strands_robots.drivers.go2 import GO2_JOINT_INDEX

    return GO2_JOINT_INDEX


def _drain(driver: Any, timeout: float = 5.0) -> None:
    """Wait for the driver's rollout thread to finish, or fail the cell."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = driver.get_task_status()["content"][0]["json"]
        if not status.get("running", False):
            return
        time.sleep(0.01)
    raise AssertionError("the rollout never finished")


def _ur_action(offset: float = 0.001) -> dict[str, float]:
    """A UR setpoint one step off the measured pose, so the step gate admits it."""
    from strands_robots.drivers.ur import JOINT_NAMES

    return {name: q + offset for name, q in zip(JOINT_NAMES, MEASURED_Q, strict=True)}


def _ur_recorded(action: dict[str, float]) -> list[float]:
    """What ``servoJ`` records for ``action``: the ordered joint vector."""
    from strands_robots.drivers.ur import JOINT_NAMES

    return [action[name] for name in JOINT_NAMES]


#: Every native driver that runs a policy loop, paired with an action its wire
#: admits and the value that wire records for it. One table, so a fourth driver
#: joining the family is one row - and every row asserts, rather than a row
#: whose expectation is unexpressible quietly grading nothing.
ROLLOUTS = [
    pytest.param(_g1_rollout, lambda offset=0.05: {G1_JOINT: offset}, lambda a: a[G1_JOINT], id="g1"),
    pytest.param(_go2_rollout, lambda offset=0.05: {GO2_JOINT: offset}, lambda a: a[GO2_JOINT], id="go2"),
    pytest.param(_ur_rollout, _ur_action, _ur_recorded, id="ur"),
]


class TestABuiltPolicyRunsOnEveryNativeDriver:
    """The typed contract: a ``Policy`` is admitted and commands the wire."""

    @pytest.mark.parametrize(("rollout", "action", "recorded"), ROLLOUTS)
    def test_a_policy_commands_every_step_of_its_budget(
        self, rollout: Any, action: Any, recorded: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-fix: G1/Go2 refused at the door, UR ended at step 0 commanding nothing."""
        policy = RecordingPolicy([action()])

        status, commanded = rollout(monkeypatch, policy, "pick up the cube", 3)

        assert status.get("exit_reason") == "n_steps", status
        assert status["steps"] == 3, status
        assert len(commanded) == 3, f"{len(commanded)} frames reached the wire, expected 3"

    @pytest.mark.parametrize(("rollout", "action", "recorded"), ROLLOUTS)
    def test_the_instruction_reaches_the_policy(
        self, rollout: Any, action: Any, recorded: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``instruction`` is the second argument of ``get_actions_sync``.

        G1 and Go2 dropped it with ``del instruction`` and a docstring asserting
        policies own their own conditioning - which is not true of a policy the
        seam hands an instruction to.
        """
        policy = RecordingPolicy([action()])

        rollout(monkeypatch, policy, "pick up the cube", 3)

        assert policy.instructions == ["pick up the cube"] * 3, policy.instructions

    @pytest.mark.parametrize(("rollout", "action", "recorded"), ROLLOUTS)
    def test_the_first_action_of_a_chunk_is_the_one_commanded(
        self, rollout: Any, action: Any, recorded: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A chunk is consumed one action per step, first action first.

        ``get_actions`` returns a list whose length is the action horizon. A
        native loop commands one frame per step, so it takes the head - the same
        convention :mod:`strands_robots.hardware_robot` applies to a chunk.
        """
        first, second = action(), action(0.5)
        assert first != second, "the two chunk entries must differ for the cell to grade"
        policy = RecordingPolicy([first, second])

        _status, commanded = rollout(monkeypatch, policy, "", 2)

        assert len(commanded) == 2, commanded
        for frame in commanded:
            assert frame == pytest.approx(recorded(first)), commanded


class TestTheUntypedShapesStillRun:
    """A ``step`` object and a bare callable are still admitted."""

    @pytest.mark.parametrize(("rollout", "action", "recorded"), ROLLOUTS)
    @pytest.mark.parametrize("shape", ["step", "callable"])
    def test_an_untyped_policy_is_not_regressed(
        self, rollout: Any, action: Any, recorded: Any, shape: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        commanded_action = action()
        if shape == "step":
            policy: Any = types.SimpleNamespace(step=lambda _obs: commanded_action)
        else:
            policy = lambda _obs: commanded_action  # noqa: E731 - the bare callable is the subject

        status, commanded = rollout(monkeypatch, policy, "", 2)

        assert status.get("exit_reason") == "n_steps", status
        assert len(commanded) == 2, commanded


class TestTheRefusalNamesTheContract:
    """An object of none of the three shapes is refused, by its real name."""

    @pytest.mark.parametrize(("rollout", "action", "recorded"), ROLLOUTS)
    def test_an_unresolvable_object_is_refused(
        self, rollout: Any, action: Any, recorded: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        envelope, commanded = rollout(monkeypatch, object(), "", 2)

        assert envelope["status"] == "error", envelope
        text = envelope["content"][0]["text"]
        assert "get_actions_sync" in text, text
        assert commanded == [], "a refused policy commanded the wire"

    def test_the_old_refusal_no_longer_names_step_as_the_contract(self) -> None:
        """``.step()`` was never the package's contract; no refusal claims it is."""
        offenders = [
            str(path)
            for path in pathlib.Path(drivers_pkg.__file__).parent.rglob("*.py")
            if "expose a .step() method" in path.read_text(encoding="utf-8")
        ]
        assert offenders == [], offenders


class TestOneOwnerResolvesThePolicyShapes:
    """The resolution lives in one place, so the dialects cannot diverge again."""

    def test_no_driver_keeps_a_private_resolver(self) -> None:
        """A driver-local ``_policy_step`` is the drift this consolidated.

        Derived over the package rather than named per driver, so a ninth driver
        reintroducing its own copy fails here instead of shipping a dialect.
        """
        root = pathlib.Path(drivers_pkg.__file__).parent
        scanned: list[str] = []
        offenders: list[str] = []
        for path in sorted(root.rglob("*.py")):
            scanned.append(path.name)
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in {"_policy_step", "policy_step"}:
                    if path.name != "base.py":
                        offenders.append(f"{path.name}::{node.name}")
        assert len(scanned) > 10, f"the scan read {len(scanned)} files; it must read the package"
        assert offenders == [], offenders

    @pytest.mark.parametrize(
        ("policy", "resolves"),
        [
            pytest.param(RecordingPolicy([{"a": 1.0}]), True, id="built-policy"),
            pytest.param(types.SimpleNamespace(step=lambda _o: {"a": 1.0}), True, id="step-object"),
            pytest.param(lambda _o: {"a": 1.0}, True, id="bare-callable"),
            pytest.param(object(), False, id="neither"),
            pytest.param(None, False, id="none"),
        ],
    )
    def test_the_owner_resolves_exactly_the_three_shapes(self, policy: Any, resolves: bool) -> None:
        assert (policy_step(policy, "go") is not None) is resolves

    @pytest.mark.parametrize(
        ("returned", "commanded"),
        [
            pytest.param({"a": 1.0}, {"a": 1.0}, id="a-single-action-passes-through"),
            pytest.param([{"a": 1.0}, {"a": 2.0}], {"a": 1.0}, id="a-chunk-yields-its-head"),
            pytest.param([], None, id="an-empty-chunk-commanded-nothing"),
            pytest.param(None, None, id="none-commanded-nothing"),
        ],
    )
    def test_the_owner_reduces_a_chunk_to_the_action_a_step_commands(self, returned: Any, commanded: Any) -> None:
        step = policy_step(lambda _obs: returned, "go")
        assert step is not None
        assert step({}) == commanded
