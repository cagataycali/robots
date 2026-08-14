"""The documented ``RuntimeError`` exit of the WBC torque-shim auto-install hook.

:meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine._maybe_install_wbc_torque_control`
documents three exits: five no-op ``None`` returns, the cleanup callable on a
successful install, and a ``RuntimeError`` when the gate accepts a scene the
installer cannot wire. That third exit is reachable because the two halves ask
different questions - :func:`~strands_robots.policies.wbc.wbc_uses_position_servo`
is satisfied by *any* WBC joint on a position servo while
:func:`~strands_robots.policies.wbc.install_wbc_torque_control` needs *every*
one - so a model carrying a subset of the WBC joint set reaches it.

The sibling modules cover the other two: the no-op enumeration is pinned by the
guard tests beside this file, and the successful install by the rollout suites.
This module pins the third - that the hook propagates rather than degrading to a
no-op, that ``run_policy`` reports it as ``status="error"`` naming the joint that
could not be resolved, and that the refused install leaves the scene untouched
rather than half-flipped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.policies.wbc import (  # noqa: E402
    WBC_G1_LEG_WAIST_JOINTS,
    WBCConfig,
    WBCPolicy,
    WBCTorqueController,
    wbc_uses_position_servo,
)
from strands_robots.simulation.models import SimRobot, SimWorld  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# The subset size is what makes the rig partial: the gate resolves these three
# and answers True, while the installer needs all of WBC_G1_LEG_WAIST_JOINTS.
_PRESENT = 3
_MISSING_JOINT = WBC_G1_LEG_WAIST_JOINTS[_PRESENT]


def _partial_rig_mjcf() -> str:
    """MJCF carrying the first ``_PRESENT`` WBC joints as position servos."""
    present = WBC_G1_LEG_WAIST_JOINTS[:_PRESENT]
    bodies = "".join(
        f'<body name="l{i}"><joint name="{joint}" type="hinge" axis="0 0 1"/>'
        f'<geom type="box" size="0.05 0.05 0.05"/></body>'
        for i, joint in enumerate(present)
    )
    actuators = "".join(f'<position name="a{i}" joint="{joint}" kp="500"/>' for i, joint in enumerate(present))
    return f"<mujoco><worldbody>{bodies}</worldbody><actuator>{actuators}</actuator></mujoco>"


class _StubSession:
    """ONNX stand-in emitting a 15-wide action so no checkpoint is needed."""

    class _Input:
        name = "obs"

    def get_inputs(self) -> list[Any]:
        return [self._Input()]

    def run(self, _outputs: Any, _feed: Any) -> list[Any]:
        return [np.zeros((1, 15), dtype=np.float32)]


@pytest.fixture
def partial_rig() -> Any:
    """A sim whose scene carries only part of the WBC joint set."""
    model = mujoco.MjModel.from_xml_string(_partial_rig_mjcf())
    data = mujoco.MjData(model)

    sim = Simulation()
    # The real dataclasses rather than hand-rolled stubs: SimWorld and SimRobot
    # default every field this fixture leaves out, so the double carries the
    # shape the backend really builds for a scene.
    sim._world = SimWorld(
        robots={"g": SimRobot(name="g", urdf_path="")},
        _model=model,
        _data=data,
    )
    return sim


@pytest.fixture
def policy() -> WBCPolicy:
    """A balance WBCPolicy with a stub session (no weights on disk)."""
    p = WBCPolicy(config=WBCConfig(policy_path="unused.onnx"), walk=False, allow_missing_models=True)
    p.policy_session = _StubSession()
    return p


class TestThePartialRigReachesTheDocumentedRaise:
    """The gate and the installer ask different questions, so the raise is live."""

    def test_the_gate_accepts_a_rig_the_installer_refuses(self, partial_rig: Any, policy: WBCPolicy) -> None:
        """*Any* driven joint on a servo satisfies the gate; the installer needs *every* one."""
        assert wbc_uses_position_servo(partial_rig, policy, "g") is True

        with pytest.raises(RuntimeError) as excinfo:
            WBCTorqueController.from_sim(partial_rig, policy, "g")
        assert _MISSING_JOINT in str(excinfo.value)

    def test_the_hook_propagates_rather_than_returning_none(self, partial_rig: Any, policy: WBCPolicy) -> None:
        """The hook has a third exit: it raises instead of degrading to a no-op."""
        with pytest.raises(RuntimeError) as excinfo:
            partial_rig._maybe_install_wbc_torque_control(policy, "g")
        assert _MISSING_JOINT in str(excinfo.value)


class TestRunPolicyReportsTheRefusalAsAStructuredError:
    """``run_policy`` converts the raise into the envelope its docstring promises."""

    def test_run_policy_returns_an_error_envelope_naming_the_unresolved_joint(
        self, partial_rig: Any, policy: WBCPolicy
    ) -> None:
        """The caller gets a structured error, not a traceback, and it names the joint."""
        result = partial_rig.run_policy(robot_name="g", policy_object=policy, duration=0.05, control_frequency=20.0)
        assert result["status"] == "error"
        text = " ".join(block.get("text", "") for block in result.get("content", []) if "text" in block)
        assert _MISSING_JOINT in text, text
        # The prefix names the method that refused. The same RuntimeError text also
        # reaches a caller who drives install_wbc_torque_control directly, so without
        # it the envelope does not say which call declined.
        assert text.startswith("run_policy: "), text

    def test_the_refused_install_leaves_the_actuator_gains_untouched(self, partial_rig: Any, policy: WBCPolicy) -> None:
        """A refused install must not leave the scene half-flipped to torque."""
        model = partial_rig._world._model
        gainprm = model.actuator_gainprm[:, 0].copy()
        biastype = model.actuator_biastype.copy()

        partial_rig.run_policy(robot_name="g", policy_object=policy, duration=0.05, control_frequency=20.0)

        assert np.array_equal(model.actuator_gainprm[:, 0], gainprm)
        assert np.array_equal(model.actuator_biastype, biastype)

    def test_the_refused_install_registers_no_controller(self, partial_rig: Any, policy: WBCPolicy) -> None:
        """No action controller is left behind for a later rollout to dispatch through."""
        partial_rig.run_policy(robot_name="g", policy_object=policy, duration=0.05, control_frequency=20.0)
        assert partial_rig._world._backend_state.get("action_controller") is None


class TestTheHandlerStaysNarrowerThanException:
    """A programming error inside the installer must not be reported as a refusal.

    The handler catches ``RuntimeError`` because that is what the installer
    raises to describe a rig it cannot drive. Widening it to ``Exception``
    would convert a genuine defect inside the hook -- a missing attribute, a
    bad index -- into the same "could not install" envelope, so the reason a
    caller reads would name the rig rather than the bug. Nothing else pins
    that narrowness: every other case here raises ``RuntimeError``, so the two
    handler widths are indistinguishable to them.
    """

    def test_a_non_runtime_error_from_the_installer_propagates(
        self, partial_rig: Any, policy: WBCPolicy, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(self: Any, _policy: Any, _robot_name: str) -> None:
            raise AttributeError("simulated defect inside the installer")

        monkeypatch.setattr(type(partial_rig), "_maybe_install_wbc_torque_control", boom)
        with pytest.raises(AttributeError, match="simulated defect"):
            partial_rig.run_policy(robot_name="g", policy_object=policy, duration=0.05, control_frequency=20.0)
