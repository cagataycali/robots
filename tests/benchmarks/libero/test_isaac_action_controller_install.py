"""LiberoAdapter action-controller install on the Isaac backend (#1812).

The MuJoCo path builds robosuite's OSC_POSE controller against the compiled
MuJoCo model; on Isaac there is no compiled model, so pre-#1812 the install
degraded to a warning and every GR00T task-space action key landed in
``send_action``'s ``unresolved_keys`` -- success_rate pinned at 0.00 while
the run read green. These tests pin the new routing:

* On an engine exposing the Isaac action seam (``install_action_controller``
  / ``get_jacobian`` / ``list_robots`` / ``robot_joint_names`` /
  ``get_observation``), ``_install_action_controller`` installs an
  :class:`IsaacDeltaEEFController` instead of warning.
* Setup breakage on a genuine Isaac engine (missing Franka joints, broken
  Jacobian, ambiguous robots) stays LOUD: strict mode raises
  ``_ControllerInstallError``; non-strict records the failure.
* Engines with neither the MuJoCo nor the Isaac path keep the pre-existing
  warn-and-degrade behaviour, and the warning now names both missing paths.

No Isaac Sim install required -- the fake engine implements only the public
seam the adapter probes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from strands_robots.benchmarks.libero.adapter import (
    LiberoAdapter,
    _ControllerInstallError,
)
from strands_robots.simulation.isaac.delta_eef import IsaacDeltaEEFController

PICK_CUBE_BDDL = """
(define (problem libero_spatial_pick_cube)
  (:domain kitchen)
  (:language "pick up the red cube and place it on the plate")
  (:objects cube_1 plate_1 table_1 - object)
  (:init (on cube_1 table_1))
  (:goal (on cube_1 plate_1)))
"""

ARM = [f"panda_joint{i}" for i in range(1, 8)]
GRIP = ["panda_finger_joint1", "panda_finger_joint2"]
ALL_JOINTS = ARM + GRIP


class _FakeIsaacSim:
    """Duck-typed stand-in for IsaacSimulation's public action seam."""

    def __init__(
        self,
        joint_names: list[str] | None = None,
        robots: list[str] | None = None,
        jacobian_status: str = "success",
    ):
        self.joint_names = list(joint_names or ALL_JOINTS)
        self.robots = list(robots or ["robot"])
        self.jacobian_status = jacobian_status
        self.installed: dict[str, Any] = {}

    def list_robots(self) -> list[str]:
        return list(self.robots)

    def robot_joint_names(self, robot_name: str) -> list[str]:  # noqa: ARG002
        return list(self.joint_names)

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:  # noqa: ARG002
        return {name: 0.1 for name in self.joint_names}

    def get_jacobian(self, body_name=None, site_name=None, geom_name=None, robot_name=None):  # noqa: ARG002
        if self.jacobian_status != "success":
            return {"status": "error", "content": [{"text": "physics simulation view not created yet"}]}
        nv = len(self.joint_names)
        jacp = np.eye(3, nv).tolist()
        jacr = np.eye(3, nv, k=3).tolist()
        return {
            "status": "success",
            "content": [
                {"text": f"Jacobian for link '{body_name}'"},
                {"json": {"jacp": jacp, "jacr": jacr, "nv": nv}},
            ],
        }

    def install_action_controller(self, robot_name: str, controller: Any) -> dict[str, Any]:
        self.installed[robot_name] = controller
        return {"status": "success", "content": [{"text": f"Action controller installed for '{robot_name}'."}]}


def _adapter(**kwargs) -> LiberoAdapter:
    return LiberoAdapter.from_text(PICK_CUBE_BDDL, **kwargs)


class TestIsaacInstallPath:
    def test_installs_delta_eef_controller_on_isaac_seam(self):
        adapter = _adapter()
        sim = _FakeIsaacSim()
        adapter._install_action_controller(sim)
        assert adapter._action_controller_error is None
        assert adapter._isaac_action_controller_robot == "robot"
        controller = sim.installed["robot"]
        assert isinstance(controller, IsaacDeltaEEFController)
        assert controller.arm_joint_names == ARM
        assert controller.gripper_joint_names == GRIP

    def test_installed_controller_converts_a_task_space_action(self):
        """End-to-end wiring: the injected closures read the fake engine's
        Jacobian/observation and produce joint-name targets."""
        adapter = _adapter()
        sim = _FakeIsaacSim()
        adapter._install_action_controller(sim)
        targets = sim.installed["robot"].compute_joint_targets({"x": 1.0, "gripper": 1.0})
        assert set(targets) == set(ALL_JOINTS)
        # q = 0.1 everywhere; identity-block Jacobian moves dof 0 for an x delta.
        assert targets["panda_joint1"] > 0.1
        assert targets["panda_finger_joint1"] == pytest.approx(0.04)

    def test_reinstall_is_idempotent(self):
        adapter = _adapter()
        sim = _FakeIsaacSim()
        adapter._install_action_controller(sim)
        adapter._install_action_controller(sim)
        assert adapter._isaac_action_controller_robot == "robot"
        assert isinstance(sim.installed["robot"], IsaacDeltaEEFController)


class TestIsaacInstallFailuresStayLoud:
    def test_missing_franka_joints_raises_in_strict_mode(self):
        adapter = _adapter()  # strict_action_controller defaults True
        sim = _FakeIsaacSim(joint_names=["shoulder", "elbow"])
        with pytest.raises(_ControllerInstallError, match="panda_joint1"):
            adapter._install_action_controller(sim)
        assert adapter._action_controller_error is not None

    def test_missing_franka_joints_warns_in_non_strict_mode(self, caplog):
        adapter = _adapter(strict_action_controller=False)
        sim = _FakeIsaacSim(joint_names=["shoulder", "elbow"])
        with caplog.at_level("WARNING"):
            adapter._install_action_controller(sim)
        assert adapter._action_controller_error is not None
        assert "panda_joint1" in adapter._action_controller_error
        assert sim.installed == {}

    def test_multiple_robots_raise_in_strict_mode(self):
        adapter = _adapter()
        sim = _FakeIsaacSim(robots=["robot_a", "robot_b"])
        with pytest.raises(_ControllerInstallError, match="exactly one robot"):
            adapter._install_action_controller(sim)

    def test_broken_jacobian_probe_raises_in_strict_mode(self):
        adapter = _adapter()
        sim = _FakeIsaacSim(jacobian_status="error")
        with pytest.raises(_ControllerInstallError, match="Jacobian"):
            adapter._install_action_controller(sim)
        assert sim.installed == {}


class TestNonIsaacEnginesKeepDegradedPath:
    def test_engine_with_no_action_path_warns_and_names_both_paths(self, caplog):
        """A sim with neither a compiled MuJoCo model nor the Isaac seam must
        keep the pre-existing warn-and-degrade behaviour -- and the warning
        must name BOTH unavailable paths so the 0.00 success_rate is
        attributable (#1812 acceptance criterion)."""
        adapter = _adapter()

        class _BareSim:
            pass

        with caplog.at_level("WARNING"):
            adapter._install_action_controller(_BareSim())
        assert adapter._action_controller_error is not None
        assert "robosuite + mujoco" in adapter._action_controller_error
        assert "Isaac" in adapter._action_controller_error
        assert adapter._isaac_action_controller_robot is None

    def test_no_raise_even_in_strict_mode_when_no_engine_seam_exists(self):
        """Missing optional deps / wrong engine is environmental, not a
        fixable setup bug: strict mode must not raise (pre-#1812 contract)."""
        adapter = _adapter()  # strict
        adapter._install_action_controller(object())
        assert adapter._action_controller_error is not None


def _jac_payload(jacp: list, jacr: list, nv: int) -> dict[str, Any]:
    return {
        "status": "success",
        "content": [{"text": "Jacobian"}, {"json": {"jacp": jacp, "jacr": jacr, "nv": nv}}],
    }


class _ObsOmitsArmJoint(_FakeIsaacSim):
    """``robot_joint_names`` lists every Franka joint; the observation omits one.

    The install's joint check reads the articulation DOFs while the solver
    reads ``get_observation``, so this engine satisfies the first and not the
    second - the mismatch a Jacobian probe cannot see.
    """

    def get_observation(self, robot_name=None, *, skip_images=False):  # noqa: ARG002
        return {n: 0.1 for n in self.joint_names if n != ARM[3]}


class _ObsNamespacesItsKeys(_FakeIsaacSim):
    """Every joint is present, under a per-robot key prefix."""

    def get_observation(self, robot_name=None, *, skip_images=False):  # noqa: ARG002
        return {f"robot/{n}": 0.1 for n in self.joint_names}


class _ObsNonFinite(_FakeIsaacSim):
    def get_observation(self, robot_name=None, *, skip_images=False):  # noqa: ARG002
        obs = {n: 0.1 for n in self.joint_names}
        obs[ARM[3]] = float("nan")
        return obs


class _JacNonFinite(_FakeIsaacSim):
    def get_jacobian(self, **kwargs):
        payload = super().get_jacobian(**kwargs)
        payload["content"][1]["json"]["jacp"][0][0] = float("nan")
        return payload


class _JacTooFewColumns(_FakeIsaacSim):
    """Fewer columns than the articulation reports - numpy raises IndexError."""

    def get_jacobian(self, **kwargs):  # noqa: ARG002
        return _jac_payload(np.eye(3, 4).tolist(), np.eye(3, 4).tolist(), 4)


class _JacRaggedRows(_FakeIsaacSim):
    """Rows of unequal width - numpy raises ValueError."""

    def get_jacobian(self, **kwargs):  # noqa: ARG002
        nv = len(ALL_JOINTS)
        return _jac_payload([[1.0] * nv, [1.0] * (nv - 1), [1.0] * nv], np.eye(3, nv).tolist(), nv)


class _JacPayloadLacksJacp(_FakeIsaacSim):
    """A payload shaped like a Jacobian answer but missing the linear block."""

    def get_jacobian(self, **kwargs):  # noqa: ARG002
        return {"status": "success", "content": [{"text": "Jacobian"}, {"json": {"nv": len(ALL_JOINTS)}}]}


class _JacSevenRows(_FakeIsaacSim):
    """Six rows expected; seven supplied."""

    def get_jacobian(self, **kwargs):  # noqa: ARG002
        nv = len(ALL_JOINTS)
        return _jac_payload(np.eye(3, nv).tolist(), np.eye(4, nv).tolist(), nv)


class _InstallRefused(_FakeIsaacSim):
    def install_action_controller(self, robot_name: str, controller: Any) -> dict[str, Any]:  # noqa: ARG002
        return {"status": "error", "content": [{"text": "physics simulation view not created yet"}]}


# Every way a kinematics read the controller depends on can be broken, paired
# with the read it breaks. Each must reach the caller as a
# ``_ControllerInstallError`` so the strict/non-strict policy applies.
BROKEN_ENGINES = [
    pytest.param(_ObsOmitsArmJoint, "joint state", id="observation-omits-an-arm-joint"),
    pytest.param(_ObsNamespacesItsKeys, "joint state", id="observation-namespaces-its-keys"),
    pytest.param(_ObsNonFinite, "joint-state probe", id="observation-non-finite"),
    pytest.param(_JacNonFinite, "Jacobian probe", id="jacobian-non-finite"),
    pytest.param(_JacTooFewColumns, "Jacobian", id="jacobian-too-few-columns"),
    pytest.param(_JacRaggedRows, "Jacobian", id="jacobian-ragged-rows"),
    pytest.param(_JacPayloadLacksJacp, "Jacobian", id="jacobian-payload-lacks-jacp"),
    pytest.param(_JacSevenRows, "Jacobian probe", id="jacobian-seven-rows"),
    pytest.param(_InstallRefused, "refused the controller", id="install-refused"),
]


class TestEveryBrokenKinematicsReadIsLoudAtInstall:
    """The install probe covers what the per-action solver checks.

    ``_try_install_isaac_action_controller`` probes at install time so that,
    in its own words, "a broken kinematics read surfaces as a loud install
    error at episode start (where the strict/non-strict policy applies)
    instead of as one error envelope per action mid-eval".

    Before this class the probe read only the Jacobian and translated only a
    ``RuntimeError`` from the getter, so four broken reads installed a
    controller with ``_action_controller_error`` unset and then failed every
    action, and three malformed Jacobian payloads escaped as a bare
    ``IndexError`` / ``ValueError`` / ``KeyError`` - outside the
    ``_ControllerInstallError`` contract the docstring's ``Raises:`` states,
    so the strict/non-strict policy never applied to them.
    """

    @pytest.mark.parametrize(("engine_cls", "phrase"), BROKEN_ENGINES)
    def test_strict_mode_raises_a_controller_install_error(self, engine_cls, phrase):
        adapter = _adapter(strict_action_controller=True)
        with pytest.raises(_ControllerInstallError) as excinfo:
            adapter._install_action_controller(engine_cls())
        assert phrase in str(excinfo.value)

    @pytest.mark.parametrize(("engine_cls", "phrase"), BROKEN_ENGINES)
    def test_non_strict_mode_records_the_failure_instead_of_installing(self, engine_cls, phrase, caplog):
        adapter = _adapter(strict_action_controller=False)
        sim = engine_cls()
        with caplog.at_level("WARNING"):
            adapter._install_action_controller(sim)
        assert adapter._action_controller_error is not None
        assert phrase in adapter._action_controller_error
        assert adapter._isaac_action_controller_robot is None

    @pytest.mark.parametrize(("engine_cls", "phrase"), BROKEN_ENGINES)
    def test_no_controller_is_left_installed_on_the_engine(self, engine_cls, phrase):  # noqa: ARG002
        """A refused install must not leave a controller that would convert
        actions against the broken read."""
        adapter = _adapter(strict_action_controller=False)
        sim = engine_cls()
        adapter._install_action_controller(sim)
        assert sim.installed == {}


class TestAnAcceptedInstallCanConvertItsFirstAction:
    """The invariant that ties the probe to the solver.

    ``IsaacDeltaEEFController._solve_arm_targets`` re-checks the shape and
    finiteness of both reads on every action. If the install probe asserts the
    same properties, then an accepted install cannot fail the solver on its
    first action - so for every engine, either the install refuses or the
    conversion succeeds, and never "installed, then fails per action".
    """

    @pytest.mark.parametrize(
        ("engine_cls", "phrase"),
        [*BROKEN_ENGINES, pytest.param(_FakeIsaacSim, "", id="healthy-engine")],
    )
    def test_install_refuses_or_the_first_conversion_succeeds(self, engine_cls, phrase):  # noqa: ARG002
        adapter = _adapter(strict_action_controller=True)
        sim = engine_cls()
        try:
            adapter._install_action_controller(sim)
        except _ControllerInstallError:
            return  # refused at install - the other half of the contract
        controller = sim.installed["robot"]
        targets = controller.compute_joint_targets({"x": 0.01, "gripper": 1.0})
        assert set(targets) == set(ALL_JOINTS)


class TestTheProbeIsAFixedInstallTimeCost:
    def test_each_read_is_probed_once(self):
        """The probe must be a one-off install cost, not a per-action read."""
        calls = {"obs": 0, "jac": 0}

        class _Counting(_FakeIsaacSim):
            def get_observation(self, robot_name=None, *, skip_images=False):
                calls["obs"] += 1
                return super().get_observation(robot_name, skip_images=skip_images)

            def get_jacobian(self, **kwargs):
                calls["jac"] += 1
                return super().get_jacobian(**kwargs)

        adapter = _adapter(strict_action_controller=True)
        adapter._install_action_controller(_Counting())
        assert calls == {"obs": 1, "jac": 1}

    def test_the_healthy_engine_still_installs_and_converts(self):
        """Over-reach control: the probe refuses nothing a working engine does."""
        adapter = _adapter(strict_action_controller=True)
        sim = _FakeIsaacSim()
        adapter._install_action_controller(sim)
        assert adapter._action_controller_error is None
        assert adapter._isaac_action_controller_robot == "robot"
        targets = sim.installed["robot"].compute_joint_targets({"x": 1.0, "gripper": 1.0})
        assert targets["panda_joint1"] > 0.1


class TestTheSolverStillOwnsTheChecksTheProbeCannotMake:
    """Scope boundary, measured rather than asserted."""

    def test_the_joint_state_width_needs_no_install_check(self):
        """The injected closure builds exactly one element per arm joint, so a
        width mismatch is unreachable from this call site - an install-time
        width check would be dead code. The solver keeps its own check because
        it accepts any injected callable."""
        adapter = _adapter(strict_action_controller=True)
        sim = _FakeIsaacSim(joint_names=[*ALL_JOINTS, "extra_joint"])
        adapter._install_action_controller(sim)
        controller = sim.installed["robot"]
        assert len(controller.arm_joint_names) == len(ARM)
        q = controller._joint_positions_fn()
        assert np.asarray(q).shape == (len(ARM),)

    def test_the_solver_refuses_a_short_joint_state_from_any_other_caller(self):
        """The solver's own width check stays reachable for a hand-built
        controller, which is why the probe does not duplicate it."""
        controller = IsaacDeltaEEFController(
            arm_joint_names=list(ARM),
            gripper_joint_names=list(GRIP),
            joint_positions_fn=lambda: np.zeros(3),
            jacobian_fn=lambda: np.eye(6, len(ARM)),
        )
        with pytest.raises(RuntimeError, match="joint_positions_fn returned shape"):
            controller.compute_joint_targets({"x": 0.01})
