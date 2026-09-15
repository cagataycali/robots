"""``attach_bodies(mode="kinematic")`` carries its child on EVERY stepping path.

``attach_bodies`` documents the kinematic mode as "every physics step, the
child's freejoint pose is overwritten to follow the parent", and the follow is
applied by one hook the backend calls after ``mj_step``. Four places in the
MuJoCo backend advance physics: the ``step()`` batch loop, the single-robot
``_apply_sim_action`` substep loop, the motion-primitive tick, and the
synchronized ``run_multi_policy`` loop. Three re-applied the follow; the
multi-robot loop did not, so a body attached for a two-arm handover was left
behind - it fell to the floor while ``attach_bodies`` reported success,
``run_multi_policy`` reported success, and the attachment registry still named
it as carried. ``run_multi_policy`` is the recommended path for recording
concurrent multi-robot episodes, so those frames went into the dataset showing
the arms moving and the "grasped" object on the ground.

A fifth path reaches physics without being one of those loops: an
``action_controller`` that declares ``owns_stepping`` runs its own ``mj_step``
burst and ``_apply_sim_action`` skips the loop holding the re-pin. That is a
shipped configuration - :class:`WBCTorqueController` declares the flag, and its
own module describes carrying the upper body of a ``CompositePolicy`` (legs from
WBC, arms from a manipulation policy), i.e. walking while holding something.
Nothing on ``attach_bodies`` makes the follow conditional on who calls
``mj_step``, so the same silent outcome applied there: the object stayed where
it was for the whole episode while the attach, the controller and the policy
loop all reported success.

Pinned here:

* the contract, on the multi-robot loop: a carried child stays at the relative
  pose the attachment recorded, and does not end the episode on the floor;
* the root cause, as a source-level guard: every function in the MuJoCo backend
  that calls ``mj_step`` also re-applies the follow, so a fifth stepping path
  cannot silently drop it;
* the same contract on a stepping-owning ``action_controller``, where the
  engine cannot re-pin per substep and re-pins per control step instead;
* the boundary: ``step()`` and single-robot ``run_policy`` still carry, a
  controller that does NOT own stepping still carries, a weld attachment
  (solver-enforced, not hook-driven) is untouched, and a loop with no
  attachments leaves unattached bodies falling normally.
"""

from __future__ import annotations

import ast
import os
import pathlib
import tempfile

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco as mj  # noqa: E402

from strands_robots.policies.mock import MockPolicy  # noqa: E402
from strands_robots.simulation import Simulation  # noqa: E402
from strands_robots.simulation import mujoco as mujoco_backend  # noqa: E402
from tests.tool_result_contract import tool_json  # noqa: E402

from .test_run_multi_policy_no_recording import _ROBOT_XML  # noqa: E402

# The arm body the cube is carried by. It swings under the mock policy's
# sinusoid, so a child that stops following diverges from it.
_PARENT = "alpha/link2"
_CHILD = "cube"

# One integration step of latency is inherent to the follow: it places the child
# from the parent pose the step that just finished produced, so the residual
# scales with the parent's speed times the timestep. Measured on this fixture
# the carrying paths land within 0.026 m while an unheld child reaches 0.65 m,
# so this bound sits about 2x above the former and 13x below the latter.
_CARRY_TOL_M = 0.05

# The cube spawns airborne; anything at or below this has landed on the ground
# plane rather than being carried.
_FLOOR_Z = 0.08


@pytest.fixture
def sim_arm_and_cube():
    """One namespaced arm plus an airborne dynamic cube, no recorder."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_ROBOT_XML)

    s = Simulation(tool_name="test_kinematic_follow", mesh=False)
    s.create_world()
    _ok(s.add_robot("alpha", urdf_path=path, position=[0, 0, 0]), "add_robot")
    _ok(s.add_object(_CHILD, shape="box", size=[0.03] * 3, position=[0.12, 0, 0.45]), "add_object")
    s.step(5)
    yield s
    s.destroy()


def _relpos_in_parent_frame(sim, parent: str, child: str) -> np.ndarray:
    """``child``'s position expressed in ``parent``'s frame - what the record stores."""
    model, data = sim._world._model, sim._world._data
    parent_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, parent)
    child_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, child)
    assert parent_id >= 0 and child_id >= 0, f"{parent!r}/{child!r} not in the compiled model"
    neg = np.zeros(4)
    mj.mju_negQuat(neg, data.xquat[parent_id])
    rel = np.zeros(3)
    mj.mju_rotVecQuat(rel, data.xpos[child_id] - data.xpos[parent_id], neg)
    return rel.copy()


def _recorded_relpos(sim, child: str) -> np.ndarray:
    record = sim._world._backend_state["kinematic_attachments"][child]
    return np.asarray(record["relpos"], dtype=float)


def _carry_error(sim, parent: str, child: str) -> float:
    """How far the child sits from the offset the attachment captured."""
    return float(np.linalg.norm(_relpos_in_parent_frame(sim, parent, child) - _recorded_relpos(sim, child)))


def _body_z(sim, name: str) -> float:
    model, data = sim._world._model, sim._world._data
    return float(data.xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)][2])


def _ok(result: dict, what: str) -> dict:
    """Return a successful tool result, or fail naming what refused and why.

    A ``raise`` rather than an ``assert`` on the call, so the scene is still
    built - and still checked - when the suite runs under ``python -O``.
    """
    if result["status"] != "success":
        raise AssertionError(f"{what} refused: {result['content'][0]['text']}")
    return result


def _attach(sim, parent: str = _PARENT, child: str = _CHILD, mode: str = "kinematic"):
    return _ok(sim.attach_bodies(parent, child, mode=mode), f"attach_bodies(mode={mode!r})")


def _drive_multi_policy(sim, n_steps: int = 200):
    return _ok(
        sim.run_multi_policy(
            policies={"alpha": MockPolicy()},
            instructions="carry the cube",
            n_steps=n_steps,
            control_frequency=50.0,
        ),
        "run_multi_policy",
    )


class TestTheMultiRobotLoopCarriesAnAttachedBody:
    def test_a_kinematically_carried_body_follows_through_run_multi_policy(self, sim_arm_and_cube):
        """The child holds the recorded offset across the synchronized loop.

        Before the fix the loop stepped physics without re-applying the follow,
        so the cube fell away from the arm while every surface reported success.
        """
        sim = sim_arm_and_cube
        _attach(sim)
        assert _carry_error(sim, _PARENT, _CHILD) < _CARRY_TOL_M, "premise: attach captures the current offset"
        z_at_attach = _body_z(sim, _CHILD)

        _drive_multi_policy(sim)

        error = _carry_error(sim, _PARENT, _CHILD)
        assert error < _CARRY_TOL_M, (
            f"run_multi_policy left the carried body {error:.4f} m from the offset the attachment "
            f"recorded (z {z_at_attach:.4f} -> {_body_z(sim, _CHILD):.4f}); "
            'attach_bodies(mode="kinematic") promises it follows every physics step'
        )

    def test_the_carried_body_is_not_left_on_the_floor(self, sim_arm_and_cube):
        """The user-visible harm, read through the public body-state surface.

        A dataset recorded through this loop is meant to show the object riding
        with the arm. Without the follow the object simply falls, so the
        episode's frames show the arms working over a cube lying on the ground.
        """
        sim = sim_arm_and_cube
        _attach(sim)
        assert _body_z(sim, _CHILD) > _FLOOR_Z, "premise: the cube starts airborne, held by the arm"

        _drive_multi_policy(sim)

        state = tool_json(sim.get_body_state(_CHILD))
        carried_z = float(state["position"][2])
        assert carried_z > _FLOOR_Z, (
            f"the carried body ended at z={carried_z:.4f} m, on the ground rather than with its "
            f"parent {_PARENT!r}, while run_multi_policy and attach_bodies both reported success"
        )


class TestEveryBackendSteppingPathReappliesTheFollow:
    """The root cause: one hook, and every stepping path has to call it.

    A source-level sweep rather than a per-path rollout, so a fifth stepping
    path added later fails here instead of silently dropping carried bodies.
    Scoped to the MuJoCo backend, which owns the attachment registry. An
    ``action_controller`` that takes over stepping (LIBERO, WBC torque control)
    replaces this loop wholesale rather than adding a call site here, so it is
    pinned behaviourally by
    :class:`TestASteppingOwningControllerCarriesToo` instead.
    """

    _BACKEND_DIR = pathlib.Path(str(next(iter(mujoco_backend.__path__))))
    _HOOK = "_apply_kinematic_attachments"
    # step(), _apply_sim_action, _primitive_tick, run_multi_policy.
    _MIN_STEPPING_FUNCTIONS = 4

    @staticmethod
    def _own_calls(node: ast.AST) -> list[str]:
        """Names of calls made by ``node`` itself, not by functions nested in it."""
        names: list[str] = []
        stack: list[ast.AST] = list(ast.iter_child_nodes(node))
        while stack:
            current = stack.pop()
            if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                continue
            if isinstance(current, ast.Call):
                func = current.func
                if isinstance(func, ast.Attribute):
                    names.append(func.attr)
                elif isinstance(func, ast.Name):
                    names.append(func.id)
            stack.extend(ast.iter_child_nodes(current))
        return names

    def _stepping_functions(self) -> dict[str, list[str]]:
        """``"module.function" -> its own call names`` for every mj_step caller."""
        found: dict[str, list[str]] = {}
        for module in sorted(self._BACKEND_DIR.glob("*.py")):
            tree = ast.parse(module.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                calls = self._own_calls(node)
                if "mj_step" in calls:
                    found[f"{module.name}::{node.name}"] = calls
        return found

    def test_the_sweep_found_every_stepping_path(self):
        """Non-vacuity: a refactor that hides the stepping paths fails here."""
        stepping = self._stepping_functions()
        assert len(stepping) >= self._MIN_STEPPING_FUNCTIONS, (
            f"expected at least {self._MIN_STEPPING_FUNCTIONS} mj_step call sites in the MuJoCo "
            f"backend, found {sorted(stepping)} - a clean result below that proves nothing"
        )

    def test_every_stepping_path_reapplies_the_kinematic_follow(self):
        """Each function that advances physics also re-applies the follow."""
        missing = sorted(where for where, calls in self._stepping_functions().items() if self._HOOK not in calls)
        assert not missing, (
            f"these MuJoCo backend functions call mj_step without re-applying the kinematic "
            f"follow, so a body attached with mode='kinematic' is left behind on their path: {missing}"
        )


class TestTheOtherStepPathsAndModesAreUnchanged:
    """Boundary controls: these hold on both sides of the fix."""

    def test_the_step_batch_loop_still_carries(self, sim_arm_and_cube):
        sim = sim_arm_and_cube
        _attach(sim)
        for _ in range(5):
            _ok(sim.send_action({"shoulder_pan": 0.7, "elbow": -0.6}, robot_name="alpha"), "send_action")
            _ok(sim.step(40), "step")
        assert _carry_error(sim, _PARENT, _CHILD) < _CARRY_TOL_M

    def test_the_single_robot_policy_loop_still_carries(self, sim_arm_and_cube):
        sim = sim_arm_and_cube
        _attach(sim)
        _ok(
            sim.run_policy(robot_name="alpha", policy_provider="mock", n_steps=200, control_frequency=50.0),
            "run_policy",
        )
        assert _carry_error(sim, _PARENT, _CHILD) < _CARRY_TOL_M

    def test_a_welded_child_is_held_by_the_solver_not_the_hook(self, sim_arm_and_cube):
        """Weld mode is an equality constraint, so it never used the step hook.

        Asserted structurally (the child is absent from the follow registry)
        plus the outcome that matters - it is still held aloft, not on the
        floor - because a weld is solver-enforced and yields elastically under
        a fast swing rather than tracking to millimetres.
        """
        sim = sim_arm_and_cube
        _attach(sim, mode="weld")
        assert _CHILD not in sim._world._backend_state.get("kinematic_attachments", {}), (
            "a weld must not register a per-step follow"
        )
        _drive_multi_policy(sim)
        assert _body_z(sim, _CHILD) > _FLOOR_Z, "the weld constraint must keep holding the child"

    def test_an_unattached_body_still_falls_during_the_loop(self, sim_arm_and_cube):
        """No-overreach: the loop does not freeze bodies nothing is carrying."""
        sim = sim_arm_and_cube
        assert not sim._world._backend_state.get("kinematic_attachments")
        z_before = _body_z(sim, _CHILD)
        _drive_multi_policy(sim, n_steps=100)
        assert _body_z(sim, _CHILD) < z_before - 0.1, "an unattached airborne cube must fall"


class _SubstepBurstController:
    """The shipped stepping-owning controller's contract, minimally.

    Mirrors :class:`~strands_robots.policies.wbc.sim_control.WBCTorqueController`:
    ``apply`` writes ``data.ctrl`` for the action - by delegating to the
    engine's own name lookup, so the arm moves exactly as it would with no
    controller installed - and, when it declares ``owns_stepping``, advances
    physics itself for ``physics_substeps_per_control`` steps. The two
    parametrisations differ in one thing only: who calls ``mj_step``.
    """

    def __init__(self, sim, *, owns_stepping: bool, substeps: int = 4) -> None:
        self.sim = sim
        self.owns_stepping = owns_stepping
        self.physics_substeps_per_control = substeps
        self.mj_steps = 0

    def apply(self, action_dict, model, data, robot_name):  # noqa: ANN001, ANN201
        robot = self.sim._world.robots[robot_name]
        self.sim._apply_action_by_name(model, data, action_dict, robot.namespace or "", mj, robot_name)
        if not self.owns_stepping:
            return
        for _ in range(self.physics_substeps_per_control):
            mj.mj_step(model, data)
            self.mj_steps += 1


def _drive_through_controller(sim, controller, n_calls: int = 120):
    """Install ``controller`` and swing the arm through ``send_action`` only.

    Deliberately never calls ``step()`` or ``run_multi_policy``: those paths own
    their own re-pin, and routing through them would pass whatever
    ``_apply_sim_action`` does. Drives the same steady target as the ``step()``
    boundary control rather than a whipping alternation, because the follow's
    inherent one-integration-step latency scales with the parent's speed - a
    residual from slewing the arm as fast as the servos allow would be measuring
    that latency, not the re-pin.
    """
    sim._world._backend_state["action_controller"] = controller
    try:
        for _ in range(n_calls):
            _ok(
                sim.send_action({"shoulder_pan": 0.7, "elbow": -0.6}, robot_name="alpha"),
                "send_action",
            )
    finally:
        sim._world._backend_state.pop("action_controller", None)


class TestASteppingOwningControllerCarriesToo:
    """``owns_stepping`` skips the loop holding the re-pin, not the promise."""

    def test_a_carried_body_follows_when_the_controller_owns_stepping(self, sim_arm_and_cube):
        """The child holds the recorded offset across the controller's bursts.

        Before the fix ``_apply_sim_action`` skipped its whole substep loop -
        including the re-pin - whenever the controller declared it stepped for
        itself, so the child simply free-fell for the episode.
        """
        sim = sim_arm_and_cube
        _attach(sim)
        assert _carry_error(sim, _PARENT, _CHILD) < _CARRY_TOL_M, "premise: attach captures the current offset"
        controller = _SubstepBurstController(sim, owns_stepping=True)

        _drive_through_controller(sim, controller)

        assert controller.mj_steps > 0, "premise: the controller advanced physics itself"
        error = _carry_error(sim, _PARENT, _CHILD)
        assert error < _CARRY_TOL_M, (
            f"a controller declaring owns_stepping left the carried body {error:.4f} m from the "
            f"offset the attachment recorded; attach_bodies(mode='kinematic') does not make the "
            f"follow conditional on who calls mj_step"
        )

    def test_the_carried_body_is_not_left_behind_on_the_floor(self, sim_arm_and_cube):
        """The user-visible harm, read through the public body-state surface."""
        sim = sim_arm_and_cube
        _attach(sim)
        assert _body_z(sim, _CHILD) > _FLOOR_Z, "premise: the cube starts airborne, held by the arm"

        _drive_through_controller(sim, _SubstepBurstController(sim, owns_stepping=True))

        carried_z = float(tool_json(sim.get_body_state(_CHILD))["position"][2])
        assert carried_z > _FLOOR_Z, (
            f"the carried body ended at z={carried_z:.4f} m, on the ground rather than with its "
            f"parent {_PARENT!r}, while attach_bodies, the controller and send_action all "
            f"reported success"
        )

    def test_a_controller_that_does_not_own_stepping_still_carries(self, sim_arm_and_cube):
        """Boundary: the flag is what selects the branch, not the controller.

        Same stub, same action, same delegation - only ``owns_stepping`` differs
        - so a failure here would mean installing any controller broke the
        carry, which is a different bug from the one above.
        """
        sim = sim_arm_and_cube
        _attach(sim)

        controller = _SubstepBurstController(sim, owns_stepping=False)
        _drive_through_controller(sim, controller)

        assert controller.mj_steps == 0, "premise: this parametrisation leaves stepping to the engine"
        assert _carry_error(sim, _PARENT, _CHILD) < _CARRY_TOL_M

    def test_an_unattached_body_still_falls_under_the_controller(self, sim_arm_and_cube):
        """No-overreach: the re-pin does not freeze bodies nothing is carrying."""
        sim = sim_arm_and_cube
        assert not sim._world._backend_state.get("kinematic_attachments")
        z_before = _body_z(sim, _CHILD)

        _drive_through_controller(sim, _SubstepBurstController(sim, owns_stepping=True))

        assert _body_z(sim, _CHILD) < z_before - 0.1, "an unattached airborne cube must fall"
