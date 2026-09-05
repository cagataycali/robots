"""The substep count one action is held for is the count the caller named.

:class:`WBCTorqueController` declares ``owns_stepping``, so
``physics_substeps_per_control`` is not a hint - it is the number of ``mj_step``
calls :meth:`WBCTorqueController.apply` makes, and therefore the control period
itself: at the SONIC 0.005 s timestep the upstream ``g1_gear_wbc.yaml`` cadence
of ``4`` is one inference per 20 ms (50 Hz).

It arrived through ``max(1, int(physics_substeps_per_control))``, which cannot
report an unusable count - it substitutes a usable one. Measured on the real
``unitree_g1`` scene, one control action then advanced physics by:

=====================  ========  =============  =================  =============
requested              stored    ``mj_step``s   physics s/action   realised rate
=====================  ========  =============  =================  =============
``4`` (SONIC nominal)  ``4``     4              0.0200             50 Hz
``0``                  ``1``     1              0.0050             200 Hz
``-5``                 ``1``     1              0.0050             200 Hz
``True``               ``1``     1              0.0050             200 Hz
``4.9``                ``4``     4              0.0200             50 Hz
``"4"``                ``4``     4              0.0200             50 Hz
=====================  ========  =============  =================  =============

Every row is ``status``-free: the controller constructs, the rollout runs, and
nothing names the count that was replaced. The three ``1`` rows are the damaging
ones - a quarter of the control period the caller asked for - and they are the
same failure ``PolicyRunner._control_substeps`` documents having already been
converted away from for the identical quantity ("It used to be clamped with
``max(1, int(override))``, so ``0``/``-5`` silently collapsed to a single
physics step"). ``float("nan")`` did raise, but as
``ValueError: cannot convert float NaN to integer`` from ``int()`` - naming
neither the parameter nor this class.

Why a wrong count is not merely a slower rollout:
``tests/policies/wbc/test_gait_clock_integrates_at_the_loop_rate.py`` pins that
the gait phase advances by the control period the loop actually runs at, so a
commanded ``gait_frequency`` under a substituted count "means something other
than steps per second ... and the robot walks at a rhythm nobody commanded while
every reported number looks right".

The count is therefore held to the shared domain its siblings already use -
:func:`~strands_robots.utils.positive_whole_number_error`, whose own docstring
names "the physics steps one applied action is held for" as one of its two
families and explains why ``0`` is refused there rather than honoured. The
grounding class below asserts the constructor's verdict IS that function's, so
the two cannot drift, and the first-class table pins the values a caller may
still legitimately ask for.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from strands_robots.policies.wbc import WBCConfig, WBCPolicy, WBCTorqueController
from strands_robots.policies.wbc.sim_control import _CONTROL_DECIMATION, _SIM_DT
from strands_robots.utils import positive_whole_number_error

#: Counts no rollout can be held for, and what the clamp turned each into.
_UNUSABLE = [
    pytest.param(0, id="zero-advances-no-physics"),
    pytest.param(-1, id="negative"),
    pytest.param(-5, id="negative-large"),
    pytest.param(True, id="bool-true-acted-as-one"),
    pytest.param(False, id="bool-false-acted-as-one"),
    pytest.param(4.9, id="fractional-was-truncated"),
    pytest.param(0.5, id="fractional-below-one"),
    pytest.param("4", id="string-was-coerced"),
    pytest.param(None, id="none"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
]

#: Counts that stay first-class, so the guard cannot creep into refusing a
#: cadence a caller may legitimately ask for. ``4.0`` and ``np.int64(4)`` are in
#: the shared domain by documented policy (an integral float computed from a
#: config, a NumPy integer read off a model).
_FIRST_CLASS = [
    pytest.param(_CONTROL_DECIMATION, 4, id="sonic-nominal-four"),
    pytest.param(1, 1, id="one-physics-step-per-action"),
    pytest.param(25, 25, id="high-decimation"),
    pytest.param(4.0, 4, id="integral-float"),
    pytest.param(np.int64(4), 4, id="numpy-integer"),
]


class _StubSession:
    """Minimal onnxruntime session stand-in - no SONIC weights needed."""

    class _In:
        name = "obs"

    def get_inputs(self) -> list[Any]:
        return [self._In()]

    def run(self, output_names: Any, feed: Any) -> list[np.ndarray]:
        return [np.zeros((1, 15), dtype=np.float32)]


def _g1_policy() -> WBCPolicy:
    policy = WBCPolicy(config=WBCConfig(policy_path="x.onnx"), walk=False, allow_missing_models=True)
    policy.policy_session = _StubSession()
    return policy


def _controller(substeps: Any, **overrides: Any) -> WBCTorqueController:
    """Construct the controller with an empty actuator set.

    The count is graded on arrival, before any id or address is read, so the
    refusal rows need no compiled model - which is what keeps them running
    wherever the package is installed rather than only where the G1 assets are.
    """
    kwargs: dict[str, Any] = {
        "leg_waist_actuator_ids": [],
        "arm_actuator_ids": [],
        "leg_waist_qpos_addrs": [],
        "leg_waist_dof_addrs": [],
        "arm_qpos_addrs": [],
        "arm_dof_addrs": [],
        "saved_actuator_gains": {},
        "model": None,
        "physics_substeps_per_control": substeps,
    }
    kwargs.update(overrides)
    return WBCTorqueController(_g1_policy(), **kwargs)


class TestACountNoRolloutCanBeHeldForIsRefused:
    """Regression: the clamp substituted a usable count for an unusable one."""

    @pytest.mark.parametrize("substeps", _UNUSABLE)
    def test_construction_raises(self, substeps: Any) -> None:
        with pytest.raises(ValueError):
            _controller(substeps)

    @pytest.mark.parametrize("substeps", _UNUSABLE)
    def test_the_refusal_names_the_parameter_and_the_class(self, substeps: Any) -> None:
        """A dead-end ``int()`` failure named neither, so both are asserted."""
        with pytest.raises(ValueError) as excinfo:
            _controller(substeps)
        message = str(excinfo.value)
        assert "physics_substeps_per_control" in message, message
        assert "WBCTorqueController" in message, message


class TestACadenceACallerMayAskForStaysFirstClass:
    """Over-reach control: the guard narrows to the unusable counts only."""

    @pytest.mark.parametrize(("substeps", "expected"), _FIRST_CLASS)
    def test_the_count_is_stored_as_the_whole_number_it_names(self, substeps: Any, expected: int) -> None:
        controller = _controller(substeps)
        assert controller.physics_substeps_per_control == expected
        # ``range()`` in ``apply`` takes a true int; an integral float would
        # raise there rather than here.
        assert type(controller.physics_substeps_per_control) is int

    def test_the_default_is_the_upstream_sonic_decimation(self) -> None:
        """Omitting the count keeps the 50 Hz cadence at the 0.005 s timestep."""
        controller = WBCTorqueController(
            _g1_policy(),
            leg_waist_actuator_ids=[],
            arm_actuator_ids=[],
            leg_waist_qpos_addrs=[],
            leg_waist_dof_addrs=[],
            arm_qpos_addrs=[],
            arm_dof_addrs=[],
            saved_actuator_gains={},
            model=None,
        )
        assert controller.physics_substeps_per_control == _CONTROL_DECIMATION
        assert _CONTROL_DECIMATION * _SIM_DT == pytest.approx(0.020)


class TestTheVerdictIsTheSharedDomains:
    """Derived, so the constructor and its siblings cannot drift apart.

    The same quantity reaches ``send_action(n_substeps=)`` on all three
    simulation backends through this function. A row added to either table above
    is graded against it rather than against a restated copy of its rule.
    """

    @pytest.mark.parametrize("substeps", _UNUSABLE + [p.values[0] for p in _FIRST_CLASS])
    def test_the_constructor_refuses_exactly_what_the_domain_refuses(self, substeps: Any) -> None:
        domain_refuses = (
            positive_whole_number_error(substeps, "physics_substeps_per_control", "WBCTorqueController") is not None
        )
        try:
            _controller(substeps)
            constructor_refuses = False
        except ValueError:
            constructor_refuses = True
        assert constructor_refuses is domain_refuses


mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")


class TestTheAcceptedCountIsThePhysicsActuallyAdvanced:
    """The consequence cell: the count is the control period, on a real scene.

    Asserted against the literal the caller passed rather than against the
    stored attribute - deriving the expectation from what was stored is what let
    a substituted count read as correct.
    """

    @pytest.fixture
    def g1(self) -> tuple[Any, Any, str]:
        from strands_robots.simulation.model_registry import resolve_model

        xml = resolve_model("unitree_g1")
        if not xml:
            pytest.skip("unitree_g1 model assets not available")
        model = mujoco.MjModel.from_xml_path(xml)
        data = mujoco.MjData(model)
        namespace = (
            "unitree_g1/"
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 1) or "").startswith("unitree_g1/")
            else ""
        )
        return model, data, namespace

    @pytest.mark.parametrize("requested", [1, 2, 4, 7])
    def test_one_action_advances_the_requested_number_of_physics_steps(
        self, g1: tuple[Any, Any, str], requested: int
    ) -> None:
        from strands_robots.policies.wbc import WBC_G1_LEG_WAIST_JOINTS, install_wbc_torque_control
        from strands_robots.simulation.base import SimEngine

        model, data, namespace = g1

        class _Robot:
            def __init__(self) -> None:
                self.namespace = namespace

        class _World:
            def __init__(self) -> None:
                self._model, self._data = model, data
                self.robots = {"unitree_g1": _Robot()}
                self._backend_state: dict[str, Any] = {}

        class _Sim:
            def __init__(self) -> None:
                self._world = _World()

        resolved = install_wbc_torque_control(cast(SimEngine, _Sim()), _g1_policy(), "unitree_g1")
        policy = resolved.policy
        controller = WBCTorqueController(
            policy,
            leg_waist_actuator_ids=resolved.leg_waist_actuator_ids,
            arm_actuator_ids=resolved.arm_actuator_ids,
            leg_waist_qpos_addrs=resolved.leg_waist_qpos_addrs,
            leg_waist_dof_addrs=resolved.leg_waist_dof_addrs,
            arm_qpos_addrs=resolved.arm_qpos_addrs,
            arm_dof_addrs=resolved.arm_dof_addrs,
            saved_actuator_gains={},
            model=model,
            physics_substeps_per_control=requested,
        )
        action = {name: float(policy.default_angles[i]) for i, name in enumerate(WBC_G1_LEG_WAIST_JOINTS)}
        before = float(data.time)
        controller.apply(action, model, data, "unitree_g1")

        advanced = float(data.time) - before
        assert advanced == pytest.approx(requested * float(model.opt.timestep))
        # The whole point of the count: the control period the gait clock is
        # integrated at is this many physics steps of the SONIC timestep.
        assert advanced == pytest.approx(requested * _SIM_DT)
