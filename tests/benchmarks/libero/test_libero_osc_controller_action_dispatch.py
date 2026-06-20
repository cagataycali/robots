"""Behavior tests for :class:`_LiberoOSCController` action dispatch.

The OSC controller converts a task-space delta-EEF action
(``{x, y, z, roll, pitch, yaw, gripper}``) into joint torques plus a
gripper open/close command, advancing physics by a fixed number of
substeps per policy step. Its construction normally goes through
``from_sim``, which requires robosuite + a compiled LIBERO scene and is
therefore skipped wherever those optional dependencies (or the scene
cache) are absent - i.e. on the standard CI image.

These tests construct the controller directly through ``__init__`` with a
**fake** OSC controller and a tiny real MuJoCo model, so the
dependency-free action-dispatch contract is exercised on every run:

* the RLDS-to-LIBERO gripper-sign conversion (``-sign(2*v - 1)``),
* the stateful per-substep gripper ramp + bias/weight rescale,
* the fixed substep loop (one ``apply`` advances physics N steps),
* the error paths (``set_goal`` / ``run_controller`` raising, torque
  shape mismatch) that must never crash the eval loop,
* ``reset`` clearing per-episode ramp + log state,
* ``_capture_eef_pose`` (valid site and the ``eef_site_id < 0`` fallback),
* the ``STRANDS_LIBERO_ACTION_LOG`` diagnostic path and its
  malformed-``_MAX`` env fallback.

None of this needs robosuite: ``apply`` only calls ``controller.update``,
``controller.set_goal`` and ``controller.run_controller`` plus
``mujoco.mj_step`` - all of which the fake / the toy model provide.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.benchmarks.libero.adapter import _LiberoOSCController  # noqa: E402

# Toy 2-DOF arm + 2-finger gripper. Actuators in order:
#   0,1 -> arm motors (ctrlrange +-10)
#   2,3 -> gripper position actuators (ctrlrange [0, 0.04])
# This mirrors the LIBERO Panda layout the controller assumes:
# arm_actuator_ids drive torques, gripper_actuator_ids take a rescaled
# [-1, +1] ramp value.
_TOY_XML = """
<mujoco>
  <worldbody>
    <body name="link1">
      <joint name="aj1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
      <body name="ee" pos="0 0 0.2">
        <joint name="aj2" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.02 0.02 0.02"/>
        <body name="f1" pos="0.02 0 0">
          <joint name="g1" type="slide" axis="1 0 0" range="0 0.04"/>
          <geom type="box" size="0.005 0.005 0.02"/>
        </body>
        <body name="f2" pos="-0.02 0 0">
          <joint name="g2" type="slide" axis="-1 0 0" range="0 0.04"/>
          <geom type="box" size="0.005 0.005 0.02"/>
        </body>
        <site name="eef" pos="0 0 0.03"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="aj1" ctrlrange="-10 10"/>
    <motor name="m2" joint="aj2" ctrlrange="-10 10"/>
    <position name="grip1" joint="g1" ctrlrange="0 0.04"/>
    <position name="grip2" joint="g2" ctrlrange="0 0.04"/>
  </actuator>
</mujoco>
"""

_ARM_ACTUATOR_IDS = [0, 1]
_GRIPPER_ACTUATOR_IDS = [2, 3]
_ARM_QPOS_ADDRS = [0, 1]


class _FakeOSC:
    """Minimal stand-in for robosuite's ``OperationalSpaceController``.

    Records the goal and the number of ``run_controller`` calls, and emits
    a constant 2-vector torque (one per arm actuator). This is all
    ``_LiberoOSCController.apply`` touches.
    """

    def __init__(self, torque: tuple[float, float] = (1.0, 2.0)):
        self.goal: np.ndarray | None = None
        self.update_calls = 0
        self.run_calls = 0
        self._torque = np.array(torque, dtype=np.float64)

    def update(self) -> None:
        self.update_calls += 1

    def set_goal(self, delta: np.ndarray) -> None:
        self.goal = np.asarray(delta, dtype=np.float64)

    def run_controller(self) -> np.ndarray:
        self.run_calls += 1
        return self._torque.copy()


def _make_model():
    model = mujoco.MjModel.from_xml_string(_TOY_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _eef_site_id(model) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef")


def _make_controller(
    model,
    data,
    *,
    controller: _FakeOSC | None = None,
    substeps: int = 5,
    eef_site_id: int | None = None,
    arm_qpos_addrs: list[int] | None = _ARM_QPOS_ADDRS,
) -> _LiberoOSCController:
    if eef_site_id is None:
        eef_site_id = _eef_site_id(model)
    return _LiberoOSCController(
        controller=controller if controller is not None else _FakeOSC(),
        sim_shim=None,
        eef_site_name="eef",
        arm_actuator_ids=_ARM_ACTUATOR_IDS,
        gripper_actuator_ids=_GRIPPER_ACTUATOR_IDS,
        model=model,
        data=data,
        physics_substeps_per_control=substeps,
        eef_site_id=eef_site_id,
        arm_qpos_addrs=arm_qpos_addrs,
    )


class TestGripperBiasWeight:
    def test_bias_and_weight_derive_from_ctrlrange(self):
        """The [-1,+1] -> [ctrl_lo, ctrl_hi] rescale constants are cached
        once at construction from ``model.actuator_ctrlrange``:
        ``bias = 0.5*(hi+lo)``, ``weight = 0.5*(hi-lo)``. For a [0, 0.04]
        finger that is bias=0.02, weight=0.02."""
        model, data = _make_model()
        ctrl = _make_controller(model, data)
        assert ctrl._gripper_bias.tolist() == [0.02, 0.02]
        assert ctrl._gripper_weight.tolist() == [0.02, 0.02]

    def test_substeps_floor_at_one(self):
        """``physics_substeps_per_control`` is floored at 1 so a degenerate
        zero/negative value never produces a no-op (and never a 0-iteration
        loop that silently drops every action)."""
        model, data = _make_model()
        ctrl = _make_controller(model, data, substeps=0)
        assert ctrl.physics_substeps_per_control == 1


class TestApplyArmAndGripper:
    def test_arm_torques_written_each_substep_and_goal_set_once(self):
        """``apply`` calls ``set_goal`` once then ``run_controller`` once per
        substep, writing the returned torques to the arm actuators. The
        6-dim Cartesian delta packs x/y/z/roll/pitch/yaw in order."""
        model, data = _make_model()
        osc = _FakeOSC(torque=(1.0, 2.0))
        ctrl = _make_controller(model, data, controller=osc, substeps=5)
        data.ctrl[:] = 0.0

        ctrl.apply(
            {"x": 0.05, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "gripper": 0.0},
            model,
            data,
            "robot",
        )

        # set_goal received the packed 6-dim delta, x first.
        assert osc.goal is not None
        assert osc.goal.tolist() == [0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
        # One run_controller per substep.
        assert osc.run_calls == 5
        # Arm actuators hold the last torque vector.
        assert data.ctrl[0] == pytest.approx(1.0)
        assert data.ctrl[1] == pytest.approx(2.0)

    def test_rlds_close_ramps_gripper_current_toward_close(self):
        """RLDS ``gripper=0.0`` means CLOSE. The conversion
        ``-sign(2*v - 1)`` maps it to +1, so ``current_action`` ramps to
        ``[-speed*n, +speed*n]`` and the rescaled ctrl straddles the bias
        (one finger below, one above mid-range)."""
        model, data = _make_model()
        ctrl = _make_controller(model, data, substeps=5)
        data.ctrl[:] = 0.0

        ctrl.apply({"x": 0.0, "gripper": 0.0}, model, data, "robot")

        # 5 substeps * speed 0.01 * sign(+1) -> current_action = [-0.05, +0.05]
        assert ctrl._gripper_current_action.tolist() == pytest.approx([-0.05, 0.05])
        # ctrl = bias + weight*current = 0.02 +- 0.001
        assert data.ctrl[2] == pytest.approx(0.019)
        assert data.ctrl[3] == pytest.approx(0.021)

    def test_rlds_open_ramps_gripper_opposite_direction(self):
        """RLDS ``gripper=1.0`` means OPEN -> conversion gives -1, ramping
        ``current_action`` the opposite way from the close case. This is the
        asymmetry that the per-finger ramp exists to get right."""
        model, data = _make_model()
        ctrl = _make_controller(model, data, substeps=5)
        data.ctrl[:] = 0.0

        ctrl.apply({"x": 0.0, "gripper": 1.0}, model, data, "robot")

        assert ctrl._gripper_current_action.tolist() == pytest.approx([0.05, -0.05])

    def test_gripper_midpoint_produces_no_ramp(self):
        """``gripper=0.5`` -> ``-sign(0) = 0`` -> zero ramp step, so
        ``current_action`` stays put. An action dict without a gripper key
        defaults to 0.5, so an empty-ish action never silently drives the
        gripper (the #168 regression: default 0.0 closed it)."""
        model, data = _make_model()
        ctrl = _make_controller(model, data, substeps=5)

        ctrl.apply({"gripper": 0.5}, model, data, "robot")
        assert ctrl._gripper_current_action.tolist() == pytest.approx([0.0, 0.0])

        ctrl.reset()
        # No gripper key at all -> defaults to 0.5 -> still no ramp.
        ctrl.apply({"x": 0.01}, model, data, "robot")
        assert ctrl._gripper_current_action.tolist() == pytest.approx([0.0, 0.0])

    def test_list_shaped_action_values_are_coerced(self):
        """GR00T-LIBERO packs every channel as a 2-element list to match
        training shape; ``apply`` must coerce via ``_to_scalar`` (first
        element) without raising and still drive the arm."""
        model, data = _make_model()
        osc = _FakeOSC()
        ctrl = _make_controller(model, data, controller=osc, substeps=2)
        data.ctrl[:] = 0.0

        ctrl.apply(
            {"x": [0.05, 0.05], "y": [0.0, 0.0], "gripper": [0.0, 0.0]},
            model,
            data,
            "robot",
        )
        assert osc.goal.tolist()[0] == pytest.approx(0.05)
        assert osc.run_calls == 2


class TestApplyErrorPaths:
    def test_set_goal_failure_still_advances_physics(self):
        """If ``set_goal`` raises, ``apply`` logs and still steps physics the
        full substep count (so eval-loop timing stays aligned) without
        calling ``run_controller``."""
        model, data = _make_model()

        class _BoomGoal(_FakeOSC):
            def set_goal(self, delta):
                raise ValueError("bad goal")

        osc = _BoomGoal()
        ctrl = _make_controller(model, data, controller=osc, substeps=3)
        before = data.time

        ctrl.apply({"x": 0.1}, model, data, "robot")  # must not raise

        assert osc.run_calls == 0
        assert data.time > before  # physics advanced

    def test_run_controller_failure_keeps_previous_arm_ctrl(self):
        """A ``run_controller`` exception leaves the previous arm ctrl in
        place for that substep but does NOT abort the gripper write or the
        step - the eval loop keeps running."""
        model, data = _make_model()

        class _BoomRun(_FakeOSC):
            def run_controller(self):
                raise RuntimeError("solver blew up")

        ctrl = _make_controller(model, data, controller=_BoomRun(), substeps=2)
        data.ctrl[:] = 0.0

        ctrl.apply({"x": 0.1, "gripper": 1.0}, model, data, "robot")  # must not raise

        # Arm ctrl untouched (stayed 0), gripper still rescaled-written.
        assert data.ctrl[0] == pytest.approx(0.0)
        assert data.ctrl[1] == pytest.approx(0.0)
        assert data.ctrl[2] != 0.0 or data.ctrl[3] != 0.0

    def test_torque_shape_mismatch_skips_arm_write(self):
        """If ``run_controller`` returns the wrong number of torques the arm
        write is skipped (logged), never raising or writing a misaligned
        vector."""
        model, data = _make_model()

        class _BadShape(_FakeOSC):
            def run_controller(self):
                return np.array([1.0, 2.0, 3.0])  # 3 != 2 arm actuators

        ctrl = _make_controller(model, data, controller=_BadShape(), substeps=1)
        data.ctrl[:] = 0.0

        ctrl.apply({"x": 0.1}, model, data, "robot")  # must not raise

        assert data.ctrl[0] == pytest.approx(0.0)
        assert data.ctrl[1] == pytest.approx(0.0)


class TestResetAndCapture:
    def test_reset_clears_gripper_ramp_and_log_step(self):
        """``reset`` zeroes the stateful gripper ramp accumulator and the
        per-episode action-log counter so a fresh episode starts from a
        canonical fully-open gripper, not wherever the prior episode ended."""
        model, data = _make_model()
        ctrl = _make_controller(model, data, substeps=5)
        ctrl.apply({"gripper": 0.0}, model, data, "robot")
        assert ctrl._gripper_current_action.tolist() != [0.0, 0.0]

        ctrl.reset()
        assert ctrl._gripper_current_action.tolist() == [0.0, 0.0]
        assert ctrl._action_log_step == 0

    def test_capture_eef_pose_returns_pos_and_unit_quat(self):
        """``_capture_eef_pose`` reads site xpos and a unit wxyz quaternion
        from the site's rotation matrix."""
        model, data = _make_model()
        ctrl = _make_controller(model, data)
        pos, quat = ctrl._capture_eef_pose(data)
        assert pos.shape == (3,)
        assert quat.shape == (4,)
        assert float(np.linalg.norm(quat)) == pytest.approx(1.0, abs=1e-6)

    def test_capture_eef_pose_zero_when_site_id_negative(self):
        """A controller built before EEF-site tracking existed (or in a test
        without a site) has ``eef_site_id < 0`` and must degrade to zero
        arrays rather than indexing with -1."""
        model, data = _make_model()
        ctrl = _make_controller(model, data, eef_site_id=-1)
        pos, quat = ctrl._capture_eef_pose(data)
        assert pos.tolist() == [0.0, 0.0, 0.0]
        assert quat.tolist() == [0.0, 0.0, 0.0, 0.0]


class TestActionLogDiagnostic:
    def test_action_log_emits_and_advances_step_when_enabled(self, monkeypatch, caplog):
        """``STRANDS_LIBERO_ACTION_LOG=1`` emits one structured INFO line per
        ``apply`` and advances the per-episode log counter."""
        monkeypatch.setenv("STRANDS_LIBERO_ACTION_LOG", "1")
        model, data = _make_model()
        ctrl = _make_controller(model, data, substeps=2)
        assert ctrl._action_log_enabled is True

        with caplog.at_level(logging.INFO, logger="strands_robots.benchmarks.libero.adapter"):
            ctrl.apply({"x": 0.05, "gripper": 1.0}, model, data, "robot")

        assert ctrl._action_log_step == 1
        assert any("ACTION_LOG" in r.getMessage() for r in caplog.records)

    def test_action_log_respects_max_cap(self, monkeypatch):
        """Only the first ``STRANDS_LIBERO_ACTION_LOG_MAX`` apply calls log;
        the counter stops advancing past the cap."""
        monkeypatch.setenv("STRANDS_LIBERO_ACTION_LOG", "1")
        monkeypatch.setenv("STRANDS_LIBERO_ACTION_LOG_MAX", "1")
        model, data = _make_model()
        ctrl = _make_controller(model, data, substeps=1)
        assert ctrl._action_log_max == 1

        ctrl.apply({"x": 0.01, "gripper": 0.5}, model, data, "robot")
        ctrl.apply({"x": 0.01, "gripper": 0.5}, model, data, "robot")
        # Capped at 1: counter does not advance past the max.
        assert ctrl._action_log_step == 1

    def test_malformed_log_max_falls_back_to_default(self, monkeypatch):
        """A non-integer ``STRANDS_LIBERO_ACTION_LOG_MAX`` logs a warning and
        falls back to 50 rather than raising at construction."""
        monkeypatch.setenv("STRANDS_LIBERO_ACTION_LOG", "1")
        monkeypatch.setenv("STRANDS_LIBERO_ACTION_LOG_MAX", "not-an-int")
        model, data = _make_model()
        ctrl = _make_controller(model, data)
        assert ctrl._action_log_max == 50

    def test_action_log_disabled_by_default(self):
        """Without the env var the diagnostic path is off (zero overhead) and
        ``apply`` never advances the log counter."""
        model, data = _make_model()
        ctrl = _make_controller(model, data, substeps=1)
        assert ctrl._action_log_enabled is False
        ctrl.apply({"x": 0.01, "gripper": 0.5}, model, data, "robot")
        assert ctrl._action_log_step == 0
