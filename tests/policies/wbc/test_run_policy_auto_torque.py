"""Regression tests for the WBC auto-torque-control path on ``run_policy``.

:class:`WBCPolicy` emits joint-**position** targets. The stock
``Robot("unitree_g1")`` ships position-servo actuators with a uniform
``kp=500`` gain that overrides SONIC's tuned per-joint PD, so a bare
``sim.run_policy(policy_provider="wbc", robot_name="unitree_g1")`` drove the
servos directly and the gait diverged within a fraction of a second - the
documented quickstart silently fell over.

The fix gives the MuJoCo engine a ``_maybe_install_wbc_torque_control`` hook
that ``run_policy`` invokes after binding the policy: when a WBCPolicy meets a
position-servo scene it auto-installs the torque shim for the duration of the
call and restores the actuators afterwards. The opt-out is the
``wbc_install_torque_control=False`` kwarg.

These run WITHOUT real SONIC weights (stub ONNX session, real config + joint
mapping) on the real torque/position-servo G1 model. The end-to-end "does it
actually WALK" validation needs real weights and lives in the gated
integration suite.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pytest

from strands_robots.policies import MockPolicy
from strands_robots.policies.wbc import WBCConfig, WBCPolicy, wbc_uses_position_servo
from strands_robots.simulation.base import SimEngine

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")


class _StubSession:
    class _In:
        name = "obs"

    def get_inputs(self):  # type: ignore[no-untyped-def]
        return [self._In()]

    def run(self, output_names, feed):  # type: ignore[no-untyped-def]
        return [np.zeros((1, 15), dtype=np.float32)]


def _g1_policy() -> WBCPolicy:
    cfg = WBCConfig(policy_path="x.onnx")
    p = WBCPolicy(config=cfg, walk=False, allow_missing_models=True)
    p.policy_session = _StubSession()
    return p


def _build_g1_model():  # type: ignore[no-untyped-def]
    from strands_robots.simulation.model_registry import resolve_model

    xml = resolve_model("unitree_g1")
    if not xml:
        pytest.skip("unitree_g1 model assets not available")
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    return model, data


class _FakeRobot:
    def __init__(self, namespace: str) -> None:
        self.namespace = namespace


class _FakeWorld:
    def __init__(self, model, data, namespace) -> None:  # type: ignore[no-untyped-def]
        self._model = model
        self._data = data
        self.robots = {"unitree_g1": _FakeRobot(namespace)}
        self._backend_state: dict = {}


def _namespace_for(model) -> str:  # type: ignore[no-untyped-def]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 1) or ""
    return "unitree_g1/" if name.startswith("unitree_g1/") else ""


def _mujoco_sim_with_world(model, data):  # type: ignore[no-untyped-def]
    """A real MuJoCo Simulation engine whose world holds the given G1 model."""
    from strands_robots.simulation.mujoco.simulation import Simulation

    sim = Simulation()
    sim._world = _FakeWorld(model, data, _namespace_for(model))  # type: ignore[assignment]
    return sim


# ---------------------------------------------------------------------------
# wbc_uses_position_servo predicate
# ---------------------------------------------------------------------------


class TestPositionServoDetection:
    def test_stock_g1_is_position_servo(self) -> None:
        model, data = _build_g1_model()
        sim = _FakeWorld(model, data, _namespace_for(model))
        wrapper = type("S", (), {"_world": sim})()
        policy = _g1_policy()
        assert wbc_uses_position_servo(cast(SimEngine, wrapper), policy, "unitree_g1") is True

    def test_false_after_actuators_flipped_to_torque(self) -> None:
        from strands_robots.policies.wbc import install_wbc_torque_control

        model, data = _build_g1_model()
        world = _FakeWorld(model, data, _namespace_for(model))
        wrapper = type("S", (), {"_world": world})()
        policy = _g1_policy()
        # Flip to torque mode; the predicate must now report "no servo".
        install_wbc_torque_control(cast(SimEngine, wrapper), policy, "unitree_g1")
        assert wbc_uses_position_servo(cast(SimEngine, wrapper), policy, "unitree_g1") is False

    def test_false_without_world(self) -> None:
        wrapper = type("S", (), {"_world": None})()
        assert wbc_uses_position_servo(cast(SimEngine, wrapper), _g1_policy(), "unitree_g1") is False


# ---------------------------------------------------------------------------
# MuJoCo engine auto-install hook
# ---------------------------------------------------------------------------


class TestAutoInstallHook:
    def test_installs_torque_shim_and_cleanup_restores(self, caplog) -> None:  # type: ignore[no-untyped-def]
        model, data = _build_g1_model()
        sim = _mujoco_sim_with_world(model, data)
        policy = _g1_policy()

        driven_before = [int(model.actuator_biastype[ai]) for ai in range(model.nu)]
        assert int(mujoco.mjtBias.mjBIAS_AFFINE) in driven_before  # stock = servo

        with caplog.at_level(logging.INFO):
            cleanup = sim._maybe_install_wbc_torque_control(policy, "unitree_g1")

        assert callable(cleanup), "expected a cleanup callable when shim is installed"
        assert "auto-installed WBC torque control" in caplog.text
        controller = sim._world._backend_state["action_controller"]
        # The driven actuators are now torque motors (biastype NONE).
        for ai in controller.leg_waist_actuator_ids:
            assert int(model.actuator_biastype[ai]) == int(mujoco.mjtBias.mjBIAS_NONE)

        # Cleanup restores the original position-servo gains.
        cleanup()
        first = controller.leg_waist_actuator_ids[0]
        assert int(model.actuator_biastype[first]) == int(mujoco.mjtBias.mjBIAS_AFFINE)

    def test_skips_when_controller_already_installed(self) -> None:
        model, data = _build_g1_model()
        sim = _mujoco_sim_with_world(model, data)
        sim._world._backend_state["action_controller"] = object()  # manual install wins
        assert sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1") is None

    def test_skips_for_non_wbc_policy(self) -> None:
        model, data = _build_g1_model()
        sim = _mujoco_sim_with_world(model, data)
        assert sim._maybe_install_wbc_torque_control(MockPolicy(), "unitree_g1") is None
        assert "action_controller" not in sim._world._backend_state
