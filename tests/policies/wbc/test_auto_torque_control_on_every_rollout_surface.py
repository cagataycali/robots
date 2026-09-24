"""Regression tests for the WBC auto-torque-control path on every rollout surface.

:class:`WBCPolicy` emits joint-**position** targets. The stock
``Robot("unitree_g1")`` ships position-servo actuators with a uniform
``kp=500`` gain that overrides SONIC's tuned per-joint PD, so a bare
``sim.run_policy(policy_provider="wbc", robot_name="unitree_g1")`` drove the
servos directly and the gait diverged within a fraction of a second - the
documented quickstart silently fell over.

The fix gives the MuJoCo engine a ``_maybe_install_wbc_torque_control`` hook
that a rollout surface invokes after binding the policy: when a WBCPolicy meets
a position-servo scene it auto-installs the torque shim for the duration of the
call and restores the actuators afterwards. The opt-out is the
``wbc_install_torque_control=False`` kwarg.

``TestEveryRolloutSurfaceInstallsTheShim`` pins that all three surfaces install
it, not just ``run_policy``. The hook was read there alone, so the SCORED
surfaces - ``eval_policy`` and ``evaluate_benchmark``, whose whole output is a
success rate - drove WBC's position targets into the stock servo gain. Measured
on ``Robot("unitree_g1")`` with the published
``GR00T-WholeBodyControl-{Balance,Walk}.onnx`` weights at 50 Hz on the shipped
``g1_walk_forward`` spec: through ``evaluate_benchmark`` the pelvis fell 0.797 m
-> 0.393 m, the spec's ``base_below_z`` failure fired at step 107 and the
benchmark reported ``success_rate: 0.0`` under ``status="success"``; the
identical call with the shim installed held 0.733 m, walked past the 2 m goal at
step 139 and scored ``success_rate: 1.0`` (``avg_reward`` 40.3 vs 153.0). A
success rate carries no field saying which pipeline produced it, so the 0% read
as an honest policy failure.

These run WITHOUT real SONIC weights (stub ONNX session, real config + joint
mapping) on the real torque/position-servo G1 model. The end-to-end "does it
actually WALK" validation needs real weights and lives in the gated
integration suite.

``TestAutoInstallHookThroughWrappers`` pins that the shim is keyed on the WBC
policy driving the joints rather than on the type of object handed to
``run_policy``: the same policy inside a ``CompositePolicy`` (legs from WBC,
arms from a manipulation policy - the composition WBC's own docs recommend) or a
``PersistentPolicy`` needs the identical shim, because the position-servo gain
it corrects is a property of the scene and the policy, not of the wrapper.

``TestAutoInstallHook`` drives the install path and every no-op condition the
hook documents: no ``[wbc]`` extra, a non-WBC policy, no compiled world, a
controller already registered, and ``wbc_uses_position_servo`` reporting no
position-servo actuator - which it does for two distinct scenes (actuators
already flipped to torque, and a scene holding none of the WBC joints). Those
last three matter because a no-op is indistinguishable from a hook that never
ran: each one is the difference between leaving a scene alone and silently
converting somebody else's actuators.
"""

from __future__ import annotations

import ast
import inspect
import logging
import sys
import textwrap
import threading
from typing import cast

import numpy as np
import pytest

from strands_robots.policies import MockPolicy
from strands_robots.policies.wbc import (
    WBC_G1_ALL_JOINTS,
    WBCConfig,
    WBCPolicy,
    WBCTorqueController,
    wbc_uses_position_servo,
)
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.benchmark import BenchmarkProtocol, StepInfo

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


# A scene holding none of the WBC joints: ``wbc_uses_position_servo`` cannot
# resolve a driven joint against it and conservatively reports False. Declared
# here rather than imported so this module needs no G1 assets for that case.
_XML_NO_WBC_JOINTS = """
<mujoco>
  <worldbody>
    <body name="b">
      <joint name="unrelated_joint" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _hook_no_op_guards() -> int:
    """Count the hook's ``return None`` early-outs by AST.

    The hook's only other exit returns the cleanup callable, so this is exactly
    the number of conditions under which it declines to touch the scene.
    """
    from strands_robots.simulation.mujoco.simulation import Simulation

    src = textwrap.dedent(inspect.getsource(Simulation._maybe_install_wbc_torque_control))
    fn = ast.parse(src).body[0]
    return sum(
        1
        for node in ast.walk(fn)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and node.value.value is None
    )


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

    def test_cleanup_unregisters_so_a_second_rollout_gets_the_shim(self) -> None:
        """The cleanup must undo the registry write as well as the gains.

        ``install_wbc_torque_control`` flips the driven actuators to torque
        *and* registers the controller for ``_apply_sim_action``;
        ``WBCTorqueController.uninstall`` only restores the gains. A cleanup
        that stopped there left the controller registered on a scene whose
        actuators were back to position servos - so it kept dispatching PD
        torques into servos that read them as position targets - and the
        "a manually-installed controller wins" check above then declined to
        install on the next ``run_policy``, leaving the second rollout without
        the shim the first one needed.
        """
        model, data = _build_g1_model()
        sim = _mujoco_sim_with_world(model, data)

        cleanup = sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1")
        assert callable(cleanup)
        controller = sim._world._backend_state["action_controller"]
        driven = controller.leg_waist_actuator_ids[0]
        assert int(model.actuator_biastype[driven]) == int(mujoco.mjtBias.mjBIAS_NONE)

        cleanup()

        # Both halves undone, not just the gains.
        assert int(model.actuator_biastype[driven]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
        assert "action_controller" not in sim._world._backend_state

        # ... so the next rollout on the same sim installs the shim again.
        again = sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1")
        assert callable(again), "the second rollout must get the shim too"
        assert int(model.actuator_biastype[driven]) == int(mujoco.mjtBias.mjBIAS_NONE)
        again()

    def test_cleanup_leaves_a_controller_it_did_not_install(self) -> None:
        """Only the entry this hook wrote is removed.

        A manual installation that happens to be registered while an
        auto-installed cleanup runs must survive it - the cleanup owns exactly
        what its own ``install_wbc_torque_control`` call wrote.
        """
        model, data = _build_g1_model()
        sim = _mujoco_sim_with_world(model, data)

        cleanup = sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1")
        assert callable(cleanup)
        sentinel = object()
        sim._world._backend_state["action_controller"] = sentinel

        cleanup()

        assert sim._world._backend_state["action_controller"] is sentinel

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

    def test_skips_when_the_wbc_extra_is_absent(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # A minimal install has no [wbc] extra, so the hook's import fails and
        # run_policy must carry on unchanged rather than raise out of binding.
        premise = _mujoco_sim_with_world(*_build_g1_model())
        assert callable(premise._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1")), (
            "premise: this pair installs the shim while [wbc] is importable"
        )

        sim = _mujoco_sim_with_world(*_build_g1_model())
        policy = _g1_policy()  # built while the extra is still importable
        monkeypatch.setitem(sys.modules, "strands_robots.policies.wbc", None)
        assert sim._maybe_install_wbc_torque_control(policy, "unitree_g1") is None
        assert "action_controller" not in sim._world._backend_state

    def test_skips_without_a_world(self) -> None:
        from strands_robots.simulation.mujoco.simulation import Simulation

        sim = Simulation()
        assert sim._world is None, "premise: a bare engine has no world yet"
        assert sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1") is None

    def test_skips_when_the_world_has_no_compiled_model(self) -> None:
        from strands_robots.simulation.mujoco.simulation import Simulation

        sim = Simulation()
        # Built directly rather than through _mujoco_sim_with_world, whose
        # namespace probe needs a compiled model. Held in a local so the
        # assertions read the world under test rather than the engine's
        # SimWorld | None attribute.
        world = _FakeWorld(None, None, "")
        sim._world = world  # type: ignore[assignment]
        assert world._model is None
        assert sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1") is None
        assert "action_controller" not in world._backend_state

    def test_skips_when_the_driven_actuators_are_already_torque_motors(self) -> None:
        model, data = _build_g1_model()
        sim = _mujoco_sim_with_world(model, data)
        # from_sim is install_wbc_torque_control minus the registry write, so
        # this is the "already torque mode" scene rather than the "controller
        # already registered" one test_skips_when_controller_already_installed
        # covers - the two conditions are checked separately and in that order.
        controller = WBCTorqueController.from_sim(cast(SimEngine, sim), _g1_policy(), "unitree_g1")
        assert "action_controller" not in sim._world._backend_state
        assert controller.leg_waist_actuator_ids, "premise: driven actuators resolved"
        for ai in controller.leg_waist_actuator_ids:
            assert int(model.actuator_biastype[ai]) == int(mujoco.mjtBias.mjBIAS_NONE)
        assert wbc_uses_position_servo(cast(SimEngine, sim), _g1_policy(), "unitree_g1") is False

        assert sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1") is None
        assert "action_controller" not in sim._world._backend_state

    def test_skips_when_no_wbc_joint_resolves_in_the_scene(self) -> None:
        model = mujoco.MjModel.from_xml_string(_XML_NO_WBC_JOINTS)
        sim = _mujoco_sim_with_world(model, mujoco.MjData(model))
        assert wbc_uses_position_servo(cast(SimEngine, sim), _g1_policy(), "unitree_g1") is False

        assert sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1") is None
        assert "action_controller" not in sim._world._backend_state


class TestEveryNoOpConditionIsDriven:
    def test_the_hook_declines_in_exactly_the_five_ways_this_module_drives(self) -> None:
        """A sixth no-op guard is a condition nothing above exercises.

        The five, in check order: no ``[wbc]`` extra; a non-WBC policy; no
        compiled world; a controller already registered; no position-servo
        actuator. Adding a guard without a test fails here.
        """
        assert _hook_no_op_guards() == 5


class TestAutoInstallHookThroughWrappers:
    """The shim resolves a WBCPolicy declared through ``Policy.children``.

    A wrapper is a different object than the policy it wraps, so the hook's
    ``isinstance`` test saw no WBCPolicy and skipped the install - leaving the
    composition WBC's own documentation recommends (legs+waist from WBC, arms
    from a manipulation policy) driving the stock uniform-gain position servos
    that override SONIC's tuned per-joint PD. The hook now walks the declared
    policy tree, so the shim follows the policy rather than the wrapper's type.
    """

    def _sim(self):  # type: ignore[no-untyped-def]
        model, data = _build_g1_model()
        return _mujoco_sim_with_world(model, data)

    def test_composite_wrapping_wbc_gets_the_shim(self) -> None:
        from strands_robots.policies.composite import CompositePolicy

        sim = self._sim()
        composite = CompositePolicy(lower=_g1_policy(), upper=MockPolicy())
        undo = sim._maybe_install_wbc_torque_control(composite, "unitree_g1")
        assert undo is not None, "a WBCPolicy inside a CompositePolicy still needs the torque shim"
        assert isinstance(sim._world._backend_state["action_controller"], WBCTorqueController)
        undo()

    def test_the_shim_is_built_for_the_wbc_child_not_the_wrapper(self) -> None:
        # The controller runs the child's PD law, so it must hold that child -
        # a controller built around the wrapper could not compute torques at all.
        from strands_robots.policies.composite import CompositePolicy

        sim = self._sim()
        wbc = _g1_policy()
        composite = CompositePolicy(lower=wbc, upper=MockPolicy())
        undo = sim._maybe_install_wbc_torque_control(composite, "unitree_g1")
        assert undo is not None
        controller = cast(WBCTorqueController, sim._world._backend_state["action_controller"])
        assert controller.policy is wbc
        undo()

    def test_persistent_wrapping_wbc_gets_the_shim(self) -> None:
        from strands_robots.policies.persistent import PersistentPolicy

        sim = self._sim()
        wbc = _g1_policy()
        undo = sim._maybe_install_wbc_torque_control(PersistentPolicy("wbc", policy_object=wbc), "unitree_g1")
        assert undo is not None, "a WBCPolicy held warm by a PersistentPolicy still needs the torque shim"
        assert cast(WBCTorqueController, sim._world._backend_state["action_controller"]).policy is wbc
        undo()

    def test_a_wrapper_holding_no_wbc_policy_is_still_a_no_op(self) -> None:
        # The walk must not turn every wrapped policy into a WBC install: a
        # composite of two non-WBC policies leaves the scene alone.
        from strands_robots.policies.composite import CompositePolicy

        sim = self._sim()
        composite = CompositePolicy(lower=MockPolicy(), upper=MockPolicy())
        assert sim._maybe_install_wbc_torque_control(composite, "unitree_g1") is None
        assert "action_controller" not in sim._world._backend_state

    def test_wbc_nested_two_wrappers_deep_is_still_found(self) -> None:
        from strands_robots.policies.composite import CompositePolicy
        from strands_robots.policies.persistent import PersistentPolicy

        sim = self._sim()
        wbc = _g1_policy()
        nested = CompositePolicy(lower=PersistentPolicy("wbc", policy_object=wbc), upper=MockPolicy())
        undo = sim._maybe_install_wbc_torque_control(nested, "unitree_g1")
        assert undo is not None
        assert cast(WBCTorqueController, sim._world._backend_state["action_controller"]).policy is wbc
        undo()


# ---------------------------------------------------------------------------
# A backend that cannot install the shim
# ---------------------------------------------------------------------------


class _OtherBackendSim(SimEngine):
    """A minimal engine that inherits the base hook instead of overriding it.

    The shipped Newton and Isaac engines are in exactly this position: the shim
    is written against a compiled ``MjModel`` / ``MjData`` pair, so neither can
    install it. ``get_observation`` answers the WBC-shaped frame the policy
    reads, so the rollout the base default refuses is one that would otherwise
    have run.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.sends = 0

    def create_world(self, timestep=None, gravity=None, ground_plane=True):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def destroy(self):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def reset(self):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def step(self, n_steps: int = 1):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def get_state(self):  # type: ignore[no-untyped-def]
        return {"sim_time": 0.0, "step_count": self.sends}

    def add_robot(self, name, **kw):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def remove_robot(self, name):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return ["unitree_g1"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(WBC_G1_ALL_JOINTS)

    def add_object(self, name, **kw):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def remove_object(self, name):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):  # type: ignore[no-untyped-def]
        obs: dict[str, object] = {
            "base_pos": [0.0, 0.0, 0.793],
            "base_quat": [1.0, 0.0, 0.0, 0.0],
            "base_ang_vel": [0.0, 0.0, 0.0],
            "base_lin_vel": [0.0, 0.0, 0.0],
        }
        for name in WBC_G1_ALL_JOINTS:
            obs[name] = 0.0
            obs[f"{name}.vel"] = 0.0
        return obs

    def send_action(self, action, robot_name=None, n_substeps=1):  # type: ignore[no-untyped-def]
        self.sends += 1

    def render(self, camera_name="default", width=None, height=None):  # type: ignore[no-untyped-def]
        return {"image": np.zeros((height or 48, width or 64, 3), dtype=np.uint8)}


def _wrapped(shape: str, wbc: WBCPolicy):  # type: ignore[no-untyped-def]
    """The policy object handed to ``run_policy``, for each declared shape."""
    from strands_robots.policies.composite import CompositePolicy
    from strands_robots.policies.persistent import PersistentPolicy

    if shape == "bare":
        return wbc
    if shape == "composite":
        return CompositePolicy(lower=wbc, upper=MockPolicy())
    return PersistentPolicy("wbc", policy_object=wbc)


def _run_on(sim: _OtherBackendSim, policy, **kwargs):  # type: ignore[no-untyped-def]
    return sim.run_policy(
        "unitree_g1", policy_object=policy, n_steps=3, control_frequency=50.0, fast_mode=True, **kwargs
    )


def _text_of(result: dict) -> str:
    return " ".join(b.get("text", "") for b in result.get("content") or [] if isinstance(b, dict))


class TestABackendThatCannotInstallTheShimRefusesInsteadOfFalling:
    """The requirement is reported, not absorbed.

    Only the MuJoCo engine overrides the hook, so every other backend ran WBC's
    position targets straight into the stock servo gain and reported success.
    Measured on the shipped Newton backend, stock ``Robot("unitree_g1",
    backend="newton")``, walk weights, 50 Hz: no controller was registered for
    any step, the pelvis sank 0.793 m -> 0.509 m in 0.8 s (0.074 m by 1 s) and
    the envelope read ``status="success"`` with ``action_errors: 0``. The same
    call on MuJoCo held 0.740 m and walked. The fall was the only report.
    """

    @pytest.mark.parametrize("shape", ["bare", "composite", "persistent"])
    def test_the_rollout_is_refused_naming_the_backend_and_both_remedies(self, shape: str) -> None:
        sim = _OtherBackendSim()
        result = _run_on(sim, _wrapped(shape, _g1_policy()))

        assert result["status"] == "error", _text_of(result)
        text = _text_of(result)
        assert "_OtherBackendSim" in text, text
        assert 'backend="mujoco"' in text, text
        assert "wbc_install_torque_control=False" in text, text
        assert sim.sends == 0, "a refused rollout applies no action"
        payload = next(b["json"] for b in result["content"] if "json" in b)
        assert payload["steps_used"] == 0

    def test_the_documented_opt_out_still_rolls_out(self) -> None:
        """``wbc_install_torque_control=False`` is the torque-scene path, unchanged."""
        sim = _OtherBackendSim()
        result = _run_on(sim, _g1_policy(), wbc_install_torque_control=False)

        assert result["status"] == "success", _text_of(result)
        assert sim.sends == 3

    def test_a_non_wbc_policy_is_untouched(self) -> None:
        sim = _OtherBackendSim()
        result = _run_on(sim, MockPolicy())

        assert result["status"] == "success", _text_of(result)
        assert sim.sends == 3

    def test_the_mujoco_engine_installs_rather_than_reports(self) -> None:
        """The override wins: a backend that can install one never refuses."""
        sim = _mujoco_sim_with_world(*_build_g1_model())
        outcome = sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1")

        assert not isinstance(outcome, str), outcome
        assert callable(outcome)
        outcome()


# ---------------------------------------------------------------------------
# Every rollout surface installs the controller its policy needs
# ---------------------------------------------------------------------------


class _StandSpec(BenchmarkProtocol):
    """A three-step G1 spec that neither succeeds nor fails.

    The shipped ``g1_walk_forward`` would do, but its 1000-step horizon buys
    nothing here: what is under test is whether the controller is registered
    while the policy is being driven, which the first step already answers.
    """

    max_steps = 3

    @property
    def supported_robots(self) -> list[str]:
        return ["unitree_g1"]

    @property
    def default_robot(self) -> str:
        return "unitree_g1"

    def on_episode_start(self, sim, rng) -> None:  # type: ignore[no-untyped-def]
        return None

    def on_step(self, sim, obs, action):  # type: ignore[no-untyped-def]
        return StepInfo(reward=0.0)

    def is_success(self, sim) -> bool:  # type: ignore[no-untyped-def]
        return False

    def is_failure(self, sim) -> bool:  # type: ignore[no-untyped-def]
        return False


def _drive(sim, surface: str, policy, **kwargs):  # type: ignore[no-untyped-def]
    """Roll ``policy`` out through one of the three rollout surfaces."""
    if surface == "run_policy":
        return sim.run_policy("unitree_g1", policy_object=policy, n_steps=3, fast_mode=True, **kwargs)
    if surface == "eval_policy":
        return sim.eval_policy("unitree_g1", policy_object=policy, max_steps=3, n_episodes=1, **kwargs)
    return sim.evaluate_benchmark(_SPEC_NAME, robot_name="unitree_g1", policy_object=policy, **kwargs)


_SURFACES = ["run_policy", "eval_policy", "evaluate_benchmark"]
_SPEC_NAME = "wbc_shim_surface_probe"


@pytest.fixture
def stand_spec():  # type: ignore[no-untyped-def]
    from strands_robots.simulation.benchmark import register_benchmark, unregister_benchmark

    register_benchmark(_SPEC_NAME, _StandSpec())
    try:
        yield _SPEC_NAME
    finally:
        unregister_benchmark(_SPEC_NAME)


class TestEveryRolloutSurfaceInstallsTheShim:
    """The controller is live while the policy drives, on all three surfaces.

    Read through the policy's own inference call rather than a per-surface hook
    (``run_policy`` takes an ``observer``, the eval surfaces an ``on_frame``):
    the session records what the scene's controller registry held at each step,
    which is the fact the gait depends on and is spelled the same way whoever
    asks for the rollout.
    """

    @pytest.fixture
    def g1(self):  # type: ignore[no-untyped-def]
        from strands_robots import Robot
        from strands_robots.simulation.model_registry import resolve_model

        if not resolve_model("unitree_g1"):
            pytest.skip("unitree_g1 model assets not available")
        return Robot("unitree_g1", mesh=False)

    @staticmethod
    def _recording_policy(sim, seen: list) -> WBCPolicy:  # type: ignore[no-untyped-def]
        policy = _g1_policy()
        stub = policy.policy_session

        class _Recording:
            def get_inputs(self):  # type: ignore[no-untyped-def]
                return stub.get_inputs()

            def run(self, output_names, feed):  # type: ignore[no-untyped-def]
                seen.append(sim._world._backend_state.get("action_controller"))
                return stub.run(output_names, feed)

        policy.policy_session = _Recording()
        return policy

    @pytest.mark.parametrize("surface", _SURFACES)
    def test_the_shim_is_live_for_every_step_and_gone_after(self, g1, stand_spec, surface: str) -> None:
        seen: list = []
        result = _drive(g1, surface, self._recording_policy(g1, seen))

        assert result["status"] == "success", _text_of(result)
        assert seen, f"{surface}: the policy was never queried, so nothing was measured"
        assert all(isinstance(c, WBCTorqueController) for c in seen), (
            f"{surface} drove a WBCPolicy with the scene's controller registry holding "
            f"{[type(c).__name__ for c in seen]} - WBC's joint-position targets went "
            "straight into the stock kp=500 servo gain"
        )
        assert g1._world._backend_state.get("action_controller") is None, (
            f"{surface} left the torque shim installed after the call"
        )

    @pytest.mark.parametrize("surface", _SURFACES)
    def test_the_opt_out_installs_nothing(self, g1, stand_spec, surface: str) -> None:
        seen: list = []
        result = _drive(g1, surface, self._recording_policy(g1, seen), wbc_install_torque_control=False)

        assert result["status"] == "success", _text_of(result)
        assert seen and all(c is None for c in seen), [type(c).__name__ for c in seen]

    @pytest.mark.parametrize("surface", _SURFACES)
    def test_the_posture_is_checked_not_read_by_truthiness(self, g1, stand_spec, surface: str) -> None:
        """``"false"`` is truthy, so reading it would install the shim it spells off."""
        result = _drive(g1, surface, _g1_policy(), wbc_install_torque_control="false")

        assert result["status"] == "error", _text_of(result)
        text = _text_of(result)
        assert "wbc_install_torque_control" in text and surface in text, text


class TestABackendThatCannotInstallRefusesOnEverySurface:
    """The requirement is reported by whichever surface was asked, naming it.

    The refusal exists so a backend without the shim reports the requirement
    instead of rolling out and letting the fall be the only evidence. Reached
    from ``run_policy`` alone it left the two surfaces that publish a number
    doing exactly that.
    """

    @pytest.mark.parametrize("surface", _SURFACES)
    def test_the_surface_names_itself_and_applies_no_action(self, stand_spec, surface: str) -> None:
        sim = _OtherBackendSim()
        result = _drive(sim, surface, _g1_policy())

        assert result["status"] == "error", _text_of(result)
        text = _text_of(result)
        assert text.startswith(f"{surface}:"), text
        assert "_OtherBackendSim" in text and 'backend="mujoco"' in text, text
        assert sim.sends == 0, f"{surface}: a refused rollout applies no action"

    @pytest.mark.parametrize("surface", _SURFACES)
    def test_the_documented_opt_out_still_rolls_out(self, stand_spec, surface: str) -> None:
        sim = _OtherBackendSim()
        result = _drive(sim, surface, _g1_policy(), wbc_install_torque_control=False)

        assert result["status"] == "success", _text_of(result)
        assert sim.sends == 3, sim.sends


class TestEveryRolloutSurfaceReadsTheOneInstaller:
    def test_no_surface_reads_the_hook_directly(self) -> None:
        """A fourth reader of the hook is a fourth chance to forget the refusal.

        The three surfaces install through ``_install_action_controller``, whose
        job is to gate the posture and turn a reason string into a refusal. A
        call site reaching past it is the shape this module's defect had.
        """
        src = inspect.getsource(SimEngine)
        assert src.count("self._maybe_install_wbc_torque_control(") == 1, (
            "_maybe_install_wbc_torque_control has readers other than _install_action_controller; route them through it"
        )
        assert src.count("self._install_action_controller(") == len(_SURFACES)
