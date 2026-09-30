"""WBCLatentTorqueController and the MuJoCo hook: SONIC PD on all 29 joints, installed for the policy tree.

The physics half runs the real Menagerie G1 for a handful of ticks with a stub
decoder session (zero action = hold the SONIC stance); no network, no weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.policies.base import Policy
from strands_robots.policies.wbc_latent import (
    SONIC_DEFAULT_ANGLES,
    SONIC_JOINT_NAMES,
    SONIC_KDS,
    SONIC_KPS,
    SonicDecoder,
    WBCLatentPolicy,
    WBCLatentTorqueController,
    install_wbc_latent_torque_control,
    wbc_latent_uses_position_servo,
)
from strands_robots.policies.wbc_latent.constants import NUM_JOINTS
from strands_robots.policies.wbc_latent.policy import TOKEN_KEYS

pytest.importorskip("mujoco")


class ZeroSession:
    def run(self, output_names, input_feed):
        return [np.zeros((1, NUM_JOINTS), dtype=np.float32)]


class StubVLA(Policy):
    requires_images = False

    @property
    def provider_name(self) -> str:
        return "stub_vla"

    def set_robot_state_keys(self, robot_state_keys):
        pass

    async def get_actions(self, observation_dict, instruction, **kwargs):
        d = {k: 0.0 for k in TOKEN_KEYS}
        d["left_gripper"] = 0.5
        d["right_gripper"] = 0.25
        return [d] * 10


def _policy() -> WBCLatentPolicy:
    return WBCLatentPolicy(StubVLA(), decoder=SonicDecoder(session=ZeroSession()))


@pytest.fixture
def g1_sim():
    from strands_robots.simulation import create_simulation

    sim = create_simulation(backend="mujoco", headless=True)
    assert sim.create_world()["status"] == "success"
    assert sim.add_robot("unitree_g1")["status"] == "success"
    return sim


def test_constructor_refuses_a_partial_joint_set():
    with pytest.raises(ValueError, match="29"):
        WBCLatentTorqueController(
            _policy(),
            actuator_ids=[0, 1],
            qpos_addrs=[0, 1],
            dof_addrs=[0, 1],
            saved_actuator_gains={},
            model=None,
            world=None,
        )


def test_pd_law_uses_the_sonic_gain_table():
    ctrl = WBCLatentTorqueController(
        _policy(),
        actuator_ids=list(range(NUM_JOINTS)),
        qpos_addrs=list(range(NUM_JOINTS)),
        dof_addrs=list(range(NUM_JOINTS)),
        saved_actuator_gains={},
        model=None,
        world=None,
    )
    q = SONIC_DEFAULT_ANGLES - 0.1
    dq = np.full(NUM_JOINTS, 0.2)
    np.testing.assert_allclose(ctrl.compute_torques(q, dq), SONIC_KPS * 0.1 - SONIC_KDS * 0.2)


def test_predicate_and_install_flip_servos_to_torque_and_restore(g1_sim):
    import mujoco as mj

    sim = g1_sim
    assert wbc_latent_uses_position_servo(sim, "unitree_g1") is True
    model = sim._world._model
    pol = _policy()
    ctrl = install_wbc_latent_torque_control(sim, pol, "unitree_g1")
    assert sim._world._backend_state["action_controller"] is ctrl
    assert len(ctrl.actuator_ids) == NUM_JOINTS
    for ai in ctrl.actuator_ids:
        assert int(model.actuator_biastype[ai]) == int(mj.mjtBias.mjBIAS_NONE)
    assert wbc_latent_uses_position_servo(sim, "unitree_g1") is False
    assert model.opt.timestep == pytest.approx(0.005)
    # the stance and base height were seeded
    data = sim._world._data
    np.testing.assert_allclose(data.qpos[ctrl.qpos_addrs], SONIC_DEFAULT_ANGLES, atol=1e-9)
    ctrl.uninstall()
    assert "action_controller" not in sim._world._backend_state
    assert wbc_latent_uses_position_servo(sim, "unitree_g1") is True


def test_apply_holds_the_stance_for_forty_ticks_and_records_grippers(g1_sim):
    import mujoco as mj

    sim = g1_sim
    pol = _policy()
    ctrl = install_wbc_latent_torque_control(sim, pol, "unitree_g1")
    model, data = sim._world._model, sim._world._data
    pelvis = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis < 0:
        pelvis = next(
            i for i in range(model.nbody) if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or "").endswith("pelvis")
        )
    action = pol.decoder.targets_dict(SONIC_DEFAULT_ANGLES)
    action["left_gripper"] = 0.5
    action["right_gripper"] = 0.25
    z0 = float(data.xpos[pelvis][2])
    for _ in range(40):
        ctrl.apply(action, model, data, "unitree_g1")
    assert data.time == pytest.approx(40 * 4 * 0.005)
    assert float(data.xpos[pelvis][2]) > z0 - 0.08  # no collapse under the SONIC gains
    assert np.all(np.isfinite(ctrl.last_tau)) and np.abs(ctrl.last_tau).max() < 200.0
    assert ctrl.last_grippers == (0.5, 0.25)
    # a missing joint key holds; a bad value holds
    ctrl.apply({"left_knee_joint": "not a number"}, model, data, "unitree_g1")
    assert ctrl.target_q[3] == pytest.approx(SONIC_DEFAULT_ANGLES[3])
    ctrl.uninstall()


def test_engine_hook_installs_for_the_policy_tree_and_run_policy_returns_success(g1_sim):
    sim = g1_sim
    pol = _policy()

    class Wrapper(Policy):
        """A wrapper that declares its child, as composite/persistent do."""

        provider_name = "wrapper"

        @property
        def children(self):
            return (pol,)

        def set_robot_state_keys(self, keys):
            pol.set_robot_state_keys(keys)

        async def get_actions(self, o, i, **k):
            return await pol.get_actions(o, i, **k)

    outcome = sim._maybe_install_action_controller(Wrapper(), "unitree_g1")
    assert callable(outcome), outcome
    assert type(sim._world._backend_state["action_controller"]).__name__ == "WBCLatentTorqueController"
    outcome()
    assert "action_controller" not in sim._world._backend_state
    res = sim.run_policy(
        "unitree_g1", policy_object=pol, instruction="stand", n_steps=5, control_frequency=50.0, fast_mode=True
    )
    assert res["status"] == "success", res
    assert pol.decoder.ticks == 5
    assert "action_controller" not in sim._world._backend_state  # cleaned up after the run


def test_opt_out_leaves_the_servos_alone(g1_sim):
    sim = g1_sim
    pol = _policy()
    res = sim.run_policy(
        "unitree_g1",
        policy_object=pol,
        instruction="stand",
        n_steps=2,
        control_frequency=50.0,
        fast_mode=True,
        wbc_install_torque_control=False,
    )
    assert res["status"] in ("success", "error")
    assert wbc_latent_uses_position_servo(sim, "unitree_g1") is True


def test_joint_names_are_the_menagerie_g1_names(g1_sim):
    import mujoco as mj

    model = g1_sim._world._model
    pfx = g1_sim._world.robots["unitree_g1"].namespace
    for n in SONIC_JOINT_NAMES:
        assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, pfx + n) >= 0, pfx + n
