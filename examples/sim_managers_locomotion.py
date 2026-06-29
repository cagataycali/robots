"""Drive a MuJoCo humanoid through the config-driven sim_managers framework.

Builds a full locomotion env recipe (command + observation + reward +
termination) from a single YAML file, then steps a Unitree G1 in headless
MuJoCo and feeds each step's physics state into the managers. It prints the
running observation dimension, the per-term reward breakdown, and the
termination classification - demonstrating that the same declarative recipe
that a trainer would consume runs end to end on real simulator data.

The ``_env_state_from_mujoco`` helper shows how a backend populates the
backend-agnostic :class:`~strands_robots.sim_managers.EnvState`: terms never
touch MuJoCo, so the identical recipe runs on any simulator that can fill an
EnvState.

Run (headless):

    MUJOCO_GL=egl python examples/sim_managers_locomotion.py
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

from strands_robots import create_simulation
from strands_robots.sim_managers import EnvState, load_managers_config

CONFIG_PATH = Path(__file__).with_suffix(".yaml")
ROBOT = "unitree_g1"
N_STEPS = 200
SUBSTEPS = 10  # physics steps per control step


def _env_state_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    last_action: np.ndarray,
    dt: float,
    step_count: int,
    max_episode_length: int,
) -> EnvState:
    """Populate an EnvState from a floating-base MuJoCo model/data.

    Assumes a free joint at the root: ``qpos[:3]`` base position, ``qpos[3:7]``
    base quaternion ``[w, x, y, z]``, ``qvel[:3]`` world linear velocity,
    ``qvel[3:6]`` world angular velocity, with the actuated DOFs following.
    """
    quat = np.array(data.qpos[3:7], dtype=np.float64)
    inv_quat = np.zeros(4)
    mujoco.mju_negQuat(inv_quat, quat)

    base_lin_vel = np.zeros(3)
    base_ang_vel = np.zeros(3)
    projected_gravity = np.zeros(3)
    mujoco.mju_rotVecQuat(base_lin_vel, np.array(data.qvel[:3]), inv_quat)
    mujoco.mju_rotVecQuat(base_ang_vel, np.array(data.qvel[3:6]), inv_quat)
    mujoco.mju_rotVecQuat(projected_gravity, np.array([0.0, 0.0, -1.0]), inv_quat)

    joint_pos = np.array(data.qpos[7:], dtype=np.float64)
    joint_vel = np.array(data.qvel[6:], dtype=np.float64)
    joint_torque = np.array(data.qfrc_actuator[6:], dtype=np.float64)
    joint_acc = np.array(data.qacc[6:], dtype=np.float64)

    return EnvState(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        action=last_action,
        last_action=last_action,
        base_lin_vel=base_lin_vel,
        base_ang_vel=base_ang_vel,
        projected_gravity=projected_gravity,
        base_height=float(data.qpos[2]),
        base_quat=quat,
        joint_torque=joint_torque,
        joint_acc=joint_acc,
        dt=dt,
        step_count=step_count,
        max_episode_length=max_episode_length,
    )


def _extract_png_bytes(render: object) -> bytes | None:
    """Pull the PNG bytes out of a ``render()`` agent-tool content payload."""
    if not isinstance(render, dict):
        return None
    for block in render.get("content", []) or []:
        image = block.get("image") if isinstance(block, dict) else None
        if isinstance(image, dict):
            source = image.get("source", {})
            data = source.get("bytes") if isinstance(source, dict) else None
            if isinstance(data, bytes):
                return data
    return None


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    managers = load_managers_config(CONFIG_PATH)
    assert managers.command and managers.observation and managers.reward and managers.termination

    sim = create_simulation(backend="mujoco")
    sim.create_world()
    if sim.add_robot(ROBOT).get("status") != "success":
        raise RuntimeError(f"failed to add robot {ROBOT!r}")
    sim.reset()

    model = sim._world._model
    data = sim._world._data
    dt = float(model.opt.timestep) * SUBSTEPS

    rng = np.random.default_rng(0)
    managers.command.reset(rng=rng)
    managers.reward.reset()

    n_act = int(model.nu)
    last_action = np.zeros(n_act)
    reward_totals: dict[str, float] = {}
    total_reward = 0.0
    obs_dim = 0

    for step in range(N_STEPS):
        sim.step(SUBSTEPS)
        state = _env_state_from_mujoco(
            model,
            data,
            last_action=last_action,
            dt=dt,
            step_count=step,
            max_episode_length=N_STEPS,
        )
        managers.command.compute(state)
        obs = managers.observation.compute(state)
        obs_dim = obs.shape[0]
        reward = managers.reward.compute(state)
        result = managers.termination.compute(state)
        total_reward += reward
        for label, value in managers.reward.term_values.items():
            reward_totals[label] = reward_totals.get(label, 0.0) + value
        if result.done:
            print(f"episode ended at step {step}: {result.terms}")
            break

    out_png = "/tmp/sim_managers_g1.png"
    render = sim.render(width=640, height=480)
    png_bytes = _extract_png_bytes(render)
    if png_bytes is not None:
        Path(out_png).write_bytes(png_bytes)
        print(f"saved render still -> {out_png}")

    print(f"\nrobot={ROBOT}  control_dt={dt:.4f}s  observation_dim={obs_dim}")
    print(f"command sample (vx, vy, wz) = {np.round(state.command('base_velocity'), 3)}")
    print(f"\ntotal reward over rollout: {total_reward:.4f}")
    print("per-term reward contribution (summed):")
    for label, value in sorted(reward_totals.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {label:<18} {value:+.5f}")

    sim.destroy()


if __name__ == "__main__":
    main()
