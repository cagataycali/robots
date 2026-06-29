"""Whole-body tracking (WBT) through the config-driven sim_managers framework.

Builds a WBT env recipe - a reference *motion clip* command plus tracking
observation / reward / termination terms - and drives a position-controlled
SO-101 arm in headless MuJoCo to imitate the clip. Each control step feeds the
simulator's joint state into the managers: the ``motion_clip`` command publishes
the per-step target pose, the observation manager assembles the policy input,
the reward manager scores how well the arm matches the reference, and the
termination manager watches for divergence.

The observation / reward / termination half of the recipe is loaded from
``sim_managers_wbt.yaml``; the command block (the clip frames) is generated here
because the frames are large arrays. The same declarative terms a WBT trainer
consumes therefore run end to end on real simulator data, and the arm visibly
performs the reference motion (saved to an MP4 + still).

Run (headless):

    MUJOCO_GL=egl python examples/sim_managers_wbt.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from strands_robots import create_simulation
from strands_robots.sim_managers import EnvState, build_managers, load_managers_config
from strands_robots.sim_managers.motion import MOTION_TARGET_POS

CONFIG_PATH = Path(__file__).with_suffix(".yaml")
ROBOT = "so101"
N_STEPS = 240
SUBSTEPS = 5
VIDEO_PATH = "/tmp/sim_managers_wbt.mp4"
STILL_PATH = "/tmp/sim_managers_wbt.png"


def _make_motion_clip(n_joints: int, *, num_frames: int = 60, fps: float = 30.0) -> dict[str, object]:
    """Build a smooth sinusoidal reference clip as a command_manager config block.

    Each joint follows a 0.6 rad sine wave phase-offset from its neighbours, with
    the analytic derivative supplied as the velocity target.
    """
    t = np.linspace(0.0, 2 * np.pi, num_frames, endpoint=False)
    amp = 0.6
    frames_pos = np.stack([amp * np.sin(t + 0.6 * j) for j in range(n_joints)], axis=1)
    frames_vel = np.stack(
        [amp * np.cos(t + 0.6 * j) * (2 * np.pi / (num_frames / fps)) for j in range(n_joints)], axis=1
    )
    return {
        "terms": [
            {
                "name": "motion",
                "func": "motion_clip",
                "params": {"frames_pos": frames_pos.tolist(), "frames_vel": frames_vel.tolist(), "fps": fps},
            }
        ]
    }


def _extract_image(render: object) -> np.ndarray | None:
    """Decode the RGB image out of a ``render()`` agent-tool content payload."""
    if not isinstance(render, dict):
        return None
    import imageio.v3 as iio

    for block in render.get("content", []) or []:
        image = block.get("image") if isinstance(block, dict) else None
        if isinstance(image, dict):
            source = image.get("source", {})
            data = source.get("bytes") if isinstance(source, dict) else None
            if isinstance(data, bytes):
                return np.asarray(iio.imread(data))
    return None


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")

    sim = create_simulation(backend="mujoco")
    sim.create_world()
    if sim.add_robot(ROBOT).get("status") != "success":
        raise RuntimeError(f"failed to add robot {ROBOT!r}")
    sim.reset()

    joint_names = sim.robot_joint_names(ROBOT)
    n_joints = len(joint_names)

    config = load_managers_config(CONFIG_PATH)  # validates the YAML half
    assert config.observation and config.reward and config.termination
    # Merge the generated command block with the YAML recipe and rebuild.
    import yaml

    recipe = yaml.safe_load(CONFIG_PATH.read_text())
    recipe["command_manager"] = _make_motion_clip(n_joints)
    managers = build_managers(recipe)
    assert managers.command and managers.observation and managers.reward and managers.termination

    model = sim._world._model
    dt = float(model.opt.timestep) * SUBSTEPS

    managers.command.reset()
    managers.reward.reset()

    last_action = np.zeros(n_joints)
    frames: list[np.ndarray] = []
    reward_totals: dict[str, float] = {}
    total_reward = 0.0
    obs_dim = 0
    ended_at = None

    for step in range(N_STEPS):
        obs = sim.get_observation(ROBOT, skip_images=True)
        joint_pos = np.array([float(obs[name]) for name in joint_names])
        joint_vel = np.array([float(obs[f"{name}.vel"]) for name in joint_names])

        state = EnvState(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            action=last_action,
            last_action=last_action,
            dt=dt,
            step_count=step,
            max_episode_length=N_STEPS,
        )
        managers.command.compute(state)
        target = np.asarray(state.extras[MOTION_TARGET_POS], dtype=np.float64)
        state.action = target

        obs_vec = managers.observation.compute(state)
        obs_dim = obs_vec.shape[0]
        total_reward += managers.reward.compute(state)
        for label, value in managers.reward.term_values.items():
            reward_totals[label] = reward_totals.get(label, 0.0) + value
        result = managers.termination.compute(state)

        sim.send_action(target.tolist(), ROBOT)
        sim.step(SUBSTEPS)
        last_action = target

        image = _extract_image(sim.render(width=640, height=480))
        if image is not None:
            frames.append(image)

        if result.done:
            ended_at = (step, result.terms)
            break

    if frames:
        import imageio.v3 as iio

        iio.imwrite(VIDEO_PATH, np.stack(frames), fps=int(round(1.0 / dt)), codec="libx264")
        iio.imwrite(STILL_PATH, frames[-1])
        print(f"saved rollout video -> {VIDEO_PATH} ({len(frames)} frames)")
        print(f"saved final still  -> {STILL_PATH}")

    print(f"\nrobot={ROBOT}  control_dt={dt:.4f}s  observation_dim={obs_dim}  joints={n_joints}")
    if ended_at is not None:
        print(f"episode ended early at step {ended_at[0]}: {ended_at[1]}")
    else:
        print(f"completed full {N_STEPS}-step rollout with no divergence")
    print(f"\ntotal reward over rollout: {total_reward:.4f}")
    print("per-term reward contribution (summed):")
    for label, value in sorted(reward_totals.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {label:<14} {value:+.5f}")

    sim.destroy()


if __name__ == "__main__":
    main()
