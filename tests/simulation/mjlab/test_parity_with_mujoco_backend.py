"""N=1 parity of the mjlab backend with the MuJoCo backend.

Both engines load the same MJCF and receive the same ctrl trajectory; the
joint trajectories must agree. mjlab's physics is MuJoCo-Warp, a port of the
same integrator, so the tolerance is tight for a fixed-base arm (so101) and
loose but bounded for a 29-DoF floating-base humanoid falling onto the plane
(Unitree G1: contacts are the one place a GPU solver may order things
differently).

Runs only where mjlab + a CUDA device are present (Thor, the lane box);
everywhere else it skips. Logs: ~/.tiny/mjlab-20260928/logs/parity_*.log.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_HAS_MJLAB = importlib.util.find_spec("mjlab") is not None and importlib.util.find_spec("mujoco_warp") is not None


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # pragma: no cover
        return False


pytestmark = [
    pytest.mark.skipif(not _HAS_MJLAB, reason="mjlab not installed (pip install 'strands-robots[sim-mjlab]')"),
    pytest.mark.skipif(not _cuda(), reason="mjlab backend parity needs a CUDA device"),
]


def _rollout(engine, robot: str, ctrl_fn, n_control: int, substeps: int) -> np.ndarray:
    keys = engine.robot_action_keys(robot)
    joints = [j for j in engine.robot_joint_names(robot) if not j.startswith("floating")]
    traj = []
    for k in range(n_control):
        ctrl = ctrl_fn(k, len(keys))
        res = engine.send_action(dict(zip(keys, ctrl, strict=True)), robot, n_substeps=substeps)
        assert res["status"] == "success", res
        obs = engine.get_observation(robot, skip_images=True)
        row = [obs[j] for j in joints]
        if "base_pos" in obs:
            row += list(obs["base_pos"]) + list(obs["base_quat"])
        traj.append(row)
    return np.asarray(traj, dtype=np.float64)


def _pair(robot: str):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from strands_robots.simulation import create_simulation

    classic = create_simulation("mujoco")
    classic.create_world(timestep=0.002)
    assert classic.add_robot(robot)["status"] == "success"
    mjl = create_simulation("mjlab", num_envs=1)
    mjl.create_world(timestep=0.002)
    assert mjl.add_robot(robot)["status"] == "success"
    return classic, mjl


def test_factory_resolves_mjlab_aliases():
    from strands_robots.simulation.factory import _resolve_name as resolve_backend_name
    from strands_robots.simulation.factory import list_backends

    assert "mjlab" in list_backends()
    assert resolve_backend_name("mjl") == "mjlab"
    assert resolve_backend_name("mujoco_warp") == "mjlab"


def test_so101_contract_matches_mujoco_backend():
    classic, mjl = _pair("so101")
    try:
        assert mjl.robot_joint_names("so101") == classic.robot_joint_names("so101")
        assert mjl.robot_action_keys("so101") == classic.robot_action_keys("so101")
        oc = classic.get_observation("so101", skip_images=True)
        om = mjl.get_observation("so101", skip_images=True)
        assert set(om) == set(oc), (sorted(om), sorted(oc))
        assert mjl.physics_timestep() == pytest.approx(0.002)
        assert mjl.mj_model.opt.integrator == classic.mj_model.opt.integrator
        assert mjl.mj_model.opt.solver == classic.mj_model.opt.solver
    finally:
        classic.destroy()
        mjl.destroy()


def test_so101_trajectory_parity_sinusoid():
    classic, mjl = _pair("so101")

    def ctrl(k: int, n: int):
        t = k * 0.02
        return [0.4 * np.sin(1.5 * t + 0.3 * i) for i in range(n)]

    try:
        a = _rollout(classic, "so101", ctrl, n_control=100, substeps=10)
        b = _rollout(mjl, "so101", ctrl, n_control=100, substeps=10)
    finally:
        classic.destroy()
        mjl.destroy()
    err = np.abs(a - b)
    _log("so101", a, b)
    assert np.isfinite(b).all()
    # 2 s of motion, 6 joints: sub-milliradian agreement expected.
    assert err.max() < 5e-4, f"max joint error {err.max():.5f} rad"


def test_unitree_g1_free_base_parity_settle():
    classic, mjl = _pair("unitree_g1")
    try:
        assert mjl.robot_joint_names("unitree_g1") == classic.robot_joint_names("unitree_g1")
        oc = classic.get_observation("unitree_g1", skip_images=True)
        om = mjl.get_observation("unitree_g1", skip_images=True)
        assert set(om) == set(oc)
        # Same spawn: keyframe root height and keyframe joint pose.
        assert np.allclose(om["base_pos"], oc["base_pos"], atol=1e-6), (om["base_pos"], oc["base_pos"])
        for j in classic.robot_action_keys("unitree_g1"):
            assert om[j] == pytest.approx(oc[j], abs=1e-6), j

        def hold(k: int, n: int):
            return [0.0] * n

        a = _rollout(classic, "unitree_g1", hold, n_control=50, substeps=10)
        b = _rollout(mjl, "unitree_g1", hold, n_control=50, substeps=10)
    finally:
        classic.destroy()
        mjl.destroy()
    _log("unitree_g1", a, b)
    assert np.isfinite(b).all()
    nj = 29
    joint_err = np.abs(a[:, :nj] - b[:, :nj]).max()
    base_err = np.abs(a[:, nj : nj + 3] - b[:, nj : nj + 3]).max()
    # 1 s of the humanoid collapsing onto the plane under zero ctrl: contact
    # ordering differs between CPU and Warp solvers, so bound rather than match.
    assert joint_err < 0.01, f"max joint error {joint_err:.4f} rad"
    assert base_err < 0.005, f"max base position error {base_err:.4f} m"


def test_num_envs_batch_shapes_and_world_zero_is_the_contract():
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch

    from strands_robots.simulation import create_simulation

    e = create_simulation("mjlab", num_envs=8)
    e.create_world(timestep=0.002)
    assert e.add_robot("so101")["status"] == "success"
    try:
        batch = e.get_observation_batch("so101")
        assert batch["1"].shape == (8,)
        block = torch.zeros(8, 6)
        block[:, 0] = torch.linspace(-0.5, 0.5, 8)
        assert e.send_action_batch(block, "so101", n_substeps=200)["status"] == "success"
        pan = e.get_observation_batch("so101")["1"].cpu().numpy()
        assert np.all(np.diff(pan) > 0), pan  # eight worlds, eight different targets
        assert e.get_observation("so101", skip_images=True)["1"] == pytest.approx(float(pan[0]))
        img = e.render()
        assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    finally:
        e.destroy()


def _log(robot: str, a: np.ndarray, b: np.ndarray) -> None:
    d = os.path.expanduser("~/.tiny/mjlab-20260928/logs")
    if not os.path.isdir(d):
        return
    with open(os.path.join(d, f"parity_{robot}.log"), "w") as fh:
        err = np.abs(a - b)
        fh.write(
            f"{robot}: steps={len(a)} cols={a.shape[1]} max_abs_err={err.max():.6f} mean_abs_err={err.mean():.6f}\n"
        )
        fh.write("per-col max err: " + " ".join(f"{x:.5f}" for x in err.max(axis=0)) + "\n")
        fh.write("classic last: " + " ".join(f"{x:.4f}" for x in a[-1]) + "\n")
        fh.write("mjlab   last: " + " ".join(f"{x:.4f}" for x in b[-1]) + "\n")
