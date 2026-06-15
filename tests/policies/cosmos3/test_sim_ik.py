"""IK-accuracy regression for the Cosmos 3 -> MuJoCo sim-loop bridge.

These tests pin the *physical* half of issue #44 that the de-normalization +
inverse-kinematics layer closes: the in-process ``diffusers`` backend emits the
model's raw unified action in ``[-1, 1]`` quantile-normalized **relative
end-effector pose** space, which is meaningless fed straight into MuJoCo joint
actuators (the earlier repro: normalized values land arbitrarily inside/outside
real joint limits, MuJoCo silently clamps, the arm doesn't track). The honest
path is de-normalize -> decode to an absolute EE-pose trajectory -> solve IK to
joint targets (:mod:`strands_robots.policies.cosmos3.sim_ik`).

The bar is the tracking-error number verified on Thor against real
``nvidia/Cosmos3-Nano`` weights: a reachable EE trajectory must be tracked to
**mean <= 12 mm / max <= 45 mm** Cartesian error. The unit test reproduces it
with a synthetic-but-reachable trajectory (no GPU / no model load) so CI guards
the bridge geometry; the GPU integration stub
(``tests_integ/policies/cosmos3/``) exercises the same path off real Cosmos
output.

``mink`` + ``mujoco`` (the ``cosmos3-sim`` extra) and ``robot_descriptions``
(for the Franka/Panda model) are imported via ``importorskip`` so the module
skips cleanly when the sim stack is absent, mirroring the groot/moveit2 service
tests.
"""

import numpy as np
import pytest

# Optional sim stack: skip cleanly when the cosmos3-sim extra is not installed.
mujoco = pytest.importorskip("mujoco", reason="mujoco not installed - pip install 'strands-robots[cosmos3-sim]'")
pytest.importorskip("mink", reason="mink not installed - pip install 'strands-robots[cosmos3-sim]'")
panda_mj_description = pytest.importorskip(
    "robot_descriptions.panda_mj_description",
    reason="robot_descriptions not installed",
)

# E402: importorskip must run before these imports so the module skips cleanly.
from strands_robots.policies.cosmos3.embodiments import get_embodiment  # noqa: E402
from strands_robots.policies.cosmos3.sim_ik import (  # noqa: E402
    MinkIKBridge,
    decode_cosmos_chunk_to_targets,
)

# Tracking-error acceptance bar (verified on Thor; reachable trajectories).
_MEAN_MM_BAR = 12.0
_MAX_MM_BAR = 45.0

# A natural, well-conditioned Panda configuration (elbow bent, wrist down) so
# the EE starts mid-workspace and synthetic targets stay reachable.
_Q_HOME = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79], dtype=np.float64)


@pytest.fixture(scope="module")
def panda_model():
    return mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)


@pytest.fixture
def q_init(panda_model):
    q = np.zeros(panda_model.nq, dtype=np.float64)
    q[:7] = _Q_HOME
    return q


@pytest.fixture
def bridge(panda_model):
    # The Franka/Panda end-effector frame is the "hand" body.
    return MinkIKBridge(panda_model, ee_frame_name="hand", ee_frame_type="body")


def test_default_solver_is_installed_with_the_extra(bridge):
    """Regression: MinkIKBridge's default solver must ship with cosmos3-sim.

    ``mink`` declares ``qpsolvers[daqp]`` as its dependency, so ``daqp`` is the
    only QP backend guaranteed by ``pip install strands-robots[cosmos3-sim]``.
    A default of ``"quadprog"`` (not pulled by the extra) made every IK solve
    raise ``qpsolvers.SolverNotFound`` on a clean install. Pin the default to a
    solver the extra actually provides.
    """
    import qpsolvers

    assert bridge.solver in qpsolvers.available_solvers, (
        f"MinkIKBridge default solver {bridge.solver!r} is not installed by the "
        f"cosmos3-sim extra (available: {qpsolvers.available_solvers})"
    )


def test_ee_pose_is_homogeneous_and_stable(bridge, q_init):
    pose = bridge.ee_pose(q_init)
    assert pose.shape == (4, 4)
    # Bottom row is [0, 0, 0, 1] and rotation block is orthonormal (a real pose).
    np.testing.assert_allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
    rot = pose[:3, :3]
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-5)


def test_solve_recovers_seed_for_zero_motion(bridge, q_init):
    """A target equal to the current EE pose must keep the arm essentially put."""
    start = bridge.ee_pose(q_init)
    q = bridge.solve(start, q_init)
    achieved = bridge.ee_pose(q)
    np.testing.assert_allclose(achieved[:3, 3], start[:3, 3], atol=1e-3)


def test_reachable_trajectory_tracks_within_bar(bridge, q_init):
    """A reachable Cartesian path is tracked to mean<=12mm / max<=45mm.

    This is the regression for the issue-#44 physics gap: with the IK bridge the
    arm actually tracks the EE trajectory (sub-cm), versus the pre-fix path that
    fed normalized [-1,1] columns straight into joint actuators.
    """
    start = bridge.ee_pose(q_init)
    # 32-step straight-ish line, ~17 cm total - inside the Franka workspace.
    steps = np.arange(1, 33)[:, None] * np.array([0.004, 0.003, -0.002])
    poses = np.repeat(start[None], 32, axis=0).copy()
    poses[:, :3, 3] = start[:3, 3] + steps
    qtraj = bridge.solve_trajectory(poses, q_init)
    assert qtraj.shape == (32, bridge.model.nq)
    err = bridge.tracking_error(poses, qtraj)
    assert err["mean_mm"] <= _MEAN_MM_BAR, err
    assert err["max_mm"] <= _MAX_MM_BAR, err


def test_solve_trajectory_warmstarts_continuously(bridge, q_init):
    """Warm-started joint trajectory is continuous (no IK-branch flips)."""
    start = bridge.ee_pose(q_init)
    steps = np.arange(1, 17)[:, None] * np.array([0.005, 0.0, 0.0])
    poses = np.repeat(start[None], 16, axis=0).copy()
    poses[:, :3, 3] = start[:3, 3] + steps
    qtraj = bridge.solve_trajectory(poses, q_init)
    # Per-step joint deltas stay small (smooth) - no large jumps between steps.
    jumps = np.linalg.norm(np.diff(qtraj[:, :7], axis=0), axis=1)
    assert float(jumps.max()) < 0.5, f"discontinuous IK trajectory: max jump {jumps.max()}"


def test_decode_cosmos_chunk_to_targets_closes_the_loop(bridge, q_init):
    """End-to-end: raw normalized Cosmos chunk -> de-norm -> EE poses -> joints.

    Reproduces the full bridge from a synthetic raw action chunk (the shape and
    [-1,1] range the diffusers backend emits) and asserts the contract: per-step
    joint targets, a gripper column, absolute poses, and a tracking error inside
    the bar.
    """
    emb = get_embodiment("droid")
    rng = np.random.default_rng(0)
    chunk = rng.uniform(-0.3, 0.3, (16, emb.raw_action_dim)).astype(np.float32)
    # Keep the rot6d block near identity so the synthetic deltas stay reachable
    # (random rotations would wander outside the arm workspace - a scaling
    # concern, not an IK one, as noted on the issue).
    chunk[:, 3:9] = np.tile([1, 0, 0, 0, 1, 0], (16, 1))

    out = decode_cosmos_chunk_to_targets(chunk, emb, bridge, q_init)

    assert out["qpos"].shape == (16, bridge.model.nq)
    assert out["poses"].shape == (16, 4, 4)
    # DROID raw layout ends in "grasp" -> gripper column is split off.
    assert out["gripper"] is not None
    assert out["gripper"].shape == (16,)
    assert out["tracking_error"]["mean_mm"] <= _MEAN_MM_BAR
    assert out["tracking_error"]["max_mm"] <= _MAX_MM_BAR


def test_decode_rejects_non_quantile_normalization(bridge, q_init):
    """A non-quantile embodiment is rejected (only quantile stats are bundled)."""
    import dataclasses

    emb = dataclasses.replace(get_embodiment("droid"), normalization="zscore")
    chunk = np.zeros((4, emb.raw_action_dim), dtype=np.float32)
    chunk[:, 3:9] = np.tile([1, 0, 0, 0, 1, 0], (4, 1))
    with pytest.raises(ValueError, match="normalization='quantile'"):
        decode_cosmos_chunk_to_targets(chunk, emb, bridge, q_init)


def test_solver_autoselects_installed_backend(panda_model):
    """The bridge runs with whatever qpsolvers backend is installed.

    Regression: the default used to hard-code ``solver="daqp"``, which raised
    ``SolverNotFound`` in environments shipping only ``quadprog``. ``None`` now
    auto-selects from ``qpsolvers.available_solvers`` (daqp preferred).
    """
    from qpsolvers import available_solvers

    bridge = MinkIKBridge(panda_model, ee_frame_name="hand", ee_frame_type="body")
    assert bridge.solver in available_solvers


def test_explicit_unknown_solver_raises_actionable_error(panda_model):
    with pytest.raises(ValueError, match="is not installed"):
        MinkIKBridge(panda_model, ee_frame_name="hand", ee_frame_type="body", solver="not_a_solver")
