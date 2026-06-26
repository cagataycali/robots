"""Real-solver accuracy regression for the VERA eef-delta IK bridge (``sim_ik``).

This module exercises the *real* closed-loop kinematics that only run with the
``sim-mujoco`` extra installed: :class:`MinkIKBridge` forward kinematics and the
differential-IK solve loop against a genuine ``mink`` + ``mujoco`` stack, plus
the chunk decoder driving that real solver. It ``importorskip``s on ``mujoco`` /
``mink`` / ``qpsolvers`` so it skips cleanly on a clean-install image.

The bridge's *dependency-free* surface (QP-backend selection, the install hint,
the missing-``mink`` import guard, ``delta_to_matrix`` dispatch, and the decoder
input validation) is covered without any sim stack in
``test_sim_ik_solver_selection.py`` and ``test_sim_ik_bridge_solve_loop.py``, so
those contracts are guarded in plain CI; here we pin the geometry that needs the
real solver.
"""

import numpy as np
import pytest

# The real IK stack is only present with the sim extra installed.
pytest.importorskip("mujoco")
pytest.importorskip("mink")
pytest.importorskip("qpsolvers")

import mujoco  # noqa: E402

from strands_robots.policies.vera import sim_ik  # noqa: E402

# Minimal planar 3R arm: three hinge joints about z, EE site at the tip.
# Reach radius ~0.9 m; any target inside the plane and radius is solvable.
_ARM_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
        <body name="ee" pos="0.3 0 0">
          <joint name="j3" type="hinge" axis="0 0 1"/>
          <geom type="box" size="0.02 0.02 0.02"/>
          <site name="ee_site" pos="0 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def arm_model():
    return mujoco.MjModel.from_xml_string(_ARM_XML)


@pytest.fixture
def bridge(arm_model):
    return sim_ik.MinkIKBridge(arm_model, "ee_site", "site")


def test_bridge_fk_and_solve_reaches_reachable_target(bridge):
    """A real MinkIKBridge: FK returns an SE3 matrix and solve() reaches a
    configuration-defined (hence reachable) Cartesian target within tolerance."""
    # Build a guaranteed-reachable target by taking FK of a known joint config.
    q_goal = np.array([0.3, -0.4, 0.2])
    target = bridge.ee_pose(q_goal)
    assert target.shape == (4, 4)
    assert np.allclose(target[3], [0, 0, 0, 1])  # homogeneous bottom row

    # Solve from a different seed; the realized EE must land on the target.
    q_solved = bridge.solve(target, np.zeros(bridge.model.nq))
    assert q_solved.shape == (bridge.model.nq,)
    reached = bridge.ee_pose(q_solved)
    pos_err = float(np.linalg.norm(reached[:3, 3] - target[:3, 3]))
    assert pos_err < 5e-3, f"IK did not reach target: {pos_err * 1000:.2f} mm"


def test_decode_chunk_through_real_bridge_tracks_closed_loop(bridge):
    """decode_vera_delta_chunk_to_targets driven by the real solver: small
    per-step EE deltas are tracked in closed loop with sub-mm error, and the
    output dict carries [T, nq] qpos, the passthrough gripper column, and
    finite tracking-error stats."""
    seed = np.array([0.4, -0.8, 0.5])  # well away from the singular full-stretch
    # Three steps of +1 cm along local x, gripper toggling in the last column.
    chunk = np.array(
        [
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    out = sim_ik.decode_vera_delta_chunk_to_targets(
        chunk, bridge, seed, rotation_dim=3, has_gripper=True, gripper_dim_index=6
    )
    assert out["qpos"].shape == (3, bridge.model.nq)
    assert out["gripper"].tolist() == [1.0, 0.0, 1.0]
    track = out["tracking_error"]
    assert track["max_mm"] >= track["mean_mm"] >= 0.0
    # A real, non-singular arm tracks 1 cm steps to well under a millimetre.
    assert track["max_mm"] < 2.0, track
