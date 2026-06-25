"""Real-solver coverage for the VERA eef-delta IK bridge (``sim_ik``).

The other VERA IK tests deliberately use a ``FakeBridge`` + fake ``mujoco`` so
they run in the light base env. That leaves the actual closed-loop kinematics
untested: ``MinkIKBridge`` (mink + mujoco construction, forward kinematics, and
the differential-IK solve loop), the ``qpsolvers`` backend auto-selection, and
the chunk decoder driving a *real* solver. These tests build a minimal 3-DoF
planar arm with ``mujoco`` and exercise that real path end to end, so a
regression in the solver math, frame-task wiring, or solver-resolution contract
is caught locally instead of only on hardware.
"""

import sys

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


def test_resolve_qp_solver_auto_and_explicit():
    """Auto-selection returns an installed backend; an explicitly requested,
    installed backend is honoured verbatim."""
    auto = sim_ik._resolve_qp_solver(None)
    from qpsolvers import available_solvers

    assert auto in available_solvers
    # daqp is pinned by mink and present in the sim extra.
    assert sim_ik._resolve_qp_solver("daqp") == "daqp"


def test_resolve_qp_solver_rejects_uninstalled_backend():
    """An explicit, not-installed backend fails loudly (no silent fallback)."""
    with pytest.raises(ValueError, match="not installed"):
        sim_ik._resolve_qp_solver("definitely-not-a-real-solver")


def test_resolve_qp_solver_raises_when_no_backend_available(monkeypatch):
    """When qpsolvers reports zero backends, the bridge refuses to proceed."""
    monkeypatch.setattr("qpsolvers.available_solvers", [])
    with pytest.raises(RuntimeError, match="No qpsolvers backend"):
        sim_ik._resolve_qp_solver(None)


def test_resolve_qp_solver_missing_qpsolvers_gives_install_hint(monkeypatch):
    """If qpsolvers itself is absent, the error carries the actionable hint."""
    monkeypatch.setitem(sys.modules, "qpsolvers", None)
    with pytest.raises(ImportError) as ei:
        sim_ik._resolve_qp_solver(None)
    assert "sim-mujoco" in str(ei.value)


def test_delta_to_matrix_dispatches_rot6d_and_rejects_bad_dim():
    """delta_to_matrix routes rotation_dim 6 -> rot6d and rejects other dims."""
    rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)
    assert np.allclose(sim_ik.delta_to_matrix(rot6d, 6), np.eye(3), atol=1e-6)
    with pytest.raises(ValueError, match="unsupported rotation_dim"):
        sim_ik.delta_to_matrix(np.zeros(4), 4)


def test_decode_rejects_non_2d_chunk_and_short_pose_block(bridge):
    """The decoder validates chunk shape and pose-dim sufficiency up front."""
    with pytest.raises(ValueError, match=r"\[T, D\]"):
        sim_ik.decode_vera_delta_chunk_to_targets(np.zeros((2, 3, 7)), bridge, np.zeros(bridge.model.nq))
    # rotation_dim=6 needs >= 9 pose dims; a 4-wide chunk (3 pose + gripper) is short.
    with pytest.raises(ValueError, match="pose dims"):
        sim_ik.decode_vera_delta_chunk_to_targets(
            np.zeros((2, 4), dtype=np.float64),
            bridge,
            np.zeros(bridge.model.nq),
            rotation_dim=6,
            has_gripper=True,
        )
