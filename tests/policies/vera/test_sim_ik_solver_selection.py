"""Dependency-free contract tests for the VERA eef-delta -> MuJoCo IK bridge.

The real-solver accuracy regression in ``test_vera_sim_ik_solver.py``
``importorskip``s on ``mink`` + ``qpsolvers`` (the ``sim-mujoco`` extra), so the
whole module is skipped wherever that stack is absent - which is the default CI
image. That leaves the bridge's *dependency-free* surface untested even though
it is the part most likely to break on a clean install:

* :func:`~strands_robots.policies.vera.sim_ik._resolve_qp_solver` - the QP
  backend auto-selection that lets the bridge run on ``daqp`` or ``quadprog``
  (or whatever ``qpsolvers`` reports), and the actionable errors it raises when
  a requested backend is missing or none is installed.
* :func:`~strands_robots.policies.vera.sim_ik._install_hint` - the actionable
  message naming the extra and what it pulls.
* :class:`~strands_robots.policies.vera.sim_ik.MinkIKBridge` construction
  failing with the install hint when ``mink`` is not importable.
* :func:`~strands_robots.policies.vera.sim_ik.delta_to_matrix` rotation-encoding
  dispatch (rot6d / axis-angle) and its rejection of unsupported dims.
* :func:`~strands_robots.policies.vera.sim_ik.decode_vera_delta_chunk_to_targets`
  input validation (action-chunk rank, pose-dim sufficiency) that runs *before*
  any sim dependency is touched.

These paths need no ``mink``/``mujoco``/``qpsolvers`` actually installed - the
solver selection is driven with a stub ``qpsolvers`` module - so they execute in
plain CI and guard the contracts a clean-install user hits first.
"""

import sys
import types

import numpy as np
import pytest

from strands_robots.policies.vera import sim_ik


@pytest.fixture
def fake_qpsolvers(monkeypatch):
    """Install a stub ``qpsolvers`` module with a settable solver list.

    Returns the stub so a test can mutate ``available_solvers`` to drive each
    branch of :func:`_resolve_qp_solver` without depending on which QP backends
    happen to be installed on the host.
    """
    stub = types.ModuleType("qpsolvers")
    stub.available_solvers = ["quadprog", "osqp"]
    monkeypatch.setitem(sys.modules, "qpsolvers", stub)
    return stub


class TestResolveQpSolver:
    """The QP-backend auto-selection contract (no real qpsolvers needed)."""

    def test_prefers_daqp_when_available(self, fake_qpsolvers):
        fake_qpsolvers.available_solvers = ["quadprog", "daqp", "osqp"]
        assert sim_ik._resolve_qp_solver(None) == "daqp"

    def test_prefers_quadprog_when_daqp_absent(self, fake_qpsolvers):
        fake_qpsolvers.available_solvers = ["osqp", "quadprog"]
        assert sim_ik._resolve_qp_solver(None) == "quadprog"

    def test_falls_back_to_first_available_when_none_preferred(self, fake_qpsolvers):
        # No name from the preferred list is installed -> first reported wins.
        fake_qpsolvers.available_solvers = ["gurobi", "highs"]
        assert sim_ik._resolve_qp_solver(None) == "gurobi"

    def test_honours_explicit_requested_solver(self, fake_qpsolvers):
        fake_qpsolvers.available_solvers = ["quadprog", "osqp"]
        assert sim_ik._resolve_qp_solver("osqp") == "osqp"

    def test_requested_solver_not_installed_raises_valueerror(self, fake_qpsolvers):
        fake_qpsolvers.available_solvers = ["quadprog"]
        with pytest.raises(ValueError, match="daqp"):
            sim_ik._resolve_qp_solver("daqp")

    def test_no_backend_installed_raises_runtimeerror(self, fake_qpsolvers):
        fake_qpsolvers.available_solvers = []
        with pytest.raises(RuntimeError, match="sim-mujoco"):
            sim_ik._resolve_qp_solver(None)

    def test_qpsolvers_absent_raises_importerror_with_hint(self, monkeypatch):
        # Force the ``import qpsolvers`` inside _resolve_qp_solver to fail.
        monkeypatch.setitem(sys.modules, "qpsolvers", None)
        with pytest.raises(ImportError, match="sim-mujoco"):
            sim_ik._resolve_qp_solver(None)


class TestInstallHint:
    """The actionable install message must name the extra and what it pulls."""

    def test_names_extra_and_dependencies(self):
        hint = sim_ik._install_hint()
        assert "sim-mujoco" in hint
        assert "mink" in hint


class TestMinkIKBridgeImportGuard:
    """Constructing the bridge without ``mink`` fails with the install hint."""

    def test_missing_mink_raises_importerror_with_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "mink", None)
        with pytest.raises(ImportError, match="sim-mujoco"):
            sim_ik.MinkIKBridge(model=object(), ee_frame_name="hand")


class TestDeltaToMatrix:
    """Rotation-delta encoding dispatch is selected by ``rotation_dim``."""

    def test_rot6d_dim_six_dispatches_to_gram_schmidt(self):
        # Canonical 6D rep (first two columns of identity) -> identity matrix.
        R = sim_ik.delta_to_matrix(np.array([1, 0, 0, 0, 1, 0], dtype=np.float64), 6)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)

    def test_axis_angle_dim_three_dispatches_to_rodrigues(self):
        # 90 deg about z.
        R = sim_ik.delta_to_matrix(np.array([0, 0, np.pi / 2], dtype=np.float64), 3)
        np.testing.assert_allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6)

    def test_unsupported_rotation_dim_raises(self):
        with pytest.raises(ValueError, match="unsupported rotation_dim"):
            sim_ik.delta_to_matrix(np.zeros(4), 4)


class TestDecodeInputValidation:
    """``decode_vera_delta_chunk_to_targets`` validates before touching the sim.

    Both guards fire before ``ik_bridge`` is ever referenced, so a sentinel
    object that would explode if called proves the validation is up front.
    """

    def test_non_2d_action_chunk_raises(self):
        with pytest.raises(ValueError, match=r"\[T, D\]"):
            sim_ik.decode_vera_delta_chunk_to_targets(
                np.zeros((2, 3, 7), dtype=np.float64),
                ik_bridge=object(),
                q_init=np.zeros(7),
            )

    def test_short_pose_block_raises(self):
        # rotation_dim=6 needs >= 9 pose dims; a 4-wide chunk (3 pose + gripper)
        # is short after the gripper column is removed.
        with pytest.raises(ValueError, match="pose dims"):
            sim_ik.decode_vera_delta_chunk_to_targets(
                np.zeros((2, 4), dtype=np.float64),
                ik_bridge=object(),
                q_init=np.zeros(7),
                rotation_dim=6,
                has_gripper=True,
            )
