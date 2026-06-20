"""Dependency-free contract tests for the Cosmos 3 -> MuJoCo IK bridge.

The accuracy regression in ``test_sim_ik.py`` ``importorskip``s on ``mink`` and
``mujoco`` (the ``cosmos3-sim`` extra), so the whole module is skipped wherever
that stack is absent - which is the default CI image. That leaves the bridge's
*dependency-free* surface untested even though it is the part most likely to
break on a clean install:

* :func:`~strands_robots.policies.cosmos3.sim_ik._resolve_qp_solver` - the QP
  backend auto-selection that lets the bridge run on ``daqp`` or ``quadprog``
  (or whatever ``qpsolvers`` reports), and the actionable errors it raises when
  a requested backend is missing or none is installed.
* :class:`~strands_robots.policies.cosmos3.sim_ik.MinkIKBridge` construction
  failing with the install hint when ``mink`` is not importable.
* :func:`~strands_robots.policies.cosmos3.sim_ik.decode_cosmos_chunk_to_targets`
  input validation (normalization method, action-chunk rank) that runs *before*
  any sim dependency is touched.

These paths need no ``mink``/``mujoco``/``qpsolvers`` actually installed - they
are driven with a stub ``qpsolvers`` module - so they execute in plain CI and
guard the contracts a clean-install user hits first.
"""

import sys
import types

import numpy as np
import pytest

from strands_robots.policies.cosmos3 import sim_ik


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
        with pytest.raises(RuntimeError, match="cosmos3-sim"):
            sim_ik._resolve_qp_solver(None)

    def test_qpsolvers_absent_raises_importerror_with_hint(self, monkeypatch):
        # Force the ``import qpsolvers`` inside _resolve_qp_solver to fail.
        monkeypatch.setitem(sys.modules, "qpsolvers", None)
        with pytest.raises(ImportError, match="cosmos3-sim"):
            sim_ik._resolve_qp_solver(None)


class TestInstallHint:
    """The actionable install message must name the extra and what it pulls."""

    def test_names_extra_and_dependencies(self):
        hint = sim_ik._install_hint()
        assert "cosmos3-sim" in hint
        assert "mink" in hint
        assert "mujoco" in hint


class TestMinkIKBridgeImportGuard:
    """Constructing the bridge without ``mink`` fails with the install hint."""

    def test_missing_mink_raises_importerror_with_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "mink", None)
        with pytest.raises(ImportError, match="cosmos3-sim"):
            sim_ik.MinkIKBridge(model=object(), ee_frame_name="hand")


class _QuantileEmbodiment:
    """Minimal stand-in for a quantile-normalized embodiment (no real model)."""

    normalization = "quantile"
    domain_name = "droid_lerobot"
    raw_action_layout = ["tx", "ty", "tz", "r0", "r1", "r2", "r3", "r4", "r5", "grasp"]


class TestDecodeInputValidation:
    """``decode_cosmos_chunk_to_targets`` validates before touching the sim."""

    def test_non_quantile_normalization_raises(self):
        emb = _QuantileEmbodiment()
        emb.normalization = "zscore"
        with pytest.raises(ValueError, match="quantile"):
            sim_ik.decode_cosmos_chunk_to_targets(
                np.zeros((4, 10), dtype=np.float32),
                emb,
                ik_bridge=None,
                q_init=np.zeros(7),
            )

    def test_non_2d_action_chunk_raises(self):
        # A quantile embodiment passes the first guard; the rank check fires
        # before any mink/qpsolvers dependency is referenced.
        with pytest.raises(ValueError, match=r"\[T, D\]"):
            sim_ik.decode_cosmos_chunk_to_targets(
                np.zeros(10, dtype=np.float32),
                _QuantileEmbodiment(),
                ik_bridge=None,
                q_init=np.zeros(7),
            )
