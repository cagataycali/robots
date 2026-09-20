# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MuJoCo-Warp solver gets a contact budget the initial pose cannot undersize.

Newton sizes ``nconmax`` / ``njmax`` by estimating from the state the world was
built in, and takes the larger of that estimate and any value passed in. A
rollout reaches poses the initial one did not: an arm that starts in the air and
later rests on the table produces more contacts than the estimate covers, and
the overflow is not a degraded frame -- MuJoCo-Warp prints ``broadphase overflow
- please increase nconmax to N`` every step and writes past the buffer, killing
the process with a raw ``Warp CUDA error 700: an illegal memory access`` out of
the constraint solver.

The pure budget rule runs everywhere; the end-to-end pin needs newton, warp and
a CUDA device and is skipped without them (pre-fix it does not fail, it aborts
the interpreter).
"""

from __future__ import annotations

import importlib.util

import pytest

from strands_robots.simulation.newton.backend import (
    CONTACT_BUDGET_FLOOR,
    CONTACTS_PER_SHAPE,
    ensure_newton,
    solver_contact_budget,
)


class _BudgetedSolver:
    """Stand-in for a solver whose constructor takes the contact keywords."""

    def __init__(self, model: object, njmax: int | None = None, nconmax: int | None = None) -> None:
        self.njmax, self.nconmax = njmax, nconmax


class _PlainSolver:
    """Stand-in for a solver whose constructor has no contact keywords."""

    def __init__(self, model: object) -> None:
        self.model = model


class TestSolverContactBudget:
    @pytest.mark.parametrize(
        ("solver_cls", "shape_count", "expected"),
        [
            # A solver without the keywords gets none -- it would reject them.
            (_PlainSolver, 34, {}),
            # A small scene still gets the pose-independent floor...
            (_BudgetedSolver, 1, {"nconmax": CONTACT_BUDGET_FLOOR, "njmax": CONTACT_BUDGET_FLOOR}),
            (_BudgetedSolver, 34, {"nconmax": CONTACT_BUDGET_FLOOR, "njmax": CONTACT_BUDGET_FLOOR}),
            # ... and a scene with enough shapes to exceed it scales past it.
            (_BudgetedSolver, 200, {"nconmax": CONTACTS_PER_SHAPE * 200, "njmax": CONTACTS_PER_SHAPE * 200}),
        ],
    )
    def test_budget_per_solver_and_scene(self, solver_cls, shape_count, expected) -> None:
        assert solver_contact_budget(solver_cls, shape_count) == expected

    def test_budget_is_never_negative(self) -> None:
        # shape_count is read off a finalized model; a nonsense value must not
        # turn into a negative buffer request the solver would reject.
        assert solver_contact_budget(_BudgetedSolver, -5) == {
            "nconmax": CONTACT_BUDGET_FLOOR,
            "njmax": CONTACT_BUDGET_FLOOR,
        }


_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None


def _has_cuda() -> bool:
    if not _HAS_NEWTON:
        return False
    _, wp = ensure_newton()
    return bool(wp.get_cuda_device_count())


@pytest.mark.skipif(not _HAS_NEWTON or not _has_cuda(), reason="newton/warp with a CUDA device required")
def test_a_rendered_rollout_keeps_the_budget_the_poses_it_reaches_need() -> None:
    """The scene an image-driven policy rollout builds, and the loop it runs.

    The buffers the solver was actually built with are the observable: sized
    from the pose the world was built in they came out at 48 contacts / 64
    constraints for this scene, while stepping it asked for 63 -- and the
    overflow aborted the process on the first ``send_action`` after a render,
    so the arm never moved and no envelope named a cause.
    """
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    sim = NewtonSimEngine(solver="mujoco")
    try:
        sim.create_world(ground_plane=True)
        sim.add_robot("so101")
        sim.add_object("cube", position=[0.02, -0.34, 0.0125], shape="box", size=[0.025] * 3, mass=0.02)
        sim.add_camera("front", position=[0.32, -0.55, 0.30], target=[0.02, -0.34, 0.08], width=256, height=256)
        data = sim._solver.mjw_data
        assert data.naconmax >= CONTACT_BUDGET_FLOOR, f"contact buffer sized from the initial pose: {data.naconmax}"
        assert data.njmax >= CONTACT_BUDGET_FLOOR, f"constraint buffer sized from the initial pose: {data.njmax}"

        sim.step(200)
        for _ in range(6):
            assert sim.render(camera_name="front")["status"] == "success"
            applied = sim.send_action({str(i): 0.2 for i in range(1, 7)}, robot_name="so101", n_substeps=25)
            assert applied["status"] == "success", applied["content"][0]["text"]
    finally:
        sim.destroy()
