"""The pour example says its zero score is the scene's doing, not the policy's.

``examples/17_pour_task.py`` authors a benchmark whose success clause needs
``cap_slide`` past 0.06 m, and mounts the carton at x = 0.55 where no
configuration of the ``so100`` it names can touch the cap: the arm's tool point
reaches x = 0.44 and even the bounding sphere of its outermost geom stops 54 mm
short of the plate's near face at x = 0.50. The example used to attribute the 0
to the policy ("the ``mock`` policy, which does not act") and close by inviting
the reader to "point --policy at a trained provider" - a task no provider can
win, on a robot that does move (every joint sweeps 0.6-1.0 rad under ``mock``).

The cells bind the prose to the geometry it now quotes, so a future scene that
moves the station into the workspace has to change the words with it rather
than keep passing, and pin the one route that does open the cap - the scripted
``set_joint_positions`` teaser - so a regression there fails here instead of
shipping a pour demo that does not pour.
"""

from __future__ import annotations

import importlib.util
import itertools
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "17_pour_task.py"


def _example():
    # A leading digit makes the module unimportable by name; load it by path.
    spec = importlib.util.spec_from_file_location("pour_task", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _prose() -> str:
    """The example's source with its line wrapping collapsed."""
    return " ".join(_EXAMPLE.read_text(encoding="utf-8").split())


def _poured_scene():
    """The example's own scene, on the robot its spec names."""
    from strands_robots import Robot

    module = _example()
    sim = Robot(module.BENCHMARK_SPEC["default_robot"], mesh=False)
    module.build_scene(sim)
    return module, sim


def _reach() -> tuple[float, float]:
    """Measured ``(outermost arm geom x, cap plate near face x)`` in metres.

    The bound is the outer edge of every arm geom's bounding sphere over a
    deterministic grid of the arm's positioning joints, so it over-states the
    reach rather than sampling its way past the answer.
    """
    import mujoco

    _, sim = _poured_scene()
    model, data = sim.mj_model, sim.mj_data
    joints = [
        j
        for j in range(model.njnt)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("so100/")
    ]
    address = [model.jnt_qposadr[j] for j in joints[:4]]
    grid = [np.linspace(float(model.jnt_range[j][0]), float(model.jnt_range[j][1]), 9) for j in joints[:4]]
    geoms = [
        g
        for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("so100/")
    ]
    farthest = -np.inf
    for pose in itertools.product(*grid):
        for slot, value in zip(address, pose, strict=True):
            data.qpos[slot] = value
        mujoco.mj_kinematics(model, data)
        farthest = max(farthest, max(data.geom_xpos[g][0] + model.geom_rbound[g] for g in geoms))
    cap = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "carton/cap_geom")
    assert cap >= 0, "the scene lost the carton's cap plate"
    return float(farthest), float(data.geom_xpos[cap][0] - model.geom_size[cap][0])


def test_no_arm_configuration_reaches_the_cap_the_success_clause_names() -> None:
    """The premise of every cell below, measured on the shipped assets."""
    arm_x, plate_x = _reach()
    assert arm_x < plate_x, f"the arm now reaches the cap (arm {arm_x:.4f} m, plate {plate_x:.4f} m)"
    assert plate_x - arm_x > 0.03, f"the gap closed to {(plate_x - arm_x) * 1000:.1f} mm"


@pytest.mark.parametrize(
    ("pattern", "measured"),
    [
        (r"tool point reaches x <= ([0-9.]+) m", 0),
        (r"cap plate starts at x = ([0-9.]+) m", 1),
    ],
)
def test_the_example_quotes_the_geometry_it_is_scored_against(pattern: str, measured: int) -> None:
    """A quoted reach that drifts from the model is a new promise, not a typo."""
    quoted = re.search(pattern, _prose())
    assert quoted, f"the example no longer states its reach: {pattern}"
    assert abs(float(quoted.group(1)) - _reach()[measured]) <= 0.01, quoted.group(0)


@pytest.mark.parametrize(
    "invitation",
    ["point --policy at a", "which does not act"],
)
def test_the_example_does_not_send_a_policy_at_the_unreachable_clause(invitation: str) -> None:
    """No provider can pass this scene, and ``mock`` does drive the joints."""
    present = invitation in _prose()
    assert not present, f"the example still claims {invitation!r}"


def test_the_scripted_route_flips_the_predicates_the_benchmark_scores() -> None:
    """The demo's one actuated route: drive the cap, the beads land in the tray."""
    from strands_robots.simulation.predicates import make_predicate

    module, sim = _poured_scene()
    cap_open = make_predicate("joint_above", joint="cap_slide", value=0.06)
    poured = make_predicate(
        "particles_inside", particles=module.BEADS, container="tray", min_fraction=0.8, xy_tol=0.12, z_tol=0.08
    )
    assert (cap_open(sim), poured(sim)) == (False, False), "the scene starts poured"
    sim.set_joint_positions({"cap_slide": 0.09}, robot_name="carton")
    sim.step(600)
    assert (cap_open(sim), poured(sim)) == (True, True), "the scripted pour no longer pours"
