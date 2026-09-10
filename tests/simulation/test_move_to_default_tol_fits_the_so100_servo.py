"""``move_to``'s default tolerance is one the bundled so100 servos can settle to.

The first motion on ``Robot("so100")`` - end-effector +0.10 m straight up, a
reachable point - returned ``did not reach ... within tol=0.01 m after
max_steps=200 (residual 0.0105 m; IK residual was 0.0085 m)``; retrying the same
target with ``tol=0.015`` from that pose "reached ... in 1 steps", because the
arm was already there. One wasted caller turn per motion for a 0.5 mm miss. The
default is now 0.015 m, where those servos settle inside the step budget.

The number lives in four places - a signature and a ``describe()`` entry per
backend - so the pins below grade all four against one another rather than
spelling 0.015 once and letting the other three drift.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import pathlib
import re

import pytest

#: The settling tolerance, asserted once here and derived everywhere below.
SETTLING_TOL = 0.015

#: ``backend -> (primitives module, mixin, engine module owning describe())``.
BACKENDS = {
    "mujoco": (
        "strands_robots.simulation.mujoco.motion_primitives",
        "MotionPrimitivesMixin",
        "strands_robots.simulation.mujoco.simulation",
    ),
    "isaac": (
        "strands_robots.simulation.isaac.motion_primitives",
        "IsaacMotionPrimitivesMixin",
        "strands_robots.simulation.isaac.simulation",
    ),
}

#: The ``describe()["methods"]["move_to"]`` signature line, in either spelling
#: ("position" on MuJoCo, "position=[x,y,z]" on Isaac).
_DESCRIBE_LINE = re.compile(r"^\(robot_name=None, position")
_DESCRIBE_TOL = re.compile(r"\btol=([\d.]+)")

requires_mujoco = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco not installed")


def _signature_default(backend: str) -> float:
    module, mixin, _ = BACKENDS[backend]
    move_to = getattr(importlib.import_module(module), mixin).move_to
    return float(inspect.signature(move_to).parameters["tol"].default)


def _describe_tol(backend: str) -> float:
    """The ``tol=`` the backend's ``describe()`` advertises for ``move_to``.

    Read out of the module source rather than a live engine so the Isaac half
    is graded without an Isaac Sim runtime.
    """
    source = pathlib.Path(inspect.getfile(importlib.import_module(BACKENDS[backend][2]))).read_text()
    lines = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _DESCRIBE_LINE.match(node.value)
    ]
    assert len(lines) == 1, f"expected one move_to describe() line in {backend}, found {lines}"
    found = _DESCRIBE_TOL.search(lines[0])
    assert found is not None, f"{backend} describe() line does not advertise tol=: {lines[0]!r}"
    return float(found.group(1))


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_default_tol_is_the_settling_tol(backend: str) -> None:
    assert _signature_default(backend) == SETTLING_TOL


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_describe_advertises_the_signature_default(backend: str) -> None:
    assert _describe_tol(backend) == _signature_default(backend)


@requires_mujoco
def test_so100_reaches_ten_cm_up_with_the_default_tol() -> None:
    from strands_robots import Robot

    sim = Robot("so100")
    try:
        ee = sim.get_body_state(body_name="so100/Wrist_Pitch_Roll")["content"][1]["json"]["position"]
        target = [ee[0], ee[1], ee[2] + 0.10]
        res = sim.move_to(robot_name="so100", position=target)
        assert res["status"] == "success", res["content"][0]["text"]
    finally:
        sim.destroy()
