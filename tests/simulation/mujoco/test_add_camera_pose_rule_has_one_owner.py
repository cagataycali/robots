"""Regression tests: ``add_camera`` applies the pose rule once, not twice.

``add_camera`` bakes ``position`` and ``target`` into the camera's ``xyaxes``, so
both are validated up front. For two weeks it validated them *twice*: a
``coerce_pose_vector`` call per parameter, and then a loop that re-ran the
non-coercing twin ``pose_vector_error`` over the very values that call had just
returned.

The second application could not refuse anything. Both helpers read a pose
through the same ``_read_pose_vector``, so a value ``coerce_pose_vector``
accepts is by construction a value ``pose_vector_error`` accepts, and the two
substituted defaults are literal finite 3-vectors. Measured on a 20-value probe
set x both parameters, with the second application neutered to always return
``None``: it was invoked 24 times and **0 of 40** outcomes changed.

That made it dead code carrying a live comment. The comment justified the loop
with the two failures its own guard now owns - a non-numeric element reaching
the element-wise ``abs(pos[i] - tgt[i])`` comparison below, and a ``nan``/``inf``
slipping into the baked ``xyaxes`` - so a reader asking "which check owns the
pose contract here?" found two answers and no way to tell them apart. The line
was also permanently unreachable, so no test could ever cover it.

These tests pin the three properties that make one owner correct and keep it
that way:

* ``add_camera`` applies the pose rule exactly once per pose parameter, and no
  scene-construction method in the backend re-checks an already-coerced pose.
  This is what fails if the second application returns.
* ``coerce_pose_vector`` is total over the domain ``pose_vector_error`` judges -
  it refuses everything the twin would refuse, and everything it accepts the
  twin accepts. This is *why* the second application was removable rather than
  merely redundant, so a future weakening of the surviving guard fails here
  instead of silently reopening the hole the loop was written for.
* the degenerate-orientation guard that sat immediately below the deleted loop
  still fires. Its explicit case - two identical literal triples - is pinned by
  ``TestAddCameraTargetOrients`` in ``test_input_validation.py``; the two cases
  that reach it by a route the deleted loop stood in the middle of are not, so
  they are pinned here: a pair that becomes identical only after a default is
  substituted, and a NumPy pair, which reaches the element-wise comparison as
  plain floats only because the surviving guard normalizes it.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco import simulation as sim_mod  # noqa: E402
from strands_robots.utils import coerce_pose_vector, pose_vector_error  # noqa: E402

Simulation = sim_mod.Simulation

#: The two shared helpers that judge a pose vector. ``coerce_pose_vector``
#: validates and normalizes; ``pose_vector_error`` only reports. A method needs
#: exactly one of them per pose parameter.
POSE_RULE_HELPERS = ("coerce_pose_vector", "pose_vector_error")

#: The MuJoCo scene-construction methods that take a caller-supplied pose.
POSE_TAKING_METHODS = ("add_object", "add_robot", "move_object", "add_camera")

#: Values no pose parameter can honor. Each must be refused by the surviving
#: guard, and none may reach the element-wise comparison below it.
UNUSABLE: list[Any] = [
    [0.5, 0.3],
    [0.5, 0.3, 0.2, 0.1],
    [],
    "abc",
    3,
    True,
    [float("nan"), 0.0, 0.0],
    [float("inf"), 0.0, 0.0],
    [-float("inf"), 0.0, 0.0],
    ["a", 1.0, 2.0],
    [None, 1.0, 2.0],
    {"x": 1},
    [10**400, 0.0, 0.0],
    np.array(1.0),
]

#: Values a pose parameter honors, including the NumPy spellings pose
#: arithmetic produces.
USABLE: list[Any] = [
    [0.55, 0.0, 0.35],
    (0.55, 0.0, 0.35),
    np.array([0.55, 0.0, 0.35]),
    [np.float32(0.55), np.float64(0.0), 0.35],
    [np.int64(1), 0, 0],
]

#: The defaults ``add_camera`` substitutes for an omitted parameter. They travel
#: the same path as a supplied pose, so the rule must accept them too.
SUBSTITUTED_DEFAULTS: list[list[float]] = [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_add_camera_pose_rule_sim", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


def _pose_rule_calls(source: str, method: str) -> list[str]:
    """Names of the pose-rule helpers ``method`` calls, in source order.

    ``ast.walk`` is breadth-first, so the calls are ordered by position here
    rather than taken in traversal order.
    """
    fn = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == method)
    calls = [
        (c.lineno, c.col_offset, c.func.id)
        for c in ast.walk(fn)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in POSE_RULE_HELPERS
    ]
    return [name for _lineno, _col, name in sorted(calls)]


def _backend_source() -> str:
    """The shipped backend source, located from the module object."""
    return pathlib.Path(inspect.getfile(sim_mod)).read_text(encoding="utf-8")


def _text(result: dict[str, Any]) -> str:
    return next(c["text"] for c in result["content"] if "text" in c)


class TestThePoseRuleHasOneOwner:
    """Each pose parameter is judged by exactly one helper."""

    def test_add_camera_applies_the_rule_once_per_pose_parameter(self) -> None:
        """Two pose parameters, two calls, both to the coercing guard.

        The second application was a loop over ``(("position", pos), ("target",
        tgt))`` re-running ``pose_vector_error`` on values ``coerce_pose_vector``
        had already returned.
        """
        assert _pose_rule_calls(_backend_source(), "add_camera") == [
            "coerce_pose_vector",
            "coerce_pose_vector",
        ]

    @pytest.mark.parametrize("method", POSE_TAKING_METHODS)
    def test_no_scene_method_re_checks_an_already_coerced_pose(self, method: str) -> None:
        """Whoever coerces a pose owns it; nothing re-reports on the result."""
        calls = _pose_rule_calls(_backend_source(), method)
        assert calls, f"{method} takes a pose but calls no pose-rule helper"
        assert set(calls) == {"coerce_pose_vector"}, f"{method} re-checks a coerced pose: {calls}"

    def test_a_second_application_would_be_detected(self) -> None:
        """The scanner is not vacuous: a planted re-check is reported."""
        planted = _backend_source().replace(
            '        target, _terr = coerce_pose_vector("add_camera", "target", target, 3)\n',
            '        target, _terr = coerce_pose_vector("add_camera", "target", target, 3)\n'
            '        _ = pose_vector_error("add_camera", "target", target, 3)\n',
            1,
        )
        baseline = _pose_rule_calls(_backend_source(), "add_camera")
        assert _pose_rule_calls(planted, "add_camera") == [*baseline, "pose_vector_error"]


class TestTheCoercingGuardIsTotal:
    """``coerce_pose_vector`` refuses everything its non-coercing twin would.

    This is the invariant that made the second application removable. Both
    helpers defer to ``_read_pose_vector``; these tests pin the consequence so
    the surviving guard cannot be narrowed without a failure here.
    """

    @pytest.mark.parametrize("vec", UNUSABLE)
    def test_an_unusable_pose_is_refused_by_the_coercing_guard(self, vec: Any) -> None:
        value, err = coerce_pose_vector("add_camera", "position", vec, 3)
        assert err is not None, f"{vec!r} passed the coercing guard"
        assert value is None
        # ... and the twin agrees, so neither ordering of the two would differ.
        assert pose_vector_error("add_camera", "position", vec, 3) is not None

    @pytest.mark.parametrize("vec", USABLE)
    def test_an_accepted_pose_is_one_the_twin_also_accepts(self, vec: Any) -> None:
        value, err = coerce_pose_vector("add_camera", "position", vec, 3)
        assert err is None, err
        assert value is not None
        assert pose_vector_error("add_camera", "position", value, 3) is None

    @pytest.mark.parametrize("default", SUBSTITUTED_DEFAULTS)
    def test_the_substituted_defaults_satisfy_the_rule(self, default: list[float]) -> None:
        assert pose_vector_error("add_camera", "position", default, 3) is None

    def test_an_omitted_pose_is_not_refused(self) -> None:
        """``None`` means omitted, so the caller's own default applies."""
        assert coerce_pose_vector("add_camera", "position", None, 3) == (None, None)


class TestTheSurvivingGuardReportsTheSharedVerdict:
    """``add_camera``'s refusal is the shared rule's message, verbatim."""

    @pytest.mark.parametrize("param", ["position", "target"])
    @pytest.mark.parametrize("vec", UNUSABLE)
    def test_an_unusable_pose_is_refused_with_the_shared_message(self, sim, param: str, vec: Any) -> None:
        kwargs: dict[str, Any] = {"position": [0.55, 0.0, 0.35], "target": [0.2, 0.0, 0.05]}
        kwargs[param] = vec
        result = sim.add_camera("cam", **kwargs)
        assert result["status"] == "error", result
        assert _text(result) == coerce_pose_vector("add_camera", param, vec, 3)[1]
        assert "cam" not in sim._world.cameras

    def test_a_usable_pose_still_registers_the_camera(self, sim) -> None:
        result = sim.add_camera("front", position=np.array([0.55, 0.0, 0.35]), target=[0.2, 0.0, 0.05])
        assert result["status"] == "success", result
        assert "front" in sim._world.cameras
        # Normalized to plain floats, as the coercing guard documents.
        assert sim._world.cameras["front"].position == [0.55, 0.0, 0.35]


class TestTheGuardBelowTheDeletedCheckStillFires:
    """The degenerate-orientation refusal sat directly under the dead loop.

    The explicit case is already pinned by ``TestAddCameraTargetOrients`` in
    ``test_input_validation.py``. These two are the ones the deleted loop stood
    between: the comparison reading a substituted default, and the comparison
    reading a pose the surviving guard normalized out of NumPy.
    """

    def test_a_pose_that_only_collides_with_a_default_is_refused(self, sim) -> None:
        """The substituted default participates in the comparison.

        ``target`` defaults to the origin, so a camera placed at the origin has
        no look direction even though the caller named only one of the two.
        """
        result = sim.add_camera("cam", position=[0.0, 0.0, 0.0])
        assert result["status"] == "error", result
        assert "no look direction" in _text(result)
        assert "cam" not in sim._world.cameras

    def test_a_numpy_pose_reaches_the_comparison_as_floats(self, sim) -> None:
        """A NumPy pair that is degenerate is still caught.

        Before the pose rule was applied by membership this raised a bare
        ``ValueError`` from the truthiness read; the comparison below now sees
        plain floats.
        """
        result = sim.add_camera("cam", position=np.array([0.4, 0.4, 0.4]), target=np.array([0.4, 0.4, 0.4]))
        assert result["status"] == "error", result
        assert "no look direction" in _text(result)
