# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: Isaac's ``add_camera`` refuses a pose or fov the siblings refuse.

``add_camera`` is an agent-callable configuration surface, and on the Isaac
backend it validated none of its numeric inputs. It was the weakest of the three
implementations: ``position``/``target`` were copied with a bare
``list(position)``, which reads no element at all, so every one of these was
registered under ``status="success"`` and handed to ``_create_camera_prim``:

* a ``nan``/``inf`` component, baked into the USD camera's world pose and into
  the ``set_camera_view`` look-at basis;
* a non-numeric element (``["x", 2.0, 2.0]``) and a ``bool`` (``True`` read as
  the coordinate ``1.0``);
* a wrong-length vector - ``[2.0, 2.0]`` and ``[]`` both reached the ``Camera``
  constructor;
* a NumPy pose, which then leaked ``np.float64(2.0)`` into the agent-visible
  status text and the ``json`` payload.

``fov`` was coerced with a bare ``float(fov)``, *outside* the method's try block,
so a non-numeric value raised a ``ValueError`` straight through the structured
``{"status": "error"}`` tool-result contract every ``AgentTool`` handler owes its
caller, and ``nan``/``inf``/``0``/``>= 180`` registered a camera the RTX pipeline
cannot use: ``_create_camera_prim`` derives
``focal_length = horizontal_aperture / (2 * tan(radians(fov) / 2))``, which
raises ``ZeroDivisionError`` for ``0`` - a type absent from that try block's
except tuple, so it escapes ``add_camera`` - and yields ``nan`` (for ``nan``) or
7.3e-16 mm (for ``180``), both of which ``set_focal_length`` accepts.

The fix routes the pose through the shared
:func:`~strands_robots.utils.coerce_pose_vector` and the fov through
:func:`~strands_robots.utils.camera_fov_error`, the same domains the MuJoCo and
Newton backends' ``add_camera`` already use.
``TestSameVerdictAsTheMujocoSibling`` pins that parity against the live MuJoCo
method, which is the invariant ``pose_vector_error``'s docstring states: a pose
either backend entry point refuses must be refused by the other.

None of this needs Isaac Sim or a GPU. The validation runs before the method
touches the stage, and the one call that would (``_create_camera_prim``) is
replaced on the instance, so the accepted-value paths are exercised too - the
skeleton-via-``__new__`` fixture shape ``test_cameras_recording_preflight_guards.py``
uses.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation

_NON_FINITE = (float("nan"), float("inf"), float("-inf"))

#: Values that are not a real number at all. ``True`` is included because
#: ``bool`` is an ``int`` subclass, so a coercion would silently write the
#: coordinate ``1.0`` where a caller passed a truth value by mistake.
_NON_NUMERIC_ELEMENTS = ("x", None, [0.0], True)

#: ``fov`` values no camera can use. ``0`` divides by zero in the derived focal
#: length and ``180`` is the open-interval boundary (``tan(pi/2)``).
_BAD_FOVS = (float("nan"), float("inf"), float("-inf"), 0.0, -1.0, -60.0, 180.0, 181.0, 360.0, True, "wide")

#: ``fov`` values that must keep working, including a NumPy scalar read out of a
#: config array and a plain ``int``.
_GOOD_FOVS = (60.0, 45, 1.0, 179.9, np.float32(58.0), np.float64(90.0))


class _FakeCameraHandle:
    """Stand-in for the Isaac ``Camera`` sensor handle."""


def _engine() -> IsaacSimulation:
    """Skeleton ``IsaacSimulation`` carrying only what ``add_camera`` reads.

    ``_create_camera_prim`` is the single call that needs the Kit runtime; it is
    replaced with a recorder so a refused configuration can be distinguished
    from an accepted one by whether the prim was ever requested.
    """
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._config = IsaacConfig()
    engine._lock = threading.RLock()
    engine._world = None
    engine._world_created = True
    engine._robots = {}
    engine._objects = {}
    engine._cameras = {}
    engine._prim_registry = []
    engine._cam_out_size = {}
    engine._camera_warmup_steps = 0
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._main_tid = threading.get_ident()

    prim_calls: list[dict[str, Any]] = []

    def _create_camera_prim(**kwargs: Any) -> tuple[Any, float]:
        prim_calls.append(kwargs)
        return _FakeCameraHandle(), 24.0

    engine._create_camera_prim = _create_camera_prim  # type: ignore[method-assign]
    engine.prim_calls = prim_calls  # type: ignore[attr-defined]
    return engine


def _add_camera(engine: IsaacSimulation, **kwargs: Any) -> dict[str, Any]:
    """Call ``add_camera`` with arguments deliberately outside its annotations.

    Most inputs below are values a caller must never pass - a non-numeric pose
    element, a ``bool``, a wrong-length vector, a string ``fov`` - because the
    contract under test is that they are refused at runtime, in a structured
    result, rather than by a type checker that an agent dispatching this tool
    never runs. The NumPy poses are the opposite case: the method accepts them
    (``coerce_pose_vector`` normalizes them) but the ``list[float]`` annotation
    all three backends share does not describe them. Funnelling every call
    through one ``**kwargs: Any`` boundary states that once, the shape the
    Newton sibling test uses, instead of scattering per-call suppressions.
    """
    return engine.add_camera(**kwargs)


def _assert_refused(engine: IsaacSimulation, result: dict, needle: str = "") -> None:
    """A refusal is a structured error that left no trace of the camera."""
    assert result["status"] == "error", result
    assert needle in result["content"][0]["text"], result
    assert engine._cameras == {}
    assert engine._prim_registry == []
    assert engine._cam_out_size == {}
    # The stage was never touched: validation runs ahead of the one call that
    # would create the USD prim.
    assert engine.prim_calls == []  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# position / target                                                           #
# --------------------------------------------------------------------------- #
class TestPoseFiniteness:
    @pytest.mark.parametrize("bad", _NON_FINITE)
    @pytest.mark.parametrize("param", ["position", "target"])
    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_non_finite_component_is_refused(self, param: str, bad: float, index: int) -> None:
        """A nan/inf component in either pose vector is refused, in any slot.

        Pre-fix every one of these returned ``success`` and passed the component
        to the USD camera's world pose.
        """
        engine = _engine()
        vec = [1.0, 1.0, 1.0]
        vec[index] = bad
        result = _add_camera(engine, name="cam", **{param: vec})
        _assert_refused(engine, result, "finite")


class TestPoseElementType:
    @pytest.mark.parametrize("bad", _NON_NUMERIC_ELEMENTS)
    @pytest.mark.parametrize("param", ["position", "target"])
    def test_non_numeric_component_is_refused(self, param: str, bad: object) -> None:
        engine = _engine()
        result = _add_camera(engine, name="cam", **{param: [bad, 1.0, 1.0]})
        _assert_refused(engine, result)

    def test_bool_component_is_not_read_as_one(self) -> None:
        """``True`` is refused rather than reaching the prim as the coordinate 1.0."""
        engine = _engine()
        result = _add_camera(engine, name="cam", position=[True, 2.0, 2.0])
        _assert_refused(engine, result)


class TestPoseLength:
    @pytest.mark.parametrize("vec", [[], [0.5], [0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
    @pytest.mark.parametrize("param", ["position", "target"])
    def test_wrong_length_is_refused(self, param: str, vec: list[float]) -> None:
        engine = _engine()
        result = _add_camera(engine, name="cam", **{param: vec})
        _assert_refused(engine, result)

    def test_empty_vector_is_not_read_as_omitted(self) -> None:
        """An empty vector is a wrong-length request, not "take the default"."""
        engine = _engine()
        result = _add_camera(engine, name="cam", position=[])
        _assert_refused(engine, result, "3-element")


class TestCoincidentPose:
    def test_eye_equal_to_target_is_refused(self) -> None:
        """A camera whose eye is its own look-at point has no look direction.

        Both siblings already refuse it; Isaac handed the pair to
        ``set_camera_view``, whose forward axis is then a zero vector.
        """
        engine = _engine()
        result = _add_camera(engine, name="cam", position=[1.0, 1.0, 1.0], target=[1.0, 1.0, 1.0])
        _assert_refused(engine, result, "look direction")

    def test_omitted_target_is_not_a_coincident_pose(self) -> None:
        """``target=None`` keeps the constructed orientation - not a refusal.

        Isaac's documented default is no look-at at all (unlike MuJoCo/Newton,
        which default the target to the origin), so the check must be scoped to
        a target the caller actually supplied.
        """
        engine = _engine()
        result = _add_camera(engine, name="cam", position=[0.0, 0.0, 0.0])
        assert result["status"] == "success", result
        assert engine.prim_calls[0]["target"] is None  # type: ignore[attr-defined]


class TestPoseAccepted:
    def test_omitted_vectors_take_the_documented_defaults(self) -> None:
        engine = _engine()
        result = _add_camera(engine, name="cam")
        assert result["status"] == "success", result
        assert result["content"][0]["json"]["position"] == [2.0, 2.0, 2.0]
        assert result["content"][0]["json"]["target"] is None

    def test_int_components_are_accepted(self) -> None:
        engine = _engine()
        result = _add_camera(engine, name="cam", position=[1, 2, 3], target=[0, 0, 0])
        assert result["status"] == "success", result

    def test_numpy_pose_is_accepted_and_normalised_to_floats(self) -> None:
        """A NumPy pose still works, and no ``np.float64`` reaches the caller.

        Pre-fix the echoed json and the status text read
        ``[np.float64(2.0), np.float64(2.0), np.float64(2.0)]``, because
        ``list(np.array(...))`` copies the array's scalars unchanged.
        """
        engine = _engine()
        result = _add_camera(
            engine,
            name="cam",
            position=np.array([0.6, 0.6, 0.5]),
            target=np.array([0.0, 0.0, 0.15]),
        )
        assert result["status"] == "success", result
        cam_info = result["content"][0]["json"]
        assert all(type(v) is float for v in cam_info["position"])
        assert all(type(v) is float for v in cam_info["target"])
        assert "np.float64" not in result["content"][0]["text"]


# --------------------------------------------------------------------------- #
# fov                                                                         #
# --------------------------------------------------------------------------- #
class TestFov:
    @pytest.mark.parametrize("bad", _BAD_FOVS)
    def test_unusable_fov_is_refused(self, bad: object) -> None:
        engine = _engine()
        result = _add_camera(engine, name="cam", fov=bad)
        _assert_refused(engine, result, "fov")

    @pytest.mark.parametrize("good", _GOOD_FOVS)
    def test_usable_fov_is_accepted(self, good: object) -> None:
        engine = _engine()
        result = _add_camera(engine, name="cam", fov=good)
        assert result["status"] == "success", result
        fov_deg = result["content"][0]["json"]["fov"]
        assert type(fov_deg) is float
        assert fov_deg == pytest.approx(float(good))  # type: ignore[arg-type]

    def test_non_numeric_fov_returns_an_error_and_does_not_raise(self) -> None:
        """The tool-result contract: a bad fov is an error dict, never an exception.

        Pre-fix ``float("wide")`` raised a bare ``ValueError`` from outside the
        method's own try block, so nothing converted it into the structured
        response.
        """
        engine = _engine()
        result = _add_camera(engine, name="cam", fov="wide")
        assert result["status"] == "error", result

    def test_zero_fov_would_divide_by_zero_in_the_derived_focal_length(self) -> None:
        """Pin why ``0`` is outside the domain rather than merely unusual.

        ``ZeroDivisionError`` is not in ``add_camera``'s except tuple, so on a
        real Isaac install this propagated out of the tool call.
        """
        with pytest.raises(ZeroDivisionError):
            24.0 / (2.0 * math.tan(math.radians(0.0) / 2.0))

    def test_boundary_fov_would_give_a_degenerate_focal_length(self) -> None:
        """``180`` does not raise - it silently produces an unusable lens."""
        assert 24.0 / (2.0 * math.tan(math.radians(180.0) / 2.0)) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Refusals leave nothing behind                                               #
# --------------------------------------------------------------------------- #
class TestRefusalLeavesNoPartialState:
    def test_a_refused_call_does_not_consume_the_camera_name(self) -> None:
        """The name must still be free afterwards.

        Otherwise the caller has to ``remove_camera`` a camera that was never
        added.
        """
        engine = _engine()
        assert _add_camera(engine, name="wrist", fov=float("nan"))["status"] == "error"
        assert engine._cameras == {}
        assert _add_camera(engine, name="wrist", fov=60.0)["status"] == "success"
        assert "wrist" in engine._cameras

    def test_a_bad_pose_is_reported_even_when_the_name_is_taken(self) -> None:
        """Pose and fov are checked before the duplicate-name check.

        That is the order both siblings use, so a call that is wrong on two
        counts gets the same diagnosis from every backend.
        """
        engine = _engine()
        assert _add_camera(engine, name="front")["status"] == "success"
        result = _add_camera(engine, name="front", position=[float("nan"), 1.0, 1.0])
        assert result["status"] == "error", result
        assert "finite" in result["content"][0]["text"]


# --------------------------------------------------------------------------- #
# Cross-backend parity                                                        #
# --------------------------------------------------------------------------- #
class TestSameVerdictAsTheMujocoSibling:
    """The backends' ``add_camera`` must accept and refuse the same configs.

    ``pose_vector_error`` documents the invariant this pins: a pose either
    backend entry point refuses must be refused by the other. Isaac was the
    entry point that refused nothing.
    """

    @pytest.fixture
    def mj_sim(self):
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import Simulation

        sim = Simulation(tool_name="test_isaac_add_camera_parity_sim", mesh=False)
        sim.create_world(gravity=[0, 0, -9.81])
        yield sim
        sim.cleanup()

    @pytest.mark.parametrize(
        "kwargs",
        [
            # Refused by both.
            {"position": [float("nan"), 1.0, 1.0]},
            {"position": [float("inf"), 1.0, 1.0]},
            {"target": [0.0, float("-inf"), 0.0]},
            {"position": [True, 1.0, 1.0]},
            {"position": ["x", 1.0, 1.0]},
            {"position": [1.0, 1.0]},
            {"position": []},
            {"target": []},
            {"position": [1.0, 1.0, 1.0], "target": [1.0, 1.0, 1.0]},
            {"fov": float("nan")},
            {"fov": 0.0},
            {"fov": -30.0},
            {"fov": 180.0},
            {"fov": 400.0},
            {"fov": "wide"},
            {"fov": True},
            # Accepted by both.
            {},
            {"fov": 60.0},
            {"fov": 179.9},
            {"fov": np.float32(58.0)},
            {"position": [1, 2, 3], "target": [0, 0, 0]},
            {"position": np.array([0.6, 0.6, 0.5]), "target": np.array([0.0, 0.0, 0.15])},
        ],
    )
    def test_verdicts_agree(self, mj_sim, kwargs: dict) -> None:
        isaac_result = _add_camera(_engine(), name="parity_cam", **kwargs)
        mujoco_result = mj_sim.add_camera(name="parity_cam", **kwargs)
        assert isaac_result["status"] == mujoco_result["status"], (
            f"{kwargs!r}: isaac said {isaac_result['status']}, mujoco said {mujoco_result['status']} "
            f"({isaac_result['content'][0].get('text')!r} vs {mujoco_result['content'][0].get('text')!r})"
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"position": [float("nan"), 1.0, 1.0]},
            {"target": [0.0, float("-inf"), 0.0]},
            {"position": ["x", 1.0, 1.0]},
            {"position": [1.0, 1.0]},
            {"fov": float("nan")},
            {"fov": 0.0},
            {"fov": "wide"},
            {"position": [1.0, 1.0, 1.0], "target": [1.0, 1.0, 1.0]},
        ],
    )
    def test_refusal_text_is_the_shared_one(self, mj_sim, kwargs: dict) -> None:
        """One definition, not a copy: the two backends word it identically.

        A duplicated interval or finiteness check would drift; identical text is
        evidence both callers reach the same helper.
        """
        isaac_result = _add_camera(_engine(), name="parity_cam", **kwargs)
        mujoco_result = mj_sim.add_camera(name="parity_cam", **kwargs)
        assert isaac_result["content"][0]["text"] == mujoco_result["content"][0]["text"]
