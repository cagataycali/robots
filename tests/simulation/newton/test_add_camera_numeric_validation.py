"""Regression tests: the Newton backend's ``add_camera`` refuses a non-finite pose or fov.

``add_camera`` is an agent-callable configuration surface, and on the Newton
backend it validated neither the contents of ``position`` / ``target`` nor
``fov`` at all. The two failure classes the numeric-input campaign targets both
slipped through, each surfacing far from the call that caused it:

* ``position`` / ``target`` were coerced with a bare ``float(v)``. That caught a
  non-numeric element but passed a ``nan`` / ``inf`` component - and a ``bool``,
  reading ``True`` as the coordinate ``1.0``. Nothing downstream caught it
  either: the degenerate-orientation check compares
  ``abs(pos[i] - tgt[i]) < 1e-9``, which is ``False`` for a ``nan``, and
  ``_look_at_quat`` then divides the view vector by a ``nan`` norm, so
  ``render`` / ``get_frame`` produced a frame from an all-NaN camera quaternion
  while ``add_camera`` reported ``status="success"``.
* ``fov`` was coerced by a bare ``float(fov)`` inside the lock. A non-numeric
  value therefore raised a ``ValueError`` straight through the structured
  ``{"status": "error"}`` tool-result contract, and ``nan`` / ``inf`` / ``0`` /
  ``>= 180`` registered a degenerate camera under a success result:
  ``get_camera_params`` derives the pinhole intrinsics
  ``0.5 * h / tan(radians(fov) / 2)``, which is ``nan`` for a ``nan`` fov and
  raises ``ZeroDivisionError`` for ``0``.

The fix routes both through the shared
:func:`~strands_robots.utils.pose_vector_error` /
:func:`~strands_robots.utils.camera_fov_error` domains that the MuJoCo backend's
``add_camera`` already uses, so a camera configuration one backend refuses is
refused by both. ``TestSameVerdictAsTheMujocoSibling`` pins that parity
directly against the live MuJoCo method.

Most of these need neither the optional ``newton`` / ``warp`` packages nor a
GPU: the validation runs before ``add_camera`` touches the solver, so calling
the unbound method with a small stand-in for ``self`` (the pattern
``test_camera_lookat_quat.py`` uses) exercises it in every environment.
"""

from __future__ import annotations

import math
import threading
import types

import numpy as np
import pytest

from strands_robots.simulation.models import SimCamera
from strands_robots.simulation.newton.simulation import NewtonSimEngine

# --------------------------------------------------------------------------- #
# Probe sets, shared by the Newton tests and the MuJoCo parity test below.     #
# --------------------------------------------------------------------------- #
_NON_FINITE = (float("nan"), float("inf"), float("-inf"))

#: Values that are not a real number at all. ``True`` is included because
#: ``bool`` is an ``int`` subclass, so ``float(True)`` would silently write the
#: coordinate ``1.0`` where a caller passed a truth value by mistake.
_NON_NUMERIC_ELEMENTS = ("x", None, [0.0], True)

#: ``fov`` values no camera can use. ``0`` divides by zero in the derived
#: intrinsics and ``180`` is the open-interval boundary (``tan(pi/2)``).
_BAD_FOVS = (float("nan"), float("inf"), float("-inf"), 0.0, -1.0, -60.0, 180.0, 181.0, 360.0, True)

#: ``fov`` values that must keep working, including a NumPy scalar read out of a
#: config array and a plain ``int``.
_GOOD_FOVS = (60.0, 45, 1.0, 179.9, np.float32(58.0), np.float64(90.0))


def _engine_stub(body_labels: tuple[str, ...] = ("ground", "ball")) -> types.SimpleNamespace:
    """A stand-in for ``self`` carrying only what ``add_camera`` reads.

    ``add_camera`` touches ``self._world`` (its ``cameras`` dict), ``self._model``
    (its ``body_label`` list) and ``self._lock``. None of those need Newton, so a
    namespace with the three lets the validation run without the optional
    ``newton`` / ``warp`` packages or a GPU.
    """
    return types.SimpleNamespace(
        _world=types.SimpleNamespace(cameras={}),
        _model=types.SimpleNamespace(body_label=list(body_labels)),
        _lock=threading.RLock(),
    )


def _add_camera(stub: types.SimpleNamespace, **kwargs: object) -> dict:
    return NewtonSimEngine.add_camera(stub, **kwargs)  # type: ignore[arg-type]


def _look_at(eye, target):
    """``_look_at_quat`` via the same newton-free stand-in the sibling test uses."""
    stub = types.SimpleNamespace(_quat_from_matrix=NewtonSimEngine._quat_from_matrix)
    return NewtonSimEngine._look_at_quat(stub, eye, target)


# --------------------------------------------------------------------------- #
# position / target                                                           #
# --------------------------------------------------------------------------- #
class TestPoseFiniteness:
    @pytest.mark.parametrize("bad", _NON_FINITE)
    @pytest.mark.parametrize("param", ["position", "target"])
    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_non_finite_component_is_refused(self, param, bad, index):
        """A nan/inf component in either pose vector is refused, in any slot.

        Pre-fix every one of these returned ``success`` and stored the value.
        """
        stub = _engine_stub()
        vec = [0.5, 0.5, 0.5]
        vec[index] = bad
        result = _add_camera(stub, name="cam", **{param: vec})
        assert result["status"] == "error", result
        assert "finite" in result["content"][0]["text"]
        assert stub._world.cameras == {}

    def test_non_finite_survives_the_degeneracy_check(self):
        """Pin the premise: the degeneracy check cannot stand in for this guard.

        ``abs(nan - 0.0) < 1e-9`` is ``False``, so the identical-position/target
        check the method already had never rejected a nan pose - which is why
        the guard has to be its own check rather than a tightening of that one.
        """
        assert not (abs(float("nan") - 0.0) < 1e-9)


class TestPoseElementType:
    @pytest.mark.parametrize("bad", _NON_NUMERIC_ELEMENTS)
    @pytest.mark.parametrize("param", ["position", "target"])
    def test_non_numeric_component_is_refused(self, param, bad):
        stub = _engine_stub()
        result = _add_camera(stub, name="cam", **{param: [bad, 0.5, 0.5]})
        assert result["status"] == "error", result
        assert stub._world.cameras == {}

    def test_bool_component_is_not_read_as_one(self):
        """``True`` is refused rather than silently coerced to the coordinate 1.0.

        Pre-fix ``float(True)`` placed the camera at x=1.0 under a success
        result, so the mistake was unobservable from the caller's side.
        """
        stub = _engine_stub()
        result = _add_camera(stub, name="cam", position=[True, 2.0, 2.0])
        assert result["status"] == "error", result
        assert stub._world.cameras == {}


class TestPoseLength:
    @pytest.mark.parametrize("vec", [[], [0.5], [0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
    @pytest.mark.parametrize("param", ["position", "target"])
    def test_wrong_length_is_refused(self, param, vec):
        stub = _engine_stub()
        result = _add_camera(stub, name="cam", **{param: vec})
        assert result["status"] == "error", result
        assert stub._world.cameras == {}

    def test_empty_vector_is_not_read_as_omitted(self):
        """An empty vector is a wrong-length request, not "take the default".

        Membership, not truthiness: reading ``[]`` as omitted would place the
        camera at the default pose under a success result, silently ignoring
        what the caller asked for.
        """
        stub = _engine_stub()
        result = _add_camera(stub, name="cam", position=[])
        assert result["status"] == "error", result
        assert "3-element" in result["content"][0]["text"]


class TestPoseAccepted:
    def test_omitted_vectors_take_the_documented_defaults(self):
        stub = _engine_stub()
        result = _add_camera(stub, name="cam")
        assert result["status"] == "success", result
        cam = stub._world.cameras["cam"]
        assert cam.position == [1.0, 1.0, 1.0]
        assert cam.target == [0.0, 0.0, 0.0]

    def test_int_components_are_accepted(self):
        stub = _engine_stub()
        result = _add_camera(stub, name="cam", position=[1, 2, 3], target=[0, 0, 0])
        assert result["status"] == "success", result

    def test_numpy_pose_is_accepted_and_normalised_to_floats(self):
        """A NumPy pose - the natural product of pose arithmetic - still works.

        The stored pose is plain ``float`` so a raw ``np.float64`` never leaks
        into the agent-visible status text or the ``SimCamera`` record.
        """
        stub = _engine_stub()
        result = _add_camera(
            stub,
            name="cam",
            position=np.array([0.6, 0.6, 0.5]),
            target=np.array([0.0, 0.0, 0.15]),
        )
        assert result["status"] == "success", result
        cam = stub._world.cameras["cam"]
        assert all(type(v) is float for v in cam.position)
        assert all(type(v) is float for v in cam.target)

    def test_coincident_pose_is_still_refused(self):
        """The pre-existing degeneracy check is preserved, not replaced."""
        stub = _engine_stub()
        result = _add_camera(stub, name="cam", position=[1.0, 1.0, 1.0], target=[1.0, 1.0, 1.0])
        assert result["status"] == "error", result
        assert "look direction" in result["content"][0]["text"]


# --------------------------------------------------------------------------- #
# fov                                                                         #
# --------------------------------------------------------------------------- #
class TestFov:
    @pytest.mark.parametrize("bad", _BAD_FOVS)
    def test_unusable_fov_is_refused(self, bad):
        stub = _engine_stub()
        result = _add_camera(stub, name="cam", fov=bad)
        assert result["status"] == "error", result
        assert "fov" in result["content"][0]["text"]
        assert stub._world.cameras == {}

    @pytest.mark.parametrize("good", _GOOD_FOVS)
    def test_usable_fov_is_accepted(self, good):
        stub = _engine_stub()
        result = _add_camera(stub, name="cam", fov=good)
        assert result["status"] == "success", result
        assert stub._world.cameras["cam"].fov == pytest.approx(float(good))

    def test_non_numeric_fov_returns_an_error_and_does_not_raise(self):
        """The tool-result contract: a bad fov is an error dict, never an exception.

        Pre-fix ``float("wide")`` raised a bare ``ValueError`` from inside the
        lock, escaping the structured ``{"status": "error"}`` contract that every
        ``AgentTool`` handler owes its caller.
        """
        stub = _engine_stub()
        result = _add_camera(stub, name="cam", fov="wide")
        assert result["status"] == "error", result

    def test_zero_fov_would_divide_by_zero_in_the_derived_intrinsics(self):
        """Pin why ``0`` is outside the domain rather than merely unusual."""
        with pytest.raises(ZeroDivisionError):
            0.5 * 480 / math.tan(math.radians(0.0) / 2.0)


# --------------------------------------------------------------------------- #
# The guard is load-bearing: accepted poses render, refused ones would not.    #
# --------------------------------------------------------------------------- #
class TestAcceptedPoseAlwaysYieldsAFiniteLookAt:
    def test_every_accepted_pose_gives_a_finite_camera_quaternion(self):
        poses = [
            ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
            ([0.6, 0.6, 0.5], [0.0, 0.0, 0.15]),
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            ([0.0, 0.0, -1.0], [0.0, 0.0, 0.0]),
            ([1, 2, 3], [0, 0, 0]),
        ]
        for pos, tgt in poses:
            stub = _engine_stub()
            assert _add_camera(stub, name="cam", position=pos, target=tgt)["status"] == "success"
            cam = stub._world.cameras["cam"]
            q = _look_at(tuple(cam.position), tuple(cam.target))
            assert np.all(np.isfinite(q)), (pos, tgt, q)

    def test_the_refused_pose_would_have_produced_a_non_finite_quaternion(self):
        """Assert the premise so this file cannot pass vacuously.

        If ``_look_at_quat`` were finite for a nan pose the guard would be
        pointless; it is not, which is exactly the harm being prevented.
        """
        with np.errstate(invalid="ignore"):
            q = _look_at((float("nan"), 1.0, 1.0), (0.0, 0.0, 0.0))
        assert not np.all(np.isfinite(q))


class TestRefusalLeavesNoPartialState:
    def test_a_refused_call_does_not_consume_the_camera_name(self):
        """A rejected configuration leaves the registry untouched.

        The name must still be free afterwards - otherwise the caller has to
        ``remove_camera`` a camera that was never added.
        """
        stub = _engine_stub()
        assert _add_camera(stub, name="wrist", fov=float("nan"))["status"] == "error"
        assert stub._world.cameras == {}
        assert _add_camera(stub, name="wrist", fov=60.0)["status"] == "success"
        assert isinstance(stub._world.cameras["wrist"], SimCamera)


# --------------------------------------------------------------------------- #
# Cross-backend parity                                                        #
# --------------------------------------------------------------------------- #
class TestSameVerdictAsTheMujocoSibling:
    """The two backends' ``add_camera`` must accept and refuse the same configs.

    ``pose_vector_error`` documents the invariant this pins: a pose either
    backend entry point refuses must be refused by the other. The headline
    inconsistency this PR closes was exactly that divergence - MuJoCo refused a
    nan pose and an unusable fov, Newton accepted both.
    """

    @pytest.fixture
    def mj_sim(self):
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import Simulation

        s = Simulation(tool_name="test_newton_add_camera_parity_sim", mesh=False)
        s.create_world(gravity=[0, 0, -9.81])
        yield s
        s.cleanup()

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
    def test_verdicts_agree(self, mj_sim, kwargs):
        newton_result = _add_camera(_engine_stub(), name="parity_cam", **kwargs)
        mujoco_result = mj_sim.add_camera(name="parity_cam", **kwargs)
        assert newton_result["status"] == mujoco_result["status"], (
            f"{kwargs!r}: newton said {newton_result['status']}, mujoco said {mujoco_result['status']} "
            f"({newton_result['content'][0].get('text')!r} vs {mujoco_result['content'][0].get('text')!r})"
        )
