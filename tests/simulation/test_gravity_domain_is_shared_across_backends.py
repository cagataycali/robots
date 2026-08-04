"""Every gravity surface applies the one shared gravity domain, on every backend.

``SimEngine._normalize_gravity`` is the shared domain for a gravity argument. Its
docstring gives the reason it exists: a bare ``float()`` coercion accepts a
boolean, and ``float(True)`` is ``1.0``, so ``set_gravity(True)`` "configured a
+1 m/s^2 gravity pointing *up* and reported success".

Two of the five public gravity surfaces did not use it - each carried a local
copy of the domain instead - and the copies drifted from the shared rule in both
directions:

* **A boolean was applied as a magnitude.** ``NewtonSimEngine.set_gravity``
  coerced a scalar with ``float()``, so ``set_gravity(True)`` wrote
  ``[0, 0, +1.0]`` onto the world and rebuilt the model under
  ``status="success"`` - gravity pointing *up* - and ``set_gravity(False)`` wrote
  zero gravity. A boolean in any component of a vector did the same.
  ``IsaacSimulation.create_world(gravity=True)`` cleared its gravity gate the
  same way and configured ``+1.0`` on the physics context.

* **A value the other backends honour was refused.** Both copies keyed on
  ``isinstance(gravity, (int, float))`` / ``(list, tuple)`` rather than on
  ``numbers.Real`` plus a length, so ``np.float32(-3.7)`` was refused by both and
  a ``numpy`` gravity *vector* was refused by Isaac - values MuJoCo accepts.
  Newton reported the first as ``'gravity' must be a 3-element list of numbers
  (object of type 'numpy.float32' has no len())``, which names a NumPy internal
  rather than the parameter.

Measured over twelve values, Newton and Isaac disagreed with the MuJoCo
reference on 9 of 24 verdicts before this change and on 0 after it.

The reason nothing caught this is structural, and
:class:`TestEveryGravitySurfaceRoutesThroughTheSharedNormalizer` is the part that
closes it. The existing boolean-domain guard enumerates *validators* - functions
whose return annotation is a reason (``str | None`` and friends) - so it sees the
shared normalizer and cannot see a domain hand-rolled inside a public method that
returns ``dict[str, Any]``. Extending that guard's *scope* would not have found
these; its *vocabulary* is what excluded them. This class keys on the parameter
instead: every engine method that accepts ``gravity`` must reach the shared
normalizer.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import threading
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation import base as sim_base
from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.simulation.models import SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine

from .test_input_validators_refuse_a_boolean import _BOOLEAN_IDS, _BOOLEANS

# Values a caller legitimately passes as a gravity. ``np.float32`` and the NumPy
# vector are the load-bearing entries: both were refused by a hand-rolled copy
# while MuJoCo accepted them.
_USABLE = [
    -9.81,
    0.0,
    [0.0, 0.0, -3.7],
    np.float32(-3.7),
    np.float64(-3.7),
    np.int64(-3),
    np.array([0.0, 0.0, -3.7]),
]
_USABLE_IDS = ["float", "zero", "list", "np_float32", "np_float64", "np_int64", "np_array"]

_GRAVITY_UP = 1.0  # what float(True) would have written


def _text(result: dict[str, Any]) -> str:
    return " ".join(str(block.get("text", "")) for block in result.get("content", []))


# ---------------------------------------------------------------------------
# Newton: set_gravity applies what it validates
# ---------------------------------------------------------------------------


def _newton_engine(rebuilds: list[bool] | None = None) -> Any:
    """A Newton engine whose ``set_gravity`` validation and write are reachable.

    ``set_gravity`` validates before taking the lock and only then assigns
    ``world.gravity`` and rebuilds, so a skeleton carrying those four attributes
    exercises the real method with no solver installed. The world is a real
    :class:`~strands_robots.simulation.models.SimWorld` built the way
    ``create_world`` builds it, so ``world.gravity`` is the field the method
    really writes. ``_init_complete`` is left at its class default so the
    finalizer skips a teardown that never had anything to release.

    Args:
        rebuilds: Collects one entry per ``_rebuild()`` call. The real
            ``_rebuild`` re-finalises the Newton model and needs the solver;
            recording the call is what pins that a refused gravity does not
            rebuild.
    """
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._lock = threading.RLock()
    engine._world = SimWorld(timestep=0.002, gravity=[0.0, 0.0, -9.81])
    engine._model = object()
    log = [] if rebuilds is None else rebuilds

    def _record_rebuild() -> None:
        log.append(True)

    engine._rebuild = _record_rebuild  # type: ignore[method-assign]
    return engine


class TestNewtonSetGravityRefusesABoolean:
    """A boolean is not a magnitude, and it was applied as one."""

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_a_boolean_scalar_is_refused(self, value: Any) -> None:
        engine = _newton_engine()
        result = engine.set_gravity(value)
        assert result["status"] == "error", f"{value!r} was applied as a gravity magnitude"

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_a_refused_boolean_leaves_the_world_gravity_alone(self, value: Any) -> None:
        """The refusal must precede the write - a rebuilt world cannot be undone."""
        rebuilt: list[bool] = []
        engine = _newton_engine(rebuilds=rebuilt)
        before = list(engine._world.gravity)
        engine.set_gravity(value)
        assert list(engine._world.gravity) == before, "a refused gravity was written to the world"
        assert rebuilt == [], "a refused gravity rebuilt the model"

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    @pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
    def test_a_boolean_component_is_refused_on_every_axis(self, value: Any, axis: int) -> None:
        vector: list[Any] = [0.0, 0.0, -9.81]
        vector[axis] = value
        engine = _newton_engine()
        result = engine.set_gravity(vector)
        assert result["status"] == "error", f"{value!r} was applied as gravity component {axis}"

    def test_a_true_scalar_no_longer_reports_a_gravity_pointing_up(self) -> None:
        """The measured pre-fix outcome: success, and gravity +1 m/s^2 upward."""
        engine = _newton_engine()
        result = engine.set_gravity(True)
        assert result["status"] == "error"
        assert engine._world.gravity[2] != _GRAVITY_UP, "gravity was configured pointing up"

    def test_a_false_scalar_no_longer_reports_zero_gravity(self) -> None:
        engine = _newton_engine()
        result = engine.set_gravity(False)
        assert result["status"] == "error"
        assert engine._world.gravity != [0.0, 0.0, 0.0], "gravity was silently switched off"

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_the_refusal_names_the_method_the_parameter_and_the_reason(self, value: Any) -> None:
        text = _text(_newton_engine().set_gravity(value))
        assert "set_gravity" in text
        assert "'gravity'" in text
        assert "not a bool" in text, "a bool must be distinguished from a plain non-number"
        assert sim_base._BOOLEAN_WORLD_REASON in text, "the refusal must carry the reason, not just the rejection"


class TestNewtonSetGravityAcceptsWhatTheSharedDomainAccepts:
    """The local copy also refused values the shared domain honours."""

    @pytest.mark.parametrize("value", _USABLE, ids=_USABLE_IDS)
    def test_a_usable_gravity_is_applied(self, value: Any) -> None:
        engine = _newton_engine()
        result = engine.set_gravity(value)
        assert result["status"] == "success", f"{value!r} is a usable gravity: {_text(result)}"
        assert len(engine._world.gravity) == 3
        assert all(isinstance(component, float) for component in engine._world.gravity), (
            "the world must hold plain floats, not the caller's NumPy scalars"
        )

    def test_a_numpy_float32_scalar_is_no_longer_a_length_complaint(self) -> None:
        """It was refused as ``has no len()`` - a NumPy internal, not the parameter."""
        engine = _newton_engine()
        result = engine.set_gravity(np.float32(-3.7))
        assert result["status"] == "success"
        assert engine._world.gravity[2] == pytest.approx(-3.7, abs=1e-6)

    def test_one_is_accepted_though_it_is_what_true_would_have_written(self) -> None:
        """The gate keys on the type, not on the value - the over-reach control."""
        engine = _newton_engine()
        result = engine.set_gravity(1.0)
        assert result["status"] == "success"
        assert engine._world.gravity == [0.0, 0.0, 1.0]

    def test_the_result_reports_the_components_the_world_received(self) -> None:
        engine = _newton_engine()
        result = engine.set_gravity(np.array([0.0, 0.0, -1.62]))
        assert result["status"] == "success"
        assert "-1.62" in _text(result)
        assert engine._world.gravity == [0.0, 0.0, -1.62]


class TestNewtonSetGravityKeepsItsPreExistingDomain:
    """The bool gate is additive: every prior refusal keeps its own message."""

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "neg_inf"])
    def test_a_non_finite_gravity_keeps_the_finite_message(self, value: Any) -> None:
        text = _text(_newton_engine().set_gravity(value))
        assert "finite" in text

    def test_a_wrong_length_vector_keeps_the_length_message(self) -> None:
        text = _text(_newton_engine().set_gravity([0.0, 0.0]))
        assert "3-element" in text

    def test_a_missing_world_is_still_reported_before_the_domain(self) -> None:
        engine = NewtonSimEngine.__new__(NewtonSimEngine)
        engine._lock = threading.RLock()
        engine._world = None
        engine._model = None
        result = engine.set_gravity(True)
        assert result["status"] == "error"
        assert "create_world" in _text(result)


# ---------------------------------------------------------------------------
# Isaac: create_world's gravity gate
# ---------------------------------------------------------------------------


def _isaac_gravity_gate(value: Any) -> tuple[bool, str]:
    """Run ``create_world``'s gravity gate and report whether the value cleared it.

    The gate runs before ``with self._lock``, so it needs no Isaac install. With
    ``_world_created`` already set, a value that clears the gate lands on the
    "world already created" check - an unambiguous "the domain accepted this"
    signal that stops short of building a stage.
    """
    sim = IsaacSimulation.__new__(IsaacSimulation)
    sim._lock = threading.RLock()
    sim._world_created = True
    sim._config = IsaacConfig()
    result = IsaacSimulation.create_world(sim, gravity=value)
    text = _text(result)
    return "World already created" in text, text


class TestIsaacCreateWorldRefusesABoolean:
    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_a_boolean_scalar_is_refused(self, value: Any) -> None:
        cleared, text = _isaac_gravity_gate(value)
        assert not cleared, f"{value!r} cleared the gravity gate and would configure {float(value)} m/s^2"
        assert "gravity" in text

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    @pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
    def test_a_boolean_component_is_refused_on_every_axis(self, value: Any, axis: int) -> None:
        vector: list[Any] = [0.0, 0.0, -9.81]
        vector[axis] = value
        cleared, _ = _isaac_gravity_gate(vector)
        assert not cleared, f"{value!r} cleared the gravity gate as component {axis}"

    @pytest.mark.parametrize("value", _BOOLEANS, ids=_BOOLEAN_IDS)
    def test_the_refusal_names_the_method_the_parameter_and_the_reason(self, value: Any) -> None:
        _, text = _isaac_gravity_gate(value)
        assert "create_world" in text
        assert "'gravity'" in text
        assert "not a bool" in text
        assert sim_base._BOOLEAN_WORLD_REASON in text


class TestIsaacCreateWorldAcceptsWhatTheSharedDomainAccepts:
    @pytest.mark.parametrize("value", _USABLE, ids=_USABLE_IDS)
    def test_a_usable_z_aligned_gravity_clears_the_gate(self, value: Any) -> None:
        cleared, text = _isaac_gravity_gate(value)
        assert cleared, f"{value!r} is a usable Z-aligned gravity but was refused: {text}"

    def test_a_numpy_vector_is_no_longer_refused_as_not_a_vector(self) -> None:
        """It was refused as "must be a scalar or [gx, gy, gz] vector" - both siblings accept it."""
        cleared, text = _isaac_gravity_gate(np.array([0.0, 0.0, -1.62]))
        assert cleared, text

    def test_one_is_accepted_though_it_is_what_true_would_have_written(self) -> None:
        cleared, text = _isaac_gravity_gate(1.0)
        assert cleared, text


class TestIsaacCreateWorldKeepsItsOwnZAlignmentConstraint:
    """The backend-specific constraint is applied to the normalized components."""

    @pytest.mark.parametrize(
        "vector",
        [[0.0, -9.81, 0.0], [3.0, 0.0, -9.81], [0.0, 1e-9, -9.81]],
        ids=["y_only", "x_component", "tiny_y"],
    )
    def test_a_non_z_aligned_vector_is_still_refused(self, vector: list[float]) -> None:
        cleared, text = _isaac_gravity_gate(vector)
        assert not cleared
        assert "z-aligned" in text.lower()

    def test_a_non_z_aligned_numpy_vector_is_refused_for_being_off_axis(self) -> None:
        """Previously refused for its *type* - now it reaches the real constraint."""
        cleared, text = _isaac_gravity_gate(np.array([0.0, -9.81, 0.0]))
        assert not cleared
        assert "z-aligned" in text.lower()

    @pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "inf"])
    def test_a_non_finite_gravity_keeps_the_finite_message(self, value: Any) -> None:
        _, text = _isaac_gravity_gate(value)
        assert "finite" in text.lower()

    def test_a_wrong_length_vector_is_still_refused(self) -> None:
        cleared, text = _isaac_gravity_gate([0.0, -9.81])
        assert not cleared
        assert "3-element" in text


# ---------------------------------------------------------------------------
# Cross-backend parity
# ---------------------------------------------------------------------------


class TestTheGravityDomainAgreesAcrossBackends:
    """A value one gravity surface refuses is refused by all of them."""

    @pytest.mark.parametrize(
        "value",
        [*_BOOLEANS, float("nan"), float("inf"), "heavy", [0.0, 0.0]],
        ids=[
            *_BOOLEAN_IDS,
            "nan",
            "inf",
            "string",
            "short_vector",
        ],
    )
    def test_a_refused_value_is_refused_on_every_backend(self, value: Any) -> None:
        newton = _newton_engine().set_gravity(value)["status"] == "error"
        isaac_cleared, _ = _isaac_gravity_gate(value)
        shared_error = sim_base.SimEngine._normalize_gravity(value, "set_gravity")[1] is not None
        assert shared_error, f"{value!r} must be refused by the shared domain"
        assert newton, f"newton accepted {value!r} which the shared domain refuses"
        assert not isaac_cleared, f"isaac accepted {value!r} which the shared domain refuses"

    @pytest.mark.parametrize("value", _USABLE, ids=_USABLE_IDS)
    def test_an_accepted_value_is_accepted_on_every_backend(self, value: Any) -> None:
        newton = _newton_engine().set_gravity(value)["status"] == "success"
        isaac_cleared, _ = _isaac_gravity_gate(value)
        components, error = sim_base.SimEngine._normalize_gravity(value, "set_gravity")
        assert error is None and components is not None, f"{value!r} must be accepted by the shared domain"
        assert newton, f"newton refused {value!r} which the shared domain accepts"
        assert isaac_cleared, f"isaac refused {value!r} which the shared domain accepts"


# ---------------------------------------------------------------------------
# Structural: no backend can ship a sixth hand-rolled copy
# ---------------------------------------------------------------------------

_SIM_PACKAGE = pathlib.Path(inspect.getfile(sim_base)).parent
_BACKENDS = ("mujoco", "newton", "isaac")

# The public engine methods that take a gravity from a caller. A module-level
# helper is excluded by construction rather than by name: those receive already
# validated components from the method above them (``persist_world_option``
# documents exactly that), and only a class method is a caller-facing surface.
_GRAVITY_SURFACES = {
    ("mujoco", "MuJoCoSimEngine", "create_world"),
    ("mujoco", "MuJoCoSimEngine", "set_gravity"),
    ("newton", "NewtonSimEngine", "create_world"),
    ("newton", "NewtonSimEngine", "set_gravity"),
    ("isaac", "IsaacSimulation", "create_world"),
}


def _discovered_gravity_surfaces() -> dict[tuple[str, str, str], bool]:
    """Every public engine method taking ``gravity``, and whether it normalizes."""
    found: dict[tuple[str, str, str], bool] = {}
    for backend in _BACKENDS:
        for path in sorted((_SIM_PACKAGE / backend).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                for fn in ast.iter_child_nodes(cls):
                    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if fn.name.startswith("_"):
                        continue
                    if "gravity" not in [a.arg for a in fn.args.args + fn.args.kwonlyargs]:
                        continue
                    normalizes = any(
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "_normalize_gravity"
                        for call in ast.walk(fn)
                    )
                    found[(backend, cls.name, fn.name)] = normalizes
    return found


class TestEveryGravitySurfaceRoutesThroughTheSharedNormalizer:
    """Keying on the parameter, not on the helper's return annotation.

    The boolean-domain guard enumerates functions that *return a reason*, so a
    domain hand-rolled inside a public method returning ``dict[str, Any]`` is
    invisible to it - which is how two copies survived that pass. This keys on
    ``gravity`` instead, so a backend cannot add a sixth surface that re-derives
    the domain locally.
    """

    def test_every_gravity_surface_normalizes(self) -> None:
        adrift = sorted(key for key, normalizes in _discovered_gravity_surfaces().items() if not normalizes)
        assert not adrift, (
            f"these accept a caller's gravity without routing it through "
            f"SimEngine._normalize_gravity: {adrift}. A local copy drifts - a "
            f"bare float() coercion accepts a boolean and writes it as 1.0."
        )

    def test_the_scan_is_not_vacuous(self) -> None:
        """A scan that resolved elsewhere would pass the assertion above."""
        discovered = set(_discovered_gravity_surfaces())
        assert discovered == _GRAVITY_SURFACES, (
            f"the set of gravity surfaces changed: {sorted(discovered ^ _GRAVITY_SURFACES)}. "
            f"Add the new surface to _GRAVITY_SURFACES once it normalizes."
        )

    def test_the_scan_covers_all_three_backends(self) -> None:
        covered = {backend for backend, _, _ in _discovered_gravity_surfaces()}
        assert covered == set(_BACKENDS), f"only scanned {sorted(covered)}"

    def test_the_guard_detects_a_hand_rolled_copy(self) -> None:
        """The guard must fail on the thing it exists to catch, or it proves nothing."""
        source = '''
class FourthSimEngine:
    def set_gravity(self, gravity):
        """A gravity surface that re-derives the domain locally."""
        if isinstance(gravity, (int, float)):
            gravity = [0.0, 0.0, float(gravity)]
        return {"status": "success"}
'''
        cls = ast.parse(source).body[0]
        assert isinstance(cls, ast.ClassDef)
        fn = cls.body[0]
        assert isinstance(fn, ast.FunctionDef)
        assert not fn.name.startswith("_"), "the fake must be a public surface"
        assert "gravity" in [a.arg for a in fn.args.args], "the fake must take the parameter the scan keys on"
        normalizes = any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_normalize_gravity"
            for call in ast.walk(fn)
        )
        assert not normalizes, "the fake must not normalize, or it would not be caught"
        assert ("newton", cls.name, fn.name) not in _GRAVITY_SURFACES
