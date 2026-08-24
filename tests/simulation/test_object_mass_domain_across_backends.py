"""Regression tests: every backend validates an ``add_object`` mass.

``SimEngine._validate_mass`` is a static method on the base class all three
backends inherit, and its docstring states the invariant these tests pin: a mass
outside ``(0, inf)`` "does not merely mis-size one object - it poisons the whole
world on the next step", and it is "the same domain ``set_body_properties``
already enforces when it writes the same ``body_mass`` field, so a mass cannot be
established at creation on terms the setter would refuse". The MuJoCo backend has
called it from ``add_object`` since object mass was hardened there. The Newton and
Isaac backends inherited it and never called it.

Measured on the 8-value probe set below, one ``add_object`` per case, with no
``newton`` / ``warp`` / ``isaacsim`` installed - 15 of 18 cells diverged from the
domain the base class defines:

* Newton stored every value verbatim on the registry entry. ``nan`` / ``inf`` /
  ``True`` reached ``builder.add_body(mass=...)`` unchanged. A *negative* mass
  silently took the ``obj.mass <= 0`` static path, so a body asked for at -1 kg
  came back immovable on a value only ``0`` is documented to mean. And a
  non-number was stored and then raised ``TypeError: '<=' not supported between
  instances of 'str' and 'int'`` out of BOTH readers of that comparison - the
  solver rebuild and ``list_objects`` - so one bad ``add_object`` left an
  already-registered object that made a later, unrelated scene query raise.
* Isaac forwarded it raw to the prim constructor and read it exactly once, at
  ``float(mass)`` while assembling the result. That is *after* the prim is
  constructed, after ``world.scene.add``, after the ``_prim_registry`` append and
  after the ``_objects`` entry - so ``mass="heavy"``, ``[0.1]`` and ``None``
  raised past the envelope this method documents as its only failure channel with
  the object already on the stage and registered. The obvious recovery, retrying
  under the same name with a usable mass, was then refused as a duplicate: the
  name was permanently taken.

One documented difference survives, and is pinned here rather than left to be
inferred: Newton documents ``mass=0`` as an alternative spelling of
``is_static=True`` and its rebuild honours it, so a zero mass is a *mode* rather
than a small mass and is not validated as a dynamic one. Isaac documents no such
spelling, so it converges on the MuJoCo contract and refuses ``0`` naming
``is_static=True`` as the remedy. A static object's mass is read by nobody on any
backend, so it is not validated there either - the scope MuJoCo already uses.

``TestNoObjectMassSurfaceDrifts`` keeps it that way structurally: every public
method of a backend engine class that takes a ``mass`` parameter must route it
through the shared helper.

What is NOT in scope, and is asserted to be unchanged in
``tests/simulation/test_pose_vector_domain_across_backends.py``: ``size``. Its
component counts are shape-dependent and the Isaac ``add_object`` docstring
promises a trailing-component fallback for a short ``size`` that neither other
backend offers, so it needs that contract settled before it can share a domain.

These tests are GL-free and need neither ``newton``/``warp`` nor ``isaacsim`` nor
a GPU: every guard runs before its method touches a solver or a stage, so calling
the unbound method with a small stand-in for ``self`` exercises it in every
environment.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import textwrap
from typing import Any

import pytest

from strands_robots.simulation import base as sim_base
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.simulation.newton.simulation import NewtonSimEngine, _is_zero_mass_sentinel
from tests.simulation.test_pose_vector_domain_across_backends import _isaac_stub, _newton_stub

NAN = float("nan")
INF = float("inf")

#: Masses no dynamic object can be given. The two non-finite values (``inf``
#: makes the first integration produce ``nan`` acceleration, and the solver
#: shares one state vector, so every *other* body goes ``nan`` with it), a
#: negative and a zero, a ``bool`` (an ``int`` subclass, so ``float(True)``
#: would mean a silent 1 kg body), and three non-numbers - the classes that
#: reached a ``<=`` comparison or a ``float()`` and raised there instead.
UNUSABLE_MASSES: tuple[Any, ...] = (0.0, -1.0, NAN, INF, True, "heavy", [0.1], None)

#: The same set minus Newton's documented ``0`` sentinel.
UNUSABLE_DYNAMIC_MASSES: tuple[Any, ...] = tuple(m for m in UNUSABLE_MASSES if m is not UNUSABLE_MASSES[0])

#: Accepted spellings of a usable mass, including the NumPy scalar a config
#: array or a randomization draw produces.
GOOD_MASSES: tuple[Any, ...] = (0.5, 2, 1e-9)


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


def _newton() -> Any:
    """A Newton stand-in that also carries the inherited mass validator."""
    return _newton_stub()


def _isaac_recording() -> tuple[Any, dict[str, int]]:
    """An Isaac stand-in that counts what a refused call must never construct."""
    stub = _isaac_stub()
    calls = {"construct": 0, "scene_add": 0}

    def construct(**kwargs: Any) -> tuple[Any, Any]:
        calls["construct"] += 1
        return object(), kwargs.get("size")

    def scene_add(handle: Any) -> None:
        calls["scene_add"] += 1

    stub._construct_shape_prim = construct
    stub._world.scene.add = scene_add
    return stub, calls


# --------------------------------------------------------------------------- #
# The shared domain                                                           #
# --------------------------------------------------------------------------- #
class TestTheSharedDomain:
    """``SimEngine._validate_mass`` is the single definition all three share."""

    @pytest.mark.parametrize("mass", UNUSABLE_MASSES)
    def test_an_unusable_mass_is_refused(self, mass: Any) -> None:
        result = SimEngine._validate_mass(mass, "add_object")
        assert result is not None, mass
        assert result["status"] == "error"
        assert "'mass'" in _text(result)

    @pytest.mark.parametrize("mass", GOOD_MASSES)
    def test_a_usable_mass_is_accepted(self, mass: Any) -> None:
        assert SimEngine._validate_mass(mass, "add_object") is None, mass

    def test_the_zero_sentinel_predicate_excludes_bool(self) -> None:
        """``False == 0`` is true, so the sentinel test must not accept it.

        Newton's ``0`` spelling of ``is_static=True`` is the one value that skips
        the shared domain. Reading a ``bool`` as that spelling would hand the
        backend a boolean under a success result, which is what every backend
        refuses.
        """
        assert _is_zero_mass_sentinel(0.0) is True
        assert _is_zero_mass_sentinel(0) is True
        assert _is_zero_mass_sentinel(False) is False
        assert _is_zero_mass_sentinel(True) is False
        assert _is_zero_mass_sentinel(0.5) is False
        assert _is_zero_mass_sentinel("0") is False
        assert _is_zero_mass_sentinel(None) is False


# --------------------------------------------------------------------------- #
# Newton                                                                      #
# --------------------------------------------------------------------------- #
class TestNewtonAddObject:
    @pytest.mark.parametrize("mass", UNUSABLE_DYNAMIC_MASSES)
    def test_an_unusable_mass_is_refused(self, mass: Any) -> None:
        result = NewtonSimEngine.add_object(_newton(), "crate", mass=mass)
        assert result["status"] == "error", (mass, result)
        assert "'mass'" in _text(result)

    @pytest.mark.parametrize("mass", UNUSABLE_DYNAMIC_MASSES)
    def test_a_refused_mass_registers_no_object(self, mass: Any) -> None:
        """No half-placed object, so the name stays reusable.

        Pre-fix every one of these registered the crate with the value stored
        verbatim on its :class:`SimObject`.
        """
        stub = _newton()
        NewtonSimEngine.add_object(stub, "crate", mass=mass)
        assert dict(stub._world.objects) == {}

    @pytest.mark.parametrize("mass", ("heavy", [0.1], None))
    def test_a_refused_mass_leaves_list_objects_usable(self, mass: Any) -> None:
        """A later, unrelated scene query still answers.

        ``list_objects`` and ``_add_object_to_builder`` both classify an object
        with ``obj.is_static or obj.mass <= 0``. Pre-fix a non-numeric mass was
        registered and that comparison then raised ``TypeError: '<=' not
        supported between instances of 'str' and 'int'`` - so one bad
        ``add_object`` made ``list_objects()`` raise for as long as the object
        stayed in the registry.
        """
        stub = _newton()
        NewtonSimEngine.add_object(stub, "crate", mass=mass)
        listed = NewtonSimEngine.list_objects(stub)
        assert listed["status"] == "success", listed

    def test_the_documented_zero_sentinel_is_still_accepted(self) -> None:
        """Documented: "``0`` or ``is_static`` makes it static"."""
        stub = _newton()
        assert NewtonSimEngine.add_object(stub, "crate", mass=0.0)["status"] == "success"
        assert stub._world.objects["crate"].mass == 0.0

    @pytest.mark.parametrize("mass", GOOD_MASSES)
    def test_a_usable_mass_is_accepted_and_stored(self, mass: Any) -> None:
        stub = _newton()
        assert NewtonSimEngine.add_object(stub, "crate", mass=mass)["status"] == "success"
        assert stub._world.objects["crate"].mass == mass

    def test_an_omitted_mass_still_takes_the_documented_default(self) -> None:
        stub = _newton()
        assert NewtonSimEngine.add_object(stub, "crate")["status"] == "success"
        assert stub._world.objects["crate"].mass == 0.1

    @pytest.mark.parametrize("mass", UNUSABLE_DYNAMIC_MASSES)
    def test_a_static_object_does_not_have_its_mass_read(self, mass: Any) -> None:
        """Scope, deliberately matching MuJoCo's.

        Both consumers of the value short-circuit on ``obj.is_static``, so a
        static object's mass reaches nothing. Validating it would refuse a call
        whose outcome the value cannot change.
        """
        stub = _newton()
        result = NewtonSimEngine.add_object(stub, "crate", mass=mass, is_static=True)
        assert result["status"] == "success", (mass, result)
        assert "crate" in stub._world.objects


# --------------------------------------------------------------------------- #
# Isaac                                                                       #
# --------------------------------------------------------------------------- #
class TestIsaacAddObject:
    @pytest.mark.parametrize("mass", UNUSABLE_MASSES)
    def test_an_unusable_mass_is_refused(self, mass: Any) -> None:
        stub, _ = _isaac_recording()
        result = IsaacSimulation.add_object(stub, "crate", mass=mass)
        assert result["status"] == "error", (mass, result)
        assert "'mass'" in _text(result)

    @pytest.mark.parametrize("mass", UNUSABLE_MASSES)
    def test_a_refused_mass_constructs_and_registers_nothing(self, mass: Any) -> None:
        """The property the raise used to break.

        Pre-fix the value was read only at ``float(mass)`` while assembling the
        result, so for ``"heavy"`` / ``[0.1]`` / ``None`` the prim was already
        constructed, added to the scene, appended to ``_prim_registry`` and
        entered in ``_objects`` before the raise escaped.
        """
        stub, calls = _isaac_recording()
        IsaacSimulation.add_object(stub, "crate", mass=mass)
        assert dict(stub._objects) == {}
        assert stub._prim_registry == []
        assert calls == {"construct": 0, "scene_add": 0}

    @pytest.mark.parametrize("mass", UNUSABLE_MASSES)
    def test_the_name_stays_reusable_after_a_refused_mass(self, mass: Any) -> None:
        """Pre-fix the retry was refused with "Object 'crate' already exists."."""
        stub, _ = _isaac_recording()
        IsaacSimulation.add_object(stub, "crate", mass=mass)
        retry = IsaacSimulation.add_object(stub, "crate", mass=0.5)
        assert retry["status"] == "success", retry
        assert "crate" in stub._objects

    def test_a_zero_mass_is_refused_with_the_static_remedy(self) -> None:
        """Isaac documents no ``mass=0`` spelling, so it takes MuJoCo's contract.

        ``0`` is not a usable mass for a dynamic body on either backend, and the
        Isaac docstring only says the value is "Ignored when ``is_static=True``"
        - so the remedy is that flag rather than a zero.
        """
        stub, _ = _isaac_recording()
        result = IsaacSimulation.add_object(stub, "crate", mass=0.0)
        assert result["status"] == "error", result
        assert "'mass'" in _text(result)

    @pytest.mark.parametrize("mass", GOOD_MASSES)
    def test_a_usable_mass_is_accepted_and_reported(self, mass: Any) -> None:
        stub, calls = _isaac_recording()
        result = IsaacSimulation.add_object(stub, "crate", mass=mass)
        assert result["status"] == "success", (mass, result)
        assert result["content"][0]["json"]["mass"] == pytest.approx(float(mass))
        assert calls == {"construct": 1, "scene_add": 1}

    @pytest.mark.parametrize("mass", UNUSABLE_MASSES)
    def test_a_static_object_does_not_have_its_mass_read(self, mass: Any) -> None:
        """Scope, matching MuJoCo's: the result reports 0.0 for a static object."""
        stub, _ = _isaac_recording()
        result = IsaacSimulation.add_object(stub, "crate", mass=mass, is_static=True)
        assert result["status"] == "success", (mass, result)
        assert result["content"][0]["json"]["mass"] == 0.0


# --------------------------------------------------------------------------- #
# Structural guard                                                            #
# --------------------------------------------------------------------------- #
#: Every public engine-class method that takes a ``mass``, as ``(backend,
#: method)``. Asserted exactly, so a scan root that resolved elsewhere fails
#: loudly instead of reporting a clean sweep over nothing.
_KNOWN_MASS_METHODS = {
    ("mujoco", "add_object"),
    ("mujoco", "set_body_properties"),
    ("newton", "add_object"),
    ("isaac", "add_object"),
}


def _scan_mass_methods(root: pathlib.Path) -> tuple[set[tuple[str, str]], list[str]]:
    """Find public engine-class methods taking ``mass``, and which skip the domain.

    Scoped to class methods deliberately: ``_construct_shape_prim`` and the
    Newton object builder also take a mass, but they are private helpers reached
    only from an already-validated ``add_object``.

    Args:
        root: The ``strands_robots/simulation`` package directory.

    Returns:
        ``(found, adrift)`` - every ``(backend, method)`` pair, and the ones with
        no call to ``_validate_mass``.
    """
    found: set[tuple[str, str]] = set()
    adrift: list[str] = []
    for backend in ("mujoco", "newton", "isaac"):
        for path in sorted((root / backend).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                for fn in [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                    if fn.name.startswith("_"):
                        continue
                    if "mass" not in [a.arg for a in fn.args.args + fn.args.kwonlyargs]:
                        continue
                    found.add((backend, fn.name))
                    validates = any(
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "_validate_mass"
                        for node in ast.walk(fn)
                    )
                    if not validates:
                        adrift.append(f"{backend}/{path.name}:{fn.lineno} {cls.name}.{fn.name}")
    return found, adrift


class TestNoObjectMassSurfaceDrifts:
    """A backend method taking a ``mass`` must route it through the domain."""

    def test_every_public_mass_surface_validates(self) -> None:
        root = pathlib.Path(inspect.getfile(sim_base)).parent
        found, adrift = _scan_mass_methods(root)
        assert found == _KNOWN_MASS_METHODS, f"the set of mass surfaces changed: {found}"
        assert adrift == [], "these accept a mass without the shared domain: " + ", ".join(adrift)

    def test_the_scanner_reports_a_planted_omission(self, tmp_path: pathlib.Path) -> None:
        """Without this, an empty result could mean a scanner matching nothing."""
        backend = tmp_path / "mujoco"
        backend.mkdir()
        (backend / "simulation.py").write_text(
            textwrap.dedent(
                """
                class Engine:
                    def add_object(self, name, mass=0.1):
                        return {"status": "success"}
                """
            ),
            encoding="utf-8",
        )
        found, adrift = _scan_mass_methods(tmp_path)
        assert found == {("mujoco", "add_object")}
        assert len(adrift) == 1
        assert "Engine.add_object" in adrift[0]
