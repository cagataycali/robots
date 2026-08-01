"""``create_world(difficulty=...)`` accepts one domain on every backend.

``difficulty`` is the terrain curriculum knob: it multiplies the heightfield's
peak elevation, so ``1.0`` is the full-height terrain and ``0.5`` a gentler
stage a trainer ramps across resets. Only the MuJoCo backend has a heightfield
to scale; Newton and Isaac reject a non-default value as inert. All three
therefore have to agree on which values are *numbers* in the first place, and
they did not:

* ``None`` and ``[0.5]`` reached a bare ``float(difficulty)`` and raised
  ``TypeError``. Every caller catches ``ValueError`` only, so the exception
  escaped the ``{"status": "error"}`` tool-result contract on all three
  backends - and on the MuJoCo path it escaped after the previous world had
  already been torn down.
* A non-numeric string surfaced ``float()``'s own message
  (``could not convert string to float: 'abc'``) on MuJoCo, naming neither the
  parameter nor the surface, and escaped outright on Newton and Isaac.
* ``bool`` was accepted asymmetrically. As an ``int`` subclass ``True`` passed
  the ``<= 0`` test as a silent ``1.0`` - a full-height terrain
  indistinguishable from the default - while ``False`` was refused as a zero
  scale.
* A numeric string (``"0.5"``) was silently honored as a ``0.5`` scale, a
  spelling every other continuous knob in the library refuses.

The domain is now owned by
:func:`strands_robots.simulation.terrain.validate_difficulty`, the raising
binding over the shared
:func:`strands_robots.utils.positive_finite_number_error`, and every backend
reports through that one binding.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.terrain import terrain_elevation, validate_difficulty
from strands_robots.utils import positive_finite_number_error

# The refusal text every backend must produce for an unusable scale. Matching on
# it rather than on ``status == "error"`` is what distinguishes a domain refusal
# from the backend-specific "this value is inert here" message, which is a
# different (and correct) rejection.
_DOMAIN_MARKER = "terrain difficulty must be a finite number > 0"

# Values no backend can honor as an elevation scale. ``bool`` appears as both
# members of the pair: the asymmetry (``True`` accepted, ``False`` refused) was
# the sharpest form of the defect.
UNUSABLE: list[tuple[str, Any]] = [
    ("True", True),
    ("False", False),
    ("numeric string", "0.5"),
    ("non-numeric string", "abc"),
    ("None", None),
    ("list", [0.5]),
    ("nan", float("nan")),
    ("inf", float("inf")),
    ("-inf", float("-inf")),
    ("zero", 0),
    ("negative", -1.0),
    ("numpy bool", np.bool_(True)),
]

# Scales that are usable as written. ``1.0`` is the default (no curriculum), the
# others are real curriculum stages; the NumPy and ``int`` spellings are the
# natural products of a config array and a hand-written ramp.
USABLE: list[tuple[str, Any]] = [
    ("half", 0.5),
    ("default", 1.0),
    ("double", 2.0),
    ("numpy float32", np.float32(0.5)),
    ("int", 2),
]


def _create_world(target: Any, **kwargs: Any) -> Any:
    """Call ``create_world`` through one funnel.

    The tests deliberately pass values the ``difficulty: float`` annotation does
    not describe - that is the contract under test - so the call is splatted
    through ``**kwargs`` rather than written out, which keeps the type checker
    from objecting to inputs a caller must never write.
    """
    return target(**kwargs)


def _domain_refused(target: Any, **kwargs: Any) -> bool:
    """Whether the shared difficulty domain refused this call.

    Returns ``True`` only for a structured error carrying the shared refusal
    text. Anything else - a success, a different structured error, or an
    exception from a step *past* the domain check - is reported as "the domain
    let this value through", which is exactly the question being compared
    across backends. Raising is never a domain refusal: escaping the
    tool-result contract is the defect this file pins.
    """
    try:
        result = _create_world(target, **kwargs)
    except BaseException:  # noqa: BLE001 - a raise is measured, not handled
        return False
    if not isinstance(result, dict) or result.get("status") != "error":
        return False
    text = " ".join(block.get("text", "") for block in result.get("content", []))
    return _DOMAIN_MARKER in text


def _newton_stub() -> Any:
    """A Newton engine with only the state ``create_world`` reads before the guard.

    The domain check runs before any Newton/warp import, so a refusal is fully
    observable with the package absent. A value that gets *past* the guard then
    surfaces an error from this deliberately incomplete stub, which
    :func:`_domain_refused` correctly reports as "not a domain refusal".
    """
    import threading
    import types

    return types.SimpleNamespace(_world=None, _lock=threading.RLock(), default_timestep=0.005)


def _isaac_stub() -> Any:
    """An Isaac engine with only the state ``create_world`` reads before the guard."""
    import threading
    import types

    return types.SimpleNamespace(
        _world=None,
        _world_created=False,
        _lock=threading.RLock(),
        _config=types.SimpleNamespace(),
        default_timestep=0.005,
    )


def _backend_create_world_targets() -> dict[str, Any]:
    """The three ``create_world`` implementations, each bound to a callable target."""
    from strands_robots.simulation.isaac.simulation import IsaacSimulation
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    return {
        "newton": lambda **kw: NewtonSimEngine.create_world(_newton_stub(), **kw),
        "isaac": lambda **kw: IsaacSimulation.create_world(_isaac_stub(), **kw),
    }


class TestMuJoCoRefusesAScaleItCannotHonor:
    """The backend that honors ``difficulty`` reports, rather than raises."""

    @pytest.mark.parametrize("value", [v for _, v in UNUSABLE], ids=[n for n, _ in UNUSABLE])
    def test_an_unusable_scale_is_refused_through_the_tool_contract(self, value: Any) -> None:
        sim = Simulation(tool_name="difficulty_domain", mesh=False)
        try:
            result = _create_world(sim.create_world, terrain="stairs", difficulty=value)
            assert result["status"] == "error"
            text = " ".join(block.get("text", "") for block in result["content"])
            # Actionable: names the parameter, the domain and the offending value.
            assert _DOMAIN_MARKER in text
            assert repr(value) in text
            # The refusal precedes world construction, so a rejected curriculum
            # step leaves the caller with no half-built world to reason about.
            assert sim._world is None
        finally:
            sim.cleanup()

    def test_both_halves_of_the_bool_pair_are_refused(self) -> None:
        """``True`` used to pass as a silent ``1.0`` while ``False`` was refused."""
        verdicts = {}
        for value in (True, False):
            sim = Simulation(tool_name="difficulty_bool", mesh=False)
            try:
                verdicts[value] = _domain_refused(sim.create_world, terrain="stairs", difficulty=value)
            finally:
                sim.cleanup()
        assert verdicts == {True: True, False: True}

    @pytest.mark.parametrize("value", [v for _, v in USABLE], ids=[n for n, _ in USABLE])
    def test_a_usable_scale_is_still_compiled_into_the_heightfield(self, value: Any) -> None:
        """The accepted domain is unchanged: the scale still reaches the hfield."""
        sim = Simulation(tool_name="difficulty_ok", mesh=False)
        try:
            result = _create_world(sim.create_world, terrain="stairs", difficulty=value)
            assert result["status"] == "success", result
            assert sim._world is not None
            model = sim._world._model
            # hfield_size is (radius_x, radius_y, elevation, base_depth): the
            # third component is the peak the curriculum scales.
            assert float(model.hfield_size[0][2]) == pytest.approx(terrain_elevation(float(value)), abs=1e-9)
        finally:
            sim.cleanup()


class TestEveryBackendAgreesOnTheDomain:
    """A scale one ``create_world`` refuses cannot be honored by another."""

    @pytest.mark.parametrize("value", [v for _, v in UNUSABLE], ids=[n for n, _ in UNUSABLE])
    def test_an_unusable_scale_is_refused_on_every_backend(self, value: Any) -> None:
        sim = Simulation(tool_name="difficulty_parity", mesh=False)
        try:
            verdicts = {"mujoco": _domain_refused(sim.create_world, terrain="stairs", difficulty=value)}
        finally:
            sim.cleanup()
        for name, target in _backend_create_world_targets().items():
            verdicts[name] = _domain_refused(target, difficulty=value)
        assert verdicts == dict.fromkeys(verdicts, True), f"backends disagree for {value!r}: {verdicts}"

    @pytest.mark.parametrize("value", [v for _, v in USABLE], ids=[n for n, _ in USABLE])
    def test_a_usable_scale_is_not_domain_refused_on_any_backend(self, value: Any) -> None:
        """Newton/Isaac may still reject it as inert - but not as an unusable number."""
        sim = Simulation(tool_name="difficulty_parity_ok", mesh=False)
        try:
            verdicts = {"mujoco": _domain_refused(sim.create_world, terrain="stairs", difficulty=value)}
        finally:
            sim.cleanup()
        for name, target in _backend_create_world_targets().items():
            verdicts[name] = _domain_refused(target, difficulty=value)
        assert verdicts == dict.fromkeys(verdicts, False), f"backends disagree for {value!r}: {verdicts}"


class TestValidateDifficultyIsTheBindingOverTheSharedDomain:
    """One owner, so the backends cannot drift from the shared positive-real rule."""

    @pytest.mark.parametrize("value", [v for _, v in UNUSABLE + USABLE], ids=[n for n, _ in UNUSABLE + USABLE])
    def test_it_raises_exactly_when_the_shared_domain_reports(self, value: Any) -> None:
        shared_rejects = positive_finite_number_error(value, "difficulty", "terrain") is not None
        try:
            validate_difficulty(value)
            binding_rejects = False
        except ValueError:
            binding_rejects = True
        assert binding_rejects is shared_rejects, f"verdicts differ for {value!r}"

    def test_it_never_raises_a_type_error(self) -> None:
        """``TypeError`` is what escaped the callers' ``except ValueError``."""
        for _, value in UNUSABLE:
            with pytest.raises(ValueError):
                validate_difficulty(value)


class TestEveryCreateWorldRoutesThroughTheOneBinding:
    """A fourth backend cannot ship ``difficulty`` with a domain of its own."""

    @staticmethod
    def _backend_modules() -> dict[str, ast.Module]:
        # Derived from a symbol, not a path literal, so a scan that resolves
        # somewhere else fails the non-vacuity test below rather than passing
        # over an empty tree.
        root = pathlib.Path(inspect.getfile(validate_difficulty)).parent
        return {
            path.parent.name: ast.parse(path.read_text(encoding="utf-8"))
            for path in sorted(root.glob("*/simulation.py"))
        }

    @staticmethod
    def _difficulty_create_worlds(module: ast.Module) -> list[ast.FunctionDef]:
        found = []
        for node in ast.walk(module):
            if not isinstance(node, ast.FunctionDef) or node.name != "create_world":
                continue
            args = [a.arg for a in node.args.args + node.args.kwonlyargs]
            if "difficulty" in args:
                found.append(node)
        return found

    @staticmethod
    def _calls_the_binding(fn: ast.FunctionDef) -> bool:
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "validate_difficulty":
                return True
        return False

    def test_the_scan_finds_every_backend(self) -> None:
        """Non-vacuity: an empty or mis-rooted scan must not read as clean."""
        found = {name for name, module in self._backend_modules().items() if self._difficulty_create_worlds(module)}
        assert found == {"mujoco", "newton", "isaac"}, found

    def test_every_backend_calls_it(self) -> None:
        adrift = [
            f"{name}.create_world"
            for name, module in self._backend_modules().items()
            for fn in self._difficulty_create_worlds(module)
            if not self._calls_the_binding(fn)
        ]
        assert not adrift, (
            f"{adrift} accept difficulty= without reaching validate_difficulty; "
            "a second copy of the domain drifts from the shared one"
        )

    def test_the_scanner_detects_a_planted_omission(self) -> None:
        """Guard the guard: a scanner that matched nothing would look clean."""
        planted = ast.parse(
            "class Engine:\n"
            "    def create_world(self, terrain=None, difficulty=1.0):\n"
            "        return {'status': 'success'}\n"
        )
        found = self._difficulty_create_worlds(planted)
        assert len(found) == 1
        assert not self._calls_the_binding(found[0])
