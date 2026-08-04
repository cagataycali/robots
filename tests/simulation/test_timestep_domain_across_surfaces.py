"""Every surface that installs a physics timestep applies the one shared domain.

The integration timestep is the ``dt`` each physics substep advances by, so a
value the integrator cannot honor poisons the whole world rather than one call.
:meth:`~strands_robots.simulation.base.SimEngine._validate_timestep` is the
shared domain that says so, and its docstring names the surface it was lifted
from: *"This is the same contract ``MuJoCoSimEngine.set_timestep`` already
enforces, so the value cannot be set at world creation on terms the setter would
refuse."*

All three backends' ``create_world`` route through it, and so does the MuJoCo
setter. The Newton setter did not - it carried a hand-rolled
``float()``/``math.isfinite()`` pair with no ``bool`` arm - even though its own
docstring says it *"Mirrors the MuJoCo backend"* and the module that pins it
says the same. Measured on one ``create_world()``, then ``set_timestep(<value>)``,
comparing the two backends over fifteen values:

* ``True`` / ``numpy.True_`` / ``numpy.bool_(True)`` -> Newton
  ``status="success"``, ``Timestep: 1.0s (1Hz)``, and ``world.timestep == 1.0``:
  a one-second step, 500x the 0.002 default, installed by a value that is not a
  number. MuJoCo refused all three, and so did Newton's *own*
  ``create_world(timestep=True)``, so one backend held two domains for one field.
* ``False`` was refused by both - but on Newton only by accident, via
  ``float(False) == 0.0`` failing the ``> 0`` test rather than by being a
  boolean. Half-handled by coincidence is why nothing noticed the other half.
* the twelve remaining values (``nan``, ``inf``, ``-inf``, ``0``, ``0.0``,
  ``-0.002``, ``None``, ``[0.002]``, ``"0.002"``, ``numpy.float64(0.002)``,
  ``0.002``) already agreed, so the divergence was exactly the boolean family:
  three cells of fifteen.

A one-second ``dt`` is not merely a coarse simulation. Newton advances
``dt = timestep / substeps`` per solver step, so with the default ten substeps
that value makes one ``step()`` call cover a full second of simulated time
instead of 0.002 s. Replaying both step sizes in MuJoCo with a 1 kg 0.12 m box
released 0.60 m above the floor:

============================== ================= =================
quantity                       dt from ``True``  dt from ``0.002``
============================== ================= =================
solver dt                      0.1 s             0.0002 s
sim time per ``step()``         1.0 s            0.002 s
control steps spanning the fall 1                1500
settled height (rest = 0.06 m) 0.05508 m         0.05989 m
penetration into the floor     4.92 mm           0.11 mm
============================== ================= =================

So the whole 0.54 m fall happens between two consecutive observations - there is
no trajectory for a policy to act on - and the contact is resolved 45x worse,
the box coming to rest almost 5 mm inside the ground plane. Both under
``status="success"``, reported as ``Timestep: 1.0s (1Hz)``.

Solver-free: ``NewtonSimEngine.set_timestep`` validates and writes before it
touches the solver, so the engine here is built via ``__new__`` with only the
attributes that path reads. The pre-existing Newton pins for this method live in
``tests/simulation/newton/test_gravity_timestep.py``, which is skipped whenever
Newton/Warp are absent and asserts only ``"positive" in text`` - a substring both
domains satisfy - so it could not have caught this on either count.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import threading
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.models import SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine

_DEFAULT_DT = 0.002

# Values no integrator can honor. ``False`` is here because it is a boolean, not
# because it is zero: the coincidence that ``float(False) == 0.0`` is what made
# the missing bool arm invisible.
UNUSABLE = [
    True,
    False,
    np.True_,
    np.bool_(True),
    float("nan"),
    float("inf"),
    float("-inf"),
    0,
    0.0,
    -0.002,
    None,
    [0.002],
]

# Values the domain accepts, so a refusal here would be a regression rather than
# a fix. ``"0.002"`` and the NumPy scalar are accepted because the shared domain
# coerces anything ``float()`` accepts - see its docstring.
USABLE = [0.002, 0.5, np.float64(0.002), "0.002"]

BOOLEANS = [True, np.True_, np.bool_(True)]


def _newton_engine() -> NewtonSimEngine:
    """A Newton engine holding a created world, without a Newton install.

    ``__init__`` imports Newton/Warp and builds a solver. ``set_timestep``
    validates, takes the lock and writes ``world.timestep``; none of that is
    physics, so the engine is built via ``__new__`` with just those attributes -
    the harness ``tests/simulation/newton/test_free_base_is_not_an_actuator.py``
    uses. ``_model`` is the non-``None`` sentinel for "world created".
    """
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = SimWorld(timestep=_DEFAULT_DT, gravity=[0.0, 0.0, -9.81])
    engine._model = object()
    engine._lock = threading.RLock()
    return engine


def _stored_timestep(engine: NewtonSimEngine) -> float:
    """The dt the world currently holds.

    ``_world`` is declared ``SimWorld | None`` on the engine and the fixture
    above always builds one, so the narrowing happens here once instead of at
    every assertion.
    """
    world = engine._world
    assert world is not None
    return float(world.timestep)


def _text(result: dict[str, Any]) -> str:
    return " ".join(block["text"] for block in result.get("content", []) if "text" in block)


def _set(engine: NewtonSimEngine, value: Any) -> dict[str, Any]:
    """Call the setter with a deliberately off-type value.

    Routed through one funnel so the off-domain values, which the annotation
    ``float`` does not describe, need a single documented ``Any`` rather than a
    suppression at every call site.
    """
    return engine.set_timestep(value)


class TestTheNewtonSetterRefusesWhatNoIntegratorCanHonor:
    @pytest.mark.parametrize("value", UNUSABLE, ids=repr)
    def test_an_unusable_timestep_is_refused(self, value: Any) -> None:
        engine = _newton_engine()
        result = _set(engine, value)
        assert result["status"] == "error", f"{value!r} was accepted"
        assert "set_timestep" in _text(result)

    @pytest.mark.parametrize("value", UNUSABLE, ids=repr)
    def test_a_refused_timestep_is_not_installed(self, value: Any) -> None:
        """The world keeps its dt, so a refused call cannot half-apply.

        The write is ``world.timestep = timestep`` under the lock, and Newton
        reads that live on every step, so a value that reached it would be in
        force for the rest of the session.
        """
        engine = _newton_engine()
        _set(engine, value)
        assert _stored_timestep(engine) == pytest.approx(_DEFAULT_DT)

    @pytest.mark.parametrize("value", BOOLEANS, ids=repr)
    def test_a_boolean_is_named_as_a_boolean(self, value: Any) -> None:
        """``True`` is refused for being a boolean, not for being out of range.

        ``float(True)`` is ``1.0``, which is finite and positive, so a domain
        that only checks the number accepts it. The message has to say which
        mistake was made or the caller reads "must be positive" against a value
        that is.
        """
        result = _set(_newton_engine(), value)
        assert result["status"] == "error"
        assert "bool" in _text(result).lower()

    @pytest.mark.parametrize("value", USABLE, ids=repr)
    def test_a_usable_timestep_is_still_accepted(self, value: Any) -> None:
        engine = _newton_engine()
        result = _set(engine, value)
        assert result["status"] == "success", _text(result)
        assert _stored_timestep(engine) == pytest.approx(float(value))

    def test_a_large_but_usable_timestep_still_warns_rather_than_refusing(self) -> None:
        """The warn-not-reject arm above 0.1 s is unchanged by the new domain."""
        result = _set(_newton_engine(), 0.5)
        assert result["status"] == "success"
        assert "unusually large" in _text(result)


class TestTheSetterAndTheWorldBuilderAgree:
    """A dt the world builder refuses cannot be installed afterwards.

    ``create_world`` calls the shared domain directly, so its verdict is the
    staticmethod's verdict. Comparing the setter against it is what stops the
    two drifting again on this backend - which is the failure this module was
    written for: ``create_world(timestep=True)`` refused while
    ``set_timestep(True)`` installed a 1-second step.
    """

    @pytest.mark.parametrize("value", UNUSABLE + USABLE, ids=repr)
    def test_the_setter_matches_the_creation_domain(self, value: Any) -> None:
        creation_refuses = SimEngine._validate_timestep(value, "create_world") is not None
        setter_refuses = _set(_newton_engine(), value)["status"] == "error"
        assert setter_refuses == creation_refuses, (
            f"{value!r}: create_world refuses={creation_refuses}, set_timestep refuses={setter_refuses}"
        )


class TestBothBackendsSetTimestepAgree:
    """The Newton setter answers as the MuJoCo setter it says it mirrors does.

    Newton's docstring claims to mirror MuJoCo, and the module pinning it says
    the same; neither claim was checked against the MuJoCo verdict, and the two
    disagreed on three values.
    """

    @pytest.mark.parametrize("value", UNUSABLE + USABLE, ids=repr)
    def test_the_two_backends_return_the_same_verdict(self, value: Any) -> None:
        pytest.importorskip("mujoco")
        from strands_robots import Simulation

        newton_refuses = _set(_newton_engine(), value)["status"] == "error"

        sim = Simulation(backend="mujoco", mesh=False)
        try:
            sim.create_world()
            mujoco_refuses = sim.set_timestep(value)["status"] == "error"
        finally:
            sim.cleanup()

        assert newton_refuses == mujoco_refuses, (
            f"{value!r}: newton refuses={newton_refuses}, mujoco refuses={mujoco_refuses}"
        )


def _backend_dir() -> pathlib.Path:
    """The simulation package, derived from a symbol rather than a path literal."""
    return pathlib.Path(inspect.getfile(SimEngine)).parent


def _setters_missing_the_shared_domain(root: pathlib.Path) -> dict[str, list[str]]:
    """Backend ``set_timestep`` methods that do not call the shared domain.

    Keyed by ``<backend>/<module>.py`` so a failure names the file to fix.
    """
    found: dict[str, list[str]] = {}
    for backend in ("mujoco", "newton", "isaac"):
        for module in sorted((root / backend).glob("*.py")):
            tree = ast.parse(module.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for member in ast.iter_child_nodes(node):
                    if not isinstance(member, ast.FunctionDef) or member.name != "set_timestep":
                        continue
                    calls = {
                        call.func.attr
                        for call in ast.walk(member)
                        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    }
                    if "_validate_timestep" not in calls:
                        found.setdefault(f"{backend}/{module.name}", []).append(node.name)
    return found


def _setters_present(root: pathlib.Path) -> set[str]:
    present = set()
    for backend in ("mujoco", "newton", "isaac"):
        for module in sorted((root / backend).glob("*.py")):
            if "def set_timestep(" in module.read_text(encoding="utf-8"):
                present.add(f"{backend}/{module.name}")
    return present


class TestNoBackendCanShipAnUnsharedTimestepDomain:
    def test_every_backend_setter_calls_the_shared_domain(self) -> None:
        assert _setters_missing_the_shared_domain(_backend_dir()) == {}

    def test_the_scan_sees_the_setters_it_claims_to_cover(self) -> None:
        """Non-vacuity: name the surfaces, so a mis-rooted scan cannot pass.

        Isaac exposes no ``set_timestep``; if it gains one it joins this set and
        the guard above starts checking it.
        """
        assert _setters_present(_backend_dir()) == {
            "mujoco/simulation.py",
            "newton/simulation.py",
        }

    def test_the_scan_detects_a_setter_that_hand_rolls_the_domain(self, tmp_path: pathlib.Path) -> None:
        """A planted copy of the defect must be found, or a clean result is luck."""
        for backend in ("mujoco", "newton", "isaac"):
            (tmp_path / backend).mkdir()
        (tmp_path / "newton" / "simulation.py").write_text(
            "import math\n"
            "class NewtonSimEngine:\n"
            "    def set_timestep(self, timestep):\n"
            "        if not math.isfinite(float(timestep)) or timestep <= 0:\n"
            '            return {"status": "error"}\n'
            "        self._world.timestep = timestep\n",
            encoding="utf-8",
        )
        assert _setters_missing_the_shared_domain(tmp_path) == {"newton/simulation.py": ["NewtonSimEngine"]}
