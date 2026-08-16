"""Regression tests: every backend validates an ``add_object`` size vector.

Third axis of the ``add_object`` numeric contract to be settled across
backends, after ``color`` (#1856, on ``coerce_rgba``) and ``mass`` (#1859, on
``SimEngine._validate_mass``). The MuJoCo backend has composed the whole extent
domain since its numeric inputs were hardened - ``finite_vector_error`` for the
components, ``_validate_size`` for the count and the consumed extents - and its
``add_object`` docstring states the reason a short vector cannot be padded: it
would compile "a differently-sized object while reporting success". Newton and
Isaac applied neither half.

Measured on the 8-value probe set below, one ``add_object`` per case, with no
``newton`` / ``warp`` / ``isaacsim`` installed:

* Newton read the vector for TRUTHINESS (``size or default_size``) - the LAST
  surviving coalesce of that shape on that constructor, where the other four
  vector parameters all test ``is None``. So ``np.array([.1, .1, .1])`` - what
  extent arithmetic or a randomization draw produces - raised a bare
  ``ValueError: truth value of an array with more than one element is
  ambiguous`` straight through the structured envelope these methods document as
  their only failure channel, and ``[]`` is falsy so it read as *omitted*: the
  default ``[0.05, 0.05, 0.05]`` was applied and the call reported success.
* Everything else Newton stored verbatim on the registry entry and handed to the
  solver at REBUILD time rather than at the ``add_object`` a caller can
  attribute it to. ``[nan, .1, .1]`` and ``[inf, .1, .1]`` became a box
  half-extent, ``[True, .1, .1]`` read ``True`` as an extent, ``[None, .1, .1]``
  and ``[[0.1], .1, .1]`` handed the shape builder a ``None`` and a nested list,
  and the bare string ``"abc"`` was stored AS the size - raising
  ``TypeError: can only concatenate str (not "list") to str`` out of the box
  branch, and silently building a sphere of ``radius='a'`` out of the sphere one.
  A scalar ``0.5`` reached ``TypeError: unsupported operand type(s) for +:
  'float' and 'list'`` the same way.
* Isaac coerced with ``list(size)``, which validated nothing - the same
  non-finite / ``bool`` / ``None`` / nested values reached the prim constructor,
  ``"abc"`` was SPLIT per character into the 3-component extent
  ``['a', 'b', 'c']``, ``[]`` was forwarded as a sizeless size, and a NumPy
  array survived as ``[np.float64(0.1), ...]`` into the result ``json``. A
  scalar ``0.5`` raised ``TypeError: 'float' object is not iterable`` out of
  that very ``list()`` call, past the envelope.

MuJoCo refused all 8 unusable values in that set and accepted the NumPy vector.
``TestEveryBackendGivesTheSameVerdict`` pins the parity against a real compiled
model, in both directions.

``TestNoObjectSizeSurfaceDrifts`` keeps it that way structurally: every public
engine method taking a ``size`` must route it through a shared numeric-vector
validator. Three spellings are accepted there and the reason is recorded with
the set - unifying them is its own refactor, not this change.

What is NOT in scope, and is asserted to be unchanged by
``TestShapeDependentAxesStayOutOfScope``: every axis whose answer depends on the
shape. The component **count** each shape requires, whether a short vector may
be completed from trailing defaults (the Isaac ``size`` docstring promises it,
MuJoCo refuses it outright, Newton stores it for a later ``size[:3]`` read - three
behaviours, two documented, that cannot all survive), and whether a component
must be positive (MuJoCo bounds only the components the shape actually consumes,
so a cylinder may legitimately pass ``size[1] == 0``). #1858 tracks all three;
they need one contract decision rather than a helper default, which is why they
are not smuggled in behind a defect fix.

These tests are GL-free apart from the MuJoCo parity class and need neither
``newton``/``warp`` nor ``isaacsim`` nor a GPU: every guard runs before its
method touches a solver or a stage, so calling the unbound method with a small
stand-in for ``self`` exercises it in every environment.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import textwrap
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.utils import coerce_size_vector
from tests.simulation.test_pose_vector_domain_across_backends import _isaac_stub, _newton_stub

NAN = float("nan")
INF = float("inf")

#: Every value no backend can build a shape from. Each row is a measured
#: pre-fix acceptance on at least one of Newton / Isaac (see the module
#: docstring), and MuJoCo already refused all of them.
UNUSABLE_SIZES: tuple[Any, ...] = (
    [],
    [NAN, 0.1, 0.1],
    [INF, 0.1, 0.1],
    [True, 0.1, 0.1],
    [None, 0.1, 0.1],
    [[0.1], 0.1, 0.1],
    "abc",
    0.5,
)

#: Accepted extents, including the NumPy vector every docstring advertises and
#: the tuple a config literal produces. Paired with the plain floats each must
#: normalize to, since surviving NumPy scalars leak into agent-visible output.
GOOD_SIZES: tuple[tuple[Any, list[float]], ...] = (
    ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]),
    ((0.1, 0.2, 0.3), [0.1, 0.2, 0.3]),
    (np.array([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3]),
    ([np.float64(0.1), 0.2, 0.3], [0.1, 0.2, 0.3]),
)


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


def _isaac_recording() -> tuple[Any, dict[str, Any]]:
    """An Isaac stand-in that records what a refused call must never do."""
    stub = _isaac_stub()
    seen: dict[str, Any] = {"construct": 0, "scene_add": 0, "size": None}

    def construct(**kwargs: Any) -> tuple[Any, Any]:
        seen["construct"] += 1
        seen["size"] = kwargs.get("size")
        return object(), kwargs.get("size")

    stub._construct_shape_prim = construct
    stub._world.scene.add = lambda handle: seen.__setitem__("scene_add", seen["scene_add"] + 1)
    return stub, seen


# --------------------------------------------------------------------------- #
# The shared domain                                                           #
# --------------------------------------------------------------------------- #
class TestTheSharedDomain:
    """``coerce_size_vector`` is the single definition the backends share."""

    @pytest.mark.parametrize("size", UNUSABLE_SIZES)
    def test_an_unusable_size_is_refused(self, size: Any) -> None:
        value, err = coerce_size_vector("add_object", "size", size)
        assert value is None, size
        assert err is not None and "'size'" in err

    @pytest.mark.parametrize(("size", "expected"), GOOD_SIZES)
    def test_a_usable_size_normalizes_to_plain_floats(self, size: Any, expected: list[float]) -> None:
        value, err = coerce_size_vector("add_object", "size", size)
        assert err is None, (size, err)
        assert value == pytest.approx(expected)
        assert all(type(component) is float for component in value or [])

    def test_an_omitted_size_is_not_an_error(self) -> None:
        """``None`` means omitted, so the caller applies its own default."""
        assert coerce_size_vector("add_object", "size", None) == (None, None)

    def test_the_empty_vector_refusal_names_the_omission_spelling(self) -> None:
        """The remedy has to be discoverable, since ``[]`` used to work."""
        _, err = coerce_size_vector("add_object", "size", [])
        assert err is not None
        assert "omission" in err and "omit 'size'" in err.lower()

    def test_a_scalar_is_refused_as_not_a_vector(self) -> None:
        """Including a 0-d NumPy array, whose ``__len__`` exists and raises."""
        for scalar in (0.5, np.float64(0.5), np.array(0.5)):
            value, err = coerce_size_vector("add_object", "size", scalar)
            assert value is None and err is not None, scalar
            assert "must be a list/tuple of numbers" in err


# --------------------------------------------------------------------------- #
# Newton                                                                      #
# --------------------------------------------------------------------------- #
class TestNewtonAddObject:
    @pytest.mark.parametrize("size", UNUSABLE_SIZES)
    def test_an_unusable_size_is_refused(self, size: Any) -> None:
        stub = _newton_stub()
        result = NewtonSimEngine.add_object(stub, "crate", size=size)
        assert result["status"] == "error", (size, result)
        assert "'size'" in _text(result)

    @pytest.mark.parametrize("size", UNUSABLE_SIZES)
    def test_a_refused_size_registers_no_object(self, size: Any) -> None:
        """The name must stay reusable, so the obvious retry is not a duplicate."""
        stub = _newton_stub()
        assert NewtonSimEngine.add_object(stub, "crate", size=size)["status"] == "error"
        assert stub._world.objects == {}
        assert NewtonSimEngine.add_object(stub, "crate", size=[0.1, 0.1, 0.1])["status"] == "success"

    @pytest.mark.parametrize(("size", "expected"), GOOD_SIZES)
    def test_a_usable_size_is_stored_as_plain_floats(self, size: Any, expected: list[float]) -> None:
        stub = _newton_stub()
        assert NewtonSimEngine.add_object(stub, "crate", size=size)["status"] == "success"
        stored = stub._world.objects["crate"].size
        assert stored == pytest.approx(expected)
        assert all(type(component) is float for component in stored)

    def test_a_numpy_size_no_longer_raises_through_the_envelope(self) -> None:
        """The defect the truthiness coalesce carried, pinned as a behaviour."""
        stub = _newton_stub()
        # Annotated ``Any`` deliberately: the parameter is declared
        # ``list[float] | None``, and a NumPy array is the documented input the
        # annotation does not spell, which is what this test exists to pin.
        vector: Any = np.array([0.1, 0.1, 0.1])
        result = NewtonSimEngine.add_object(stub, "crate", size=vector)
        assert result["status"] == "success"
        assert stub._world.objects["crate"].size == pytest.approx([0.1, 0.1, 0.1])

    def test_an_omitted_size_still_takes_the_backend_default(self) -> None:
        """Membership, not truthiness: only ``None`` means omitted."""
        stub = _newton_stub()
        assert NewtonSimEngine.add_object(stub, "crate", shape="box")["status"] == "success"
        assert stub._world.objects["crate"].size == [0.05, 0.05, 0.05]

    def test_an_omitted_mesh_size_still_takes_the_mesh_default(self, tmp_path: pathlib.Path) -> None:
        """The default extent is per-shape, so ``is None`` must reach both branches."""
        asset = tmp_path / "part.obj"
        asset.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
        stub = _newton_stub()
        result = NewtonSimEngine.add_object(stub, "part", shape="mesh", mesh_path=str(asset))
        assert result["status"] == "success", result
        assert stub._world.objects["part"].size == [1.0, 1.0, 1.0]

    def test_an_empty_mesh_size_is_refused_too(self, tmp_path: pathlib.Path) -> None:
        """A mesh scale is consumed like any extent, so ``[]`` is no omission there."""
        asset = tmp_path / "part.obj"
        asset.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
        stub = _newton_stub()
        result = NewtonSimEngine.add_object(stub, "part", shape="mesh", mesh_path=str(asset), size=[])
        assert result["status"] == "error"
        assert "'size'" in _text(result)
        assert stub._world.objects == {}

    def test_an_empty_size_is_refused_rather_than_defaulted(self) -> None:
        """Replaces the pre-fix pin: ``[]`` used to apply the default extent."""
        stub = _newton_stub()
        result = NewtonSimEngine.add_object(stub, "crate", size=[])
        assert result["status"] == "error"
        assert "component count, not an" in _text(result)
        assert stub._world.objects == {}


# --------------------------------------------------------------------------- #
# Isaac                                                                       #
# --------------------------------------------------------------------------- #
class TestIsaacAddObject:
    @pytest.mark.parametrize("size", UNUSABLE_SIZES)
    def test_an_unusable_size_is_refused(self, size: Any) -> None:
        stub, _ = _isaac_recording()
        result = IsaacSimulation.add_object(stub, "crate", size=size)
        assert result["status"] == "error", (size, result)
        assert "'size'" in _text(result)

    @pytest.mark.parametrize("size", UNUSABLE_SIZES)
    def test_a_refused_size_constructs_no_prim_and_registers_nothing(self, size: Any) -> None:
        """Same invariant the mass fix established: no half-placed object."""
        stub, seen = _isaac_recording()
        assert IsaacSimulation.add_object(stub, "crate", size=size)["status"] == "error"
        assert seen["construct"] == 0
        assert seen["scene_add"] == 0
        assert stub._objects == {}
        assert stub._prim_registry == []
        assert IsaacSimulation.add_object(stub, "crate", size=[0.1, 0.1, 0.1])["status"] == "success"

    @pytest.mark.parametrize("size", UNUSABLE_SIZES)
    def test_the_scale_alias_goes_through_the_same_domain(self, size: Any) -> None:
        """``scale`` and ``size`` name one parameter, so one domain covers both."""
        stub, seen = _isaac_recording()
        result = IsaacSimulation.add_object(stub, "crate", scale=size)
        assert result["status"] == "error", (size, result)
        assert "'size'" in _text(result)
        assert seen["construct"] == 0

    @pytest.mark.parametrize(("size", "expected"), GOOD_SIZES)
    def test_a_usable_size_reaches_the_prim_as_plain_floats(self, size: Any, expected: list[float]) -> None:
        stub, seen = _isaac_recording()
        assert IsaacSimulation.add_object(stub, "crate", size=size)["status"] == "success"
        assert seen["size"] == pytest.approx(expected)
        assert all(type(component) is float for component in seen["size"])

    def test_a_numpy_size_no_longer_leaks_into_the_result_json(self) -> None:
        """The result ``json`` is agent-visible, so ``np.float64`` cannot survive."""
        stub, _ = _isaac_recording()
        vector: Any = np.array([0.1, 0.2, 0.3])
        result = IsaacSimulation.add_object(stub, "crate", size=vector)
        assert result["status"] == "success"
        reported = result["content"][0]["json"]["size"]
        assert reported == pytest.approx([0.1, 0.2, 0.3])
        assert all(type(component) is float for component in reported)

    def test_a_scalar_size_is_refused_rather_than_raising(self) -> None:
        """``list(0.5)`` used to raise ``TypeError`` past the envelope."""
        stub, _ = _isaac_recording()
        scalar: Any = 0.5
        result = IsaacSimulation.add_object(stub, "crate", size=scalar)
        assert result["status"] == "error"
        assert "must be a list/tuple of numbers" in _text(result)


# --------------------------------------------------------------------------- #
# Cross-backend parity                                                        #
# --------------------------------------------------------------------------- #
class TestEveryBackendGivesTheSameVerdict:
    """An extent one backend refuses is refused by all of them."""

    @pytest.fixture
    def mj_sim(self) -> Any:
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import Simulation

        sim = Simulation(tool_name="test_object_size_domain_parity_sim", mesh=False)
        assert sim.create_world()["status"] == "success"
        yield sim
        sim.cleanup()

    @pytest.mark.parametrize("size", UNUSABLE_SIZES)
    def test_an_unusable_size_is_refused_everywhere(self, mj_sim: Any, size: Any) -> None:
        mj = mj_sim.add_object("crate", size=size)
        nt = NewtonSimEngine.add_object(_newton_stub(), "crate", size=size)
        ic = IsaacSimulation.add_object(_isaac_recording()[0], "crate", size=size)
        assert mj["status"] == nt["status"] == ic["status"] == "error", (size, mj, nt, ic)
        assert "'size'" in _text(mj) and "'size'" in _text(nt) and "'size'" in _text(ic)

    @pytest.mark.parametrize(("size", "expected"), GOOD_SIZES)
    def test_a_usable_size_is_accepted_everywhere(self, mj_sim: Any, size: Any, expected: list[float]) -> None:
        """The parity is two-way: no backend refuses an extent another honors."""
        assert mj_sim.add_object("crate", shape="box", size=size)["status"] == "success"
        assert NewtonSimEngine.add_object(_newton_stub(), "crate", size=size)["status"] == "success"
        assert IsaacSimulation.add_object(_isaac_recording()[0], "crate", size=size)["status"] == "success"

    #: The values whose refusal is entirely the shared component domain, so all
    #: three backends must not merely agree on the verdict but state it
    #: identically. ``[]`` is excluded deliberately: MuJoCo reaches it through
    #: its per-shape count and so names the count the shape needs, which is the
    #: shape-dependent axis this change leaves alone.
    IDENTICALLY_WORDED = tuple(s for s in UNUSABLE_SIZES if not (isinstance(s, list) and not s))

    @pytest.mark.parametrize("size", IDENTICALLY_WORDED)
    def test_the_shared_refusal_has_one_wording(self, mj_sim: Any, size: Any) -> None:
        """Two spellings of one verdict is how backend domains start to drift."""
        mj = mj_sim.add_object("crate", size=size)
        nt = NewtonSimEngine.add_object(_newton_stub(), "crate", size=size)
        ic = IsaacSimulation.add_object(_isaac_recording()[0], "crate", size=size)
        assert {_text(mj), _text(nt), _text(ic)} == {_text(mj)}, (size, _text(mj), _text(nt), _text(ic))

    @pytest.mark.parametrize("size", UNUSABLE_SIZES)
    def test_a_refused_size_leaves_the_name_reusable_everywhere(self, mj_sim: Any, size: Any) -> None:
        """A refusal that consumes the name makes the obvious retry impossible."""
        assert mj_sim.add_object("crate", size=size)["status"] == "error"
        assert mj_sim.add_object("crate", shape="box", size=[0.1, 0.1, 0.1])["status"] == "success"


# --------------------------------------------------------------------------- #
# Structural: no size surface drifts off a shared domain                      #
# --------------------------------------------------------------------------- #
#: Every public engine method taking a ``size``, and the shared validator it
#: routes the vector through. Three spellings, because the live-geom writers in
#: ``mujoco/physics.py`` return the tool envelope directly while ``utils.py``
#: returns a ``(value, reason)`` pair - so ``_coerce_finite_vector`` is that
#: module's local equivalent rather than a second domain. All three refuse the
#: same component classes; folding them into one is its own refactor.
_KNOWN_SIZE_SURFACES: dict[tuple[str, str], tuple[str, ...]] = {
    ("mujoco", "add_object"): ("finite_vector_error",),
    ("mujoco", "set_geom_properties"): ("_coerce_finite_vector",),
    ("newton", "add_object"): ("coerce_size_vector",),
    ("isaac", "add_object"): ("coerce_size_vector",),
}

#: Any of these names, called on the vector, means the surface is on a shared
#: domain rather than reading the caller's value directly.
_SHARED_SIZE_VALIDATORS = ("coerce_size_vector", "finite_vector_error", "_coerce_finite_vector")


def _scan_size_surfaces(root: pathlib.Path) -> tuple[dict[tuple[str, str], tuple[str, ...]], list[str]]:
    """Find public engine-class methods taking ``size``, and which skip a domain.

    Scoped to public methods of a class deliberately: ``_construct_shape_prim``
    and the Newton object builder also take an extent, but they receive an
    already-validated one from ``add_object`` and are not caller-facing.

    Args:
        root: The ``strands_robots/simulation`` package directory.

    Returns:
        ``(found, adrift)`` - every ``(backend, method)`` pair mapped to the
        shared validators it calls, and the ones that call none.
    """
    found: dict[tuple[str, str], tuple[str, ...]] = {}
    adrift: list[str] = []
    for backend in ("mujoco", "newton", "isaac"):
        for path in sorted((root / backend).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                for fn in [n for n in ast.iter_child_nodes(cls) if isinstance(n, ast.FunctionDef)]:
                    if fn.name.startswith("_"):
                        continue
                    if "size" not in [a.arg for a in fn.args.args + fn.args.kwonlyargs]:
                        continue
                    called = {
                        node.func.id
                        for node in ast.walk(fn)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    }
                    validators = tuple(sorted(name for name in _SHARED_SIZE_VALIDATORS if name in called))
                    found[(backend, fn.name)] = validators
                    if not validators:
                        adrift.append(f"{backend}/{path.name}:{fn.lineno} {cls.name}.{fn.name}")
    return found, adrift


class TestNoObjectSizeSurfaceDrifts:
    """A backend method taking a ``size`` must route it through a shared domain."""

    def test_every_public_size_surface_validates(self) -> None:
        root = pathlib.Path(inspect.getfile(NewtonSimEngine)).parent.parent
        found, adrift = _scan_size_surfaces(root)
        assert adrift == [], "these accept a size without a shared domain: " + ", ".join(adrift)
        assert found == {k: tuple(sorted(v)) for k, v in _KNOWN_SIZE_SURFACES.items()}, (
            f"the set of size surfaces changed: {found}"
        )

    def test_the_scanner_reports_a_planted_omission(self, tmp_path: pathlib.Path) -> None:
        """Without this, an empty result could mean a scanner matching nothing."""
        backend = tmp_path / "newton"
        backend.mkdir()
        (backend / "simulation.py").write_text(
            textwrap.dedent(
                """
                class Engine:
                    def add_object(self, name, size=None):
                        return {"status": "success"}
                """
            ),
            encoding="utf-8",
        )
        found, adrift = _scan_size_surfaces(tmp_path)
        assert found == {("newton", "add_object"): ()}
        assert len(adrift) == 1
        assert "Engine.add_object" in adrift[0]

    def test_the_scanner_sees_a_planted_validator(self, tmp_path: pathlib.Path) -> None:
        """The other direction: a routed surface must not read as adrift."""
        backend = tmp_path / "newton"
        backend.mkdir()
        (backend / "simulation.py").write_text(
            textwrap.dedent(
                """
                class Engine:
                    def add_object(self, name, size=None):
                        size, err = coerce_size_vector("add_object", "size", size)
                        return {"status": "success"}
                """
            ),
            encoding="utf-8",
        )
        found, adrift = _scan_size_surfaces(tmp_path)
        assert found == {("newton", "add_object"): ("coerce_size_vector",)}
        assert adrift == []


# --------------------------------------------------------------------------- #
# The boundary: what this change deliberately does not decide                  #
# --------------------------------------------------------------------------- #
class TestShapeDependentAxesStayOutOfScope:
    """Counts, the short-vector fallback and positivity remain per-backend.

    Asserted rather than omitted so the divergence cannot be mistaken for
    settled, and so #1858 landing has to replace these rather than delete them.
    """

    def test_a_short_size_is_still_accepted_by_newton(self) -> None:
        """MuJoCo refuses ``[0.1]`` on a box; Newton stores it for a later read."""
        stub = _newton_stub()
        assert NewtonSimEngine.add_object(stub, "crate", shape="box", size=[0.1])["status"] == "success"
        assert stub._world.objects["crate"].size == [0.1]

    def test_a_short_size_is_still_accepted_by_isaac(self) -> None:
        """Its ``size`` docstring promises a trailing-component fallback."""
        stub, seen = _isaac_recording()
        assert IsaacSimulation.add_object(stub, "crate", shape="box", size=[0.1])["status"] == "success"
        assert seen["size"] == [0.1]

    def test_a_short_size_is_still_refused_by_mujoco(self) -> None:
        """The third behaviour, and the one the other two would converge on."""
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import Simulation

        sim = Simulation(tool_name="test_object_size_short_vector_scope_sim", mesh=False)
        try:
            assert sim.create_world()["status"] == "success"
            result = sim.add_object("crate", shape="box", size=[0.1])
            assert result["status"] == "error"
            assert "'size' component(s)" in _text(result)
        finally:
            sim.cleanup()

    def test_a_zero_extent_is_still_accepted_by_newton_and_isaac(self) -> None:
        """Positivity is bounded per consumed component, so it needs the counts."""
        stub = _newton_stub()
        assert NewtonSimEngine.add_object(stub, "crate", size=[0.0, 0.0, 0.0])["status"] == "success"
        isaac_stub, _ = _isaac_recording()
        assert IsaacSimulation.add_object(isaac_stub, "crate", size=[0.0, 0.0, 0.0])["status"] == "success"

    def test_the_shared_helper_takes_no_shape(self) -> None:
        """The scope boundary in one signature: no shape means no count check."""
        assert list(inspect.signature(coerce_size_vector).parameters) == ["method", "param_name", "size"]
