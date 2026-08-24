"""Regression tests: every backend validates an ``add_object`` colour.

``coerce_rgba`` is the single definition of the colour contract - three
components read as RGB and completed with an opaque alpha, four read as RGBA
verbatim, any other count refused - and the MuJoCo backend has honored it on
``add_object`` and ``set_geom_properties`` since colour counts were settled
there. The Newton and Isaac backends read the caller's colour directly.
Measured on the probe set below, one ``add_object`` per case, with no ``newton``
/ ``isaacsim`` installed:

* Newton read the colour for TRUTHINESS (``color or [0.5, 0.5, 0.5, 1.0]``),
  which is wrong three ways. ``np.array([1.0, 0.0, 0.0])`` - what a palette
  lookup or any colour arithmetic produces - raised a bare
  ``ValueError: truth value of an array with more than one element is
  ambiguous`` straight through the structured envelope this method documents as
  its only failure channel. ``[]`` is falsy, so it read as *omitted* and the
  default grey was painted under a success result. And every other value was
  stored verbatim: ``[0.1]`` and ``[0.1, 0.2]`` became 1- and 2-component
  colours, ``[nan, 0, 0]`` a non-finite one, ``[True, 0, 0]`` read ``True`` as
  the channel 1.0, and the bare string ``"abcd"`` was stored AS the colour.
  ``_add_object_to_builder`` then handed ``tuple(obj.color[:3])`` to the solver
  at rebuild time - a 1-component colour, reported nowhere near the call that
  supplied it.
* Isaac forwarded the colour raw and then TRUNCATED it:
  ``_construct_shape_prim`` writes ``list(color)[:3]``, so a 5-component request
  was applied as its first 3 under a success result, ``"abcd"`` was split per
  character into the colour ``['a', 'b', 'c']``, ``[]`` wrote an empty colour
  array, and a ``np.float64`` component survived into the prim. A scalar was
  accepted by ``add_object`` and raised inside ``np.asarray`` afterwards.

MuJoCo refused all 9 unusable values in that set and accepted all 3 NumPy
colours. Every message here is the shared one, so a colour one backend refuses
is refused by all of them, and an accepted colour reaches every backend as
exactly 4 plain floats - which is what makes the ``color[:3]`` reads the shape
builders do well-defined by construction rather than by the caller's discipline.

``TestNoColorSurfaceDrifts`` keeps it that way structurally: every public method
of a backend engine class that takes a ``color`` parameter must route it through
the shared helper. The scope is public methods deliberately -
``_construct_shape_prim`` / ``_create_shape_prim`` receive an already-validated
colour from ``add_object`` and are not caller-facing.

These tests are GL-free apart from the MuJoCo parity class and need neither
``newton``/``warp`` nor ``isaacsim`` nor a GPU: every guard runs before its
method touches a solver or a stage, so calling the unbound method with a small
stand-in for ``self`` exercises it in every environment.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.utils import RGBA_ACCEPTED_LENGTHS, coerce_rgba
from tests.simulation.test_pose_vector_domain_across_backends import _isaac_stub, _newton_stub

NAN = float("nan")
INF = float("inf")

#: Colours no ``add_object`` call can honor. An empty vector (the one a
#: truthiness read swallowed as *omitted*), three unusable component counts, the
#: two non-finite channels, a ``bool`` (an ``int`` subclass, so ``float(True)``
#: would silently mean the channel 1.0), a bare string (an iterable of
#: 1-character strings), and a non-iterable scalar.
UNUSABLE_COLORS: tuple[Any, ...] = (
    [],
    [0.1],
    [0.1, 0.2],
    [0.1, 0.2, 0.3, 1.0, 0.5],
    [NAN, 0.0, 0.0],
    [INF, 0.0, 0.0],
    [True, 0.0, 0.0],
    "abcd",
    0.5,
)

#: Accepted spellings, each with the 4 components it must become. The NumPy
#: forms are the point: a palette lookup produces them and the Args advertise
#: them.
GOOD_COLORS: tuple[tuple[Any, list[float]], ...] = (
    ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
    ((1.0, 0.0, 0.0), [1.0, 0.0, 0.0, 1.0]),
    ([0.1, 0.2, 0.3, 0.25], [0.1, 0.2, 0.3, 0.25]),
    (np.array([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0, 1.0]),
    (np.array([0.1, 0.2, 0.3, 0.25], dtype=np.float32), [0.1, 0.2, 0.3, 0.25]),
    ([np.float64(1.0), 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
)

#: The colour every backend documents for an omitted ``color``.
DEFAULT_GREY = [0.5, 0.5, 0.5, 1.0]


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


def _isaac_recording_stub() -> tuple[Any, dict[str, Any]]:
    """An Isaac stand-in that records what reaches the prim constructor.

    The shared stub discards its kwargs; the colour a refused call must never
    write - and the 4 plain floats an accepted one must - are only observable
    here.
    """
    stub = _isaac_stub()
    captured: dict[str, Any] = {}

    def construct(**kwargs: Any) -> tuple[Any, Any]:
        captured.update(kwargs)
        return object(), kwargs.get("size")

    stub._construct_shape_prim = construct
    return stub, captured


# --------------------------------------------------------------------------- #
# The shared domain                                                           #
# --------------------------------------------------------------------------- #
class TestTheSharedDomain:
    """``coerce_rgba`` is the single definition every call site shares."""

    @pytest.mark.parametrize("color", UNUSABLE_COLORS)
    def test_an_unusable_color_is_refused(self, color: Any) -> None:
        rgba, err = coerce_rgba("add_object", "color", color)
        assert err is not None, color
        assert rgba is None, "a refused colour must coerce nothing"
        assert err.startswith("add_object: 'color'")

    @pytest.mark.parametrize(("color", "expected"), GOOD_COLORS)
    def test_a_usable_color_normalizes_to_four_plain_floats(self, color: Any, expected: list[float]) -> None:
        """Accepted, completed to 4, and the NumPy scalars do not survive.

        The colour is stored on :class:`SimObject` (annotated ``list[float]``)
        and echoed in agent-visible status text, so a surviving ``np.float64``
        would leak ``np.float64(1.0)`` into it.
        """
        rgba, err = coerce_rgba("add_object", "color", color)
        assert err is None, (color, err)
        assert rgba is not None
        assert rgba == pytest.approx(expected)
        assert len(rgba) == 4
        assert all(type(component) is float for component in rgba)

    def test_omitted_is_not_refused(self) -> None:
        """``None`` means omitted - the caller applies its own documented default."""
        assert coerce_rgba("add_object", "color", None) == (None, None)

    def test_an_rgb_triple_is_completed_with_an_opaque_alpha(self) -> None:
        """Alpha is the one component with a default; the channels are not."""
        rgba, _ = coerce_rgba("add_object", "color", [0.2, 0.4, 0.6])
        assert rgba == [0.2, 0.4, 0.6, 1.0]

    @pytest.mark.parametrize("length", [0, 1, 2, 5, 6])
    def test_only_the_documented_counts_are_accepted(self, length: int) -> None:
        assert length not in RGBA_ACCEPTED_LENGTHS
        _rgba, err = coerce_rgba("add_object", "color", [0.5] * length)
        assert err is not None
        assert "3 or 4" in err

    def test_the_refusal_names_the_reason_for_a_boolean_channel(self) -> None:
        """A bool is refused for a stated reason, not as a generic non-number."""
        _rgba, err = coerce_rgba("add_object", "color", [True, 0.0, 0.0])
        assert err is not None
        assert "not a bool" in err
        assert "float(True) is 1.0" in err


# --------------------------------------------------------------------------- #
# Newton                                                                      #
# --------------------------------------------------------------------------- #
class TestNewtonAddObject:
    @pytest.mark.parametrize("color", UNUSABLE_COLORS)
    def test_an_unusable_color_is_refused(self, color: Any) -> None:
        result = NewtonSimEngine.add_object(_newton_stub(), "crate", color=color)
        assert result["status"] == "error", (color, result)
        assert "'color'" in _text(result)

    @pytest.mark.parametrize("color", UNUSABLE_COLORS)
    def test_a_refused_color_registers_no_object(self, color: Any) -> None:
        """No half-painted object: the registry is untouched.

        Pre-fix ``[]`` reported success and registered the crate with the
        default grey, and ``[0.1]`` registered a 1-component colour that
        ``_add_object_to_builder`` then handed to the solver.
        """
        stub = _newton_stub()
        assert NewtonSimEngine.add_object(stub, "crate", color=color)["status"] == "error"
        assert stub._world.objects == {}

    @pytest.mark.parametrize(("color", "expected"), GOOD_COLORS)
    def test_a_usable_color_is_stored_as_four_plain_floats(self, color: Any, expected: list[float]) -> None:
        stub = _newton_stub()
        assert NewtonSimEngine.add_object(stub, "crate", color=color)["status"] == "success"
        stored = stub._world.objects["crate"].color
        assert stored == pytest.approx(expected)
        assert all(type(component) is float for component in stored)

    def test_an_omitted_color_still_takes_the_documented_default(self) -> None:
        stub = _newton_stub()
        assert NewtonSimEngine.add_object(stub, "crate")["status"] == "success"
        assert stub._world.objects["crate"].color == DEFAULT_GREY

    @pytest.mark.parametrize(("color", "expected"), GOOD_COLORS)
    def test_the_builder_always_gets_three_components(self, color: Any, expected: list[float]) -> None:
        """``tuple(obj.color[:3])`` is well-defined by construction now.

        Pre-fix a stored ``[0.1]`` made that slice a 1-component colour, and the
        solver saw it at rebuild time rather than at the call.
        """
        stub = _newton_stub()
        assert NewtonSimEngine.add_object(stub, "crate", color=color)["status"] == "success"
        assert tuple(stub._world.objects["crate"].color[:3]) == pytest.approx(tuple(expected[:3]))


# --------------------------------------------------------------------------- #
# Isaac                                                                       #
# --------------------------------------------------------------------------- #
class TestIsaacAddObject:
    @pytest.mark.parametrize("color", UNUSABLE_COLORS)
    def test_an_unusable_color_is_refused(self, color: Any) -> None:
        stub, _captured = _isaac_recording_stub()
        result = IsaacSimulation.add_object(stub, "crate", color=color)
        assert result["status"] == "error", (color, result)
        assert "'color'" in _text(result)

    @pytest.mark.parametrize("color", UNUSABLE_COLORS)
    def test_a_refused_color_constructs_no_prim(self, color: Any) -> None:
        """The refusal precedes the prim constructor, so nothing is written."""
        stub, captured = _isaac_recording_stub()
        assert IsaacSimulation.add_object(stub, "crate", color=color)["status"] == "error"
        assert captured == {}
        assert stub._objects == {}

    @pytest.mark.parametrize(("color", "expected"), GOOD_COLORS)
    def test_a_usable_color_reaches_the_prim_as_four_plain_floats(self, color: Any, expected: list[float]) -> None:
        """Which is what makes ``list(color)[:3]`` in the constructor well-defined.

        Pre-fix a 5-component request was silently applied as its first 3, and a
        ``np.float64`` component survived into the prim.
        """
        stub, captured = _isaac_recording_stub()
        assert IsaacSimulation.add_object(stub, "crate", color=color)["status"] == "success"
        assert captured["color"] == pytest.approx(expected)
        assert all(type(component) is float for component in captured["color"])
        assert list(captured["color"])[:3] == pytest.approx(expected[:3])

    def test_an_omitted_color_is_still_forwarded_as_omitted(self) -> None:
        """``None`` keeps meaning "let the prim constructor default it"."""
        stub, captured = _isaac_recording_stub()
        assert IsaacSimulation.add_object(stub, "crate")["status"] == "success"
        assert captured["color"] is None


# --------------------------------------------------------------------------- #
# Cross-backend parity                                                        #
# --------------------------------------------------------------------------- #
class TestEveryBackendGivesTheSameVerdict:
    """A colour one backend refuses is refused by all of them, with one message."""

    @pytest.fixture
    def mj_sim(self) -> Any:
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import Simulation

        sim = Simulation(tool_name="test_color_domain_parity_sim", mesh=False)
        assert sim.create_world()["status"] == "success"
        yield sim
        sim.cleanup()

    @pytest.mark.parametrize("color", UNUSABLE_COLORS)
    def test_add_object_color_verdicts_match(self, mj_sim: Any, color: Any) -> None:
        mj = mj_sim.add_object("crate", color=color)
        nt = NewtonSimEngine.add_object(_newton_stub(), "crate", color=color)
        ic = IsaacSimulation.add_object(_isaac_recording_stub()[0], "crate", color=color)
        assert mj["status"] == nt["status"] == ic["status"] == "error", (color, mj, nt, ic)
        texts = {_text(mj), _text(nt), _text(ic)}
        assert len(texts) == 1, texts

    @pytest.mark.parametrize(("color", "expected"), GOOD_COLORS)
    def test_a_usable_color_is_accepted_everywhere(self, mj_sim: Any, color: Any, expected: list[float]) -> None:
        """The parity is two-way: no backend refuses a colour another honors."""
        assert mj_sim.add_object("crate", color=color)["status"] == "success"
        assert NewtonSimEngine.add_object(_newton_stub(), "crate", color=color)["status"] == "success"
        assert IsaacSimulation.add_object(_isaac_recording_stub()[0], "crate", color=color)["status"] == "success"

    @pytest.mark.parametrize(("color", "expected"), GOOD_COLORS)
    def test_the_same_four_components_are_applied_everywhere(
        self, mj_sim: Any, color: Any, expected: list[float]
    ) -> None:
        """Accepted is not enough - each backend must apply the same colour."""
        import mujoco as mj_module

        assert mj_sim.add_object("crate", shape="box", color=color)["status"] == "success"
        gid = mj_module.mj_name2id(mj_sim._world._model, mj_module.mjtObj.mjOBJ_GEOM, "crate_geom")
        assert gid >= 0
        assert list(mj_sim._world._model.geom_rgba[gid]) == pytest.approx(expected, abs=1e-6)

        nt_stub = _newton_stub()
        NewtonSimEngine.add_object(nt_stub, "crate", color=color)
        assert nt_stub._world.objects["crate"].color == pytest.approx(expected)

        ic_stub, captured = _isaac_recording_stub()
        IsaacSimulation.add_object(ic_stub, "crate", color=color)
        assert captured["color"] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Structural: no colour surface drifts back off the shared domain             #
# --------------------------------------------------------------------------- #
def _backend_dir() -> pathlib.Path:
    """Derive the scan root from a symbol rather than a path literal."""
    return pathlib.Path(inspect.getfile(NewtonSimEngine)).parent.parent


def _public_color_methods(root: pathlib.Path) -> dict[tuple[str, str], bool]:
    """Map ``(module, method)`` to whether it routes ``color`` through the helper.

    Scoped to public methods of a class: ``_construct_shape_prim`` and
    ``_create_shape_prim`` take an already-validated colour from ``add_object``
    and are not caller-facing, so they are excluded by construction rather than
    by a name-based exemption that could go stale.
    """
    found: dict[tuple[str, str], bool] = {}
    for path in sorted(root.glob("*/*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in ast.iter_child_nodes(cls) if isinstance(n, ast.FunctionDef)]:
                if fn.name.startswith("_"):
                    continue
                if "color" not in [a.arg for a in fn.args.args + fn.args.kwonlyargs]:
                    continue
                body = ast.unparse(fn)
                found[(path.parent.name, fn.name)] = "coerce_rgba" in body
    return found


class TestNoColorSurfaceDrifts:
    """A backend cannot ship an ``add_object`` colour off the shared domain."""

    #: Every public backend method that takes a colour today.
    EXPECTED = {
        ("mujoco", "add_object"),
        ("mujoco", "set_geom_properties"),
        ("newton", "add_object"),
        ("isaac", "add_object"),
    }

    def test_every_color_surface_routes_through_the_shared_helper(self) -> None:
        found = _public_color_methods(_backend_dir())
        adrift = sorted(key for key, routed in found.items() if not routed)
        assert not adrift, f"colour parameters not routed through coerce_rgba: {adrift}"

    def test_the_scan_sees_the_surfaces_it_is_meant_to_cover(self) -> None:
        """Non-vacuity: an empty result would otherwise read as a clean sweep."""
        found = _public_color_methods(_backend_dir())
        assert self.EXPECTED <= set(found), f"scan missed {sorted(self.EXPECTED - set(found))}"

    def test_the_scan_flags_a_planted_omission(self, tmp_path: pathlib.Path) -> None:
        """A scanner that silently matched nothing would look like a clean suite."""
        backend = tmp_path / "phony"
        backend.mkdir()
        (backend / "simulation.py").write_text(
            "class PhonySimEngine:\n"
            "    def add_object(self, name, color=None):\n"
            "        return {'status': 'success', 'content': [{'text': color}]}\n",
            encoding="utf-8",
        )
        found = _public_color_methods(tmp_path)
        assert found == {("phony", "add_object"): False}
