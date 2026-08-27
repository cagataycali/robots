"""A geom whose placement or extent is not finite is refused, not measured around.

``_geom_aabb`` folds each of a body's geoms into the axis-aligned box
``load_mjcf_scene_objects`` reports as a
:class:`~strands_robots.simulation.isaac.loaders.SceneObject`'s ``size`` and
``position`` -- the collision proxy the Isaac realization stands the object up
with. ``_body_collision_aabb`` unions those with a running ``min``/``max``, and
Python orders a NaN as neither smaller nor larger than anything, so a comparison
against one keeps the accumulator it started with. The geom carrying it
disappears from the bound and nothing reports that.

The consequence is not a small error. A static fixture whose leg declares
``euler="nan 0 0"`` was measured as its tabletop alone::

    kitchen_table with a healthy leg   -> size (0.8, 0.6, 0.77)
    kitchen_table, leg euler="nan 0 0" -> size (0.8, 0.6, 0.04)
    kitchen_table with no leg at all   -> size (0.8, 0.6, 0.04)

-- a 4 cm slab where the file declares a 77 cm table, and byte-identical to the
same fixture with the leg deleted, under ``status`` success. Every flavour is
reachable and each is wrong differently: ``pos="nan ..."`` drops the geom on the
axis that is not finite and keeps it on the others; ``pos="inf ..."`` reports an
infinite centre and a NaN extent; ``size="inf ..."`` reports a NaN centre and,
because ``inf - inf`` is a NaN that the *outer* accumulator in
``_recursive_collision_aabb`` drops in turn, sizes that axis at the ``1e-4``
floor -- the smallest proxy the reader can emit, for the largest geom the file
can declare.

Reachability is the part worth being precise about. MuJoCo refuses a non-finite
geom on a body with a free joint, because the inertia it derives is then not
finite (``mass and inertia of moving bodies must be larger than mjMINVAL``). It
*compiles* the same geom on a body without one -- and a body without a free
joint is exactly what this loader calls a fixture, the tables and cabinets whose
footprint a manipulation scene is planned against. So the scenes that reach the
reader are precisely the ones that load, which is why the reader cannot defer the
question to a compile step.

MuJoCo is the oracle for the disposition as well as for reachability. It refuses
a non-finite geom quantity wherever it checks one -- ``fromto="0 0 0 nan 0 0.5"``
is rejected as ``nan size in geom`` -- and warns about the document as a whole
(``XML contains a 'NaN'``). Refusing is the format's own answer; measuring around
it silently is not an answer either half of MuJoCo gives. It is also what this
module's stated failure semantics already promise: *"Loaders never silently
return a phantom robot."*

The same defect class was closed one module over for mesh assets in #2740, whose
``_non_finite_vertex_error`` describes this mechanism in the same words -- bounds
"the same numbers a mesh declaring only those would produce, under no error at
all". ``loaders.py`` carried no finiteness guard at all.

Deliberately unchanged, and pinned below:

* An attribute that cannot be *parsed* keeps falling back to its documented
  default. Only a value that parsed and is non-finite is refused, so
  ``pos="garbage"`` still reads as the origin exactly as before.
* A geom with no analytic AABB (mesh, plane) still returns ``None`` so the
  caller falls back to another geom.
* Finite values are untouched, including extreme ones -- the guard tests
  finiteness, not magnitude.

Over the shipped corpus -- 542 MJCF documents, 14940 ``<geom>`` elements under
``robot_descriptions`` and this package's own assets -- zero declare a non-finite
placement or extent, so no shipped asset changes behaviour. The fix is latent by
that measurement and the measurement is also the regression proof.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from strands_robots.simulation.isaac.loaders import (
    _ORIENTATION_SPELLINGS,
    _body_collision_aabb,
    _geom_aabb,
    _mjcf_class_defaults,
    _refuse_non_finite_geom,
    load_mjcf_scene_objects,
)

mujoco = pytest.importorskip("mujoco")

#: A static fixture: a tabletop plus one leg, no free joint. The tabletop alone
#: spans z 0.73..0.77 and the leg carries the fixture down to the floor, so
#: losing the leg collapses the reported height from 0.77 m to 0.04 m.
FIXTURE = """<mujoco model="t">
  <worldbody>
    <body name="floor"><geom name="fl" type="plane" size="2 2 0.1"/></body>
    <body name="kitchen_table" pos="0.4 0.1 0.0">
      <geom name="t_top" type="box" size="0.40 0.30 0.02" pos="0 0 0.75"/>
      {LEG}
    </body>
  </worldbody>
</mujoco>"""

HEALTHY_LEG = '<geom name="t_leg" type="box" size="0.03 0.03 0.375" pos="0.35 0.25 0.375"/>'

#: One spelling per row, each stating the attribute the refusal must name. Every
#: one of these compiles in MuJoCo on this fixture -- asserted as a premise
#: rather than assumed, because the whole finding rests on it.
NON_FINITE_LEGS: list[tuple[str, str, str]] = [
    ("pos-nan", "pos", '<geom name="t_leg" type="box" size="0.03 0.03 0.375" pos="nan 0.25 0.375"/>'),
    ("pos-inf", "pos", '<geom name="t_leg" type="box" size="0.03 0.03 0.375" pos="inf 0.25 0.375"/>'),
    ("size-inf", "size", '<geom name="t_leg" type="box" size="inf 0.03 0.375" pos="0.35 0.25 0.375"/>'),
    (
        "euler-nan",
        "euler",
        '<geom name="t_leg" type="box" size="0.03 0.03 0.375" pos="0.35 0.25 0.375" euler="nan 0 0"/>',
    ),
    (
        "quat-nan",
        "quat",
        '<geom name="t_leg" type="box" size="0.03 0.03 0.375" pos="0.35 0.25 0.375" quat="nan 0 0 0"/>',
    ),
    (
        "axisangle-nan",
        "axisangle",
        '<geom name="t_leg" type="box" size="0.03 0.03 0.375" pos="0.35 0.25 0.375" axisangle="0 0 1 nan"/>',
    ),
    (
        "zaxis-nan",
        "zaxis",
        '<geom name="t_leg" type="box" size="0.03 0.03 0.375" pos="0.35 0.25 0.375" zaxis="nan 0 1"/>',
    ),
    (
        "xyaxes-nan",
        "xyaxes",
        '<geom name="t_leg" type="box" size="0.03 0.03 0.375" pos="0.35 0.25 0.375" xyaxes="nan 0 0 0 1 0"/>',
    ),
    (
        "fromto-inf",
        "fromto",
        '<geom name="t_leg" type="capsule" size="0.03" fromto="0.35 0.25 0 inf 0.25 0.75"/>',
    ),
]

_IDS = [row[0] for row in NON_FINITE_LEGS]


def _scene(leg: str) -> str:
    return FIXTURE.format(LEG=leg)


def _write(tmp_path: Path, leg: str) -> str:
    path = tmp_path / "scene.xml"
    path.write_text(_scene(leg))
    return str(path)


def _table(tmp_path: Path, leg: str):
    """The loader's ``SceneObject`` for the fixture body."""
    objects = load_mjcf_scene_objects(_write(tmp_path, leg))
    return next(obj for obj in objects if obj.name == "kitchen_table")


def _bound(leg: str):
    """``_body_collision_aabb`` for the fixture body, defaults resolved."""
    root = ET.fromstring(_scene(leg))
    body = root.find(".//body[@name='kitchen_table']")
    assert body is not None
    return _body_collision_aabb(body, _mjcf_class_defaults(root, ".", "geom"), "")


class TestThePremisesTheFindingRestsOn:
    """These hold before and after the fix; the report is unfounded without them."""

    @pytest.mark.parametrize(("_label", "_attribute", "leg"), NON_FINITE_LEGS, ids=_IDS)
    def test_mujoco_compiles_the_fixture(self, tmp_path: Path, _label: str, _attribute: str, leg: str) -> None:
        # A non-finite geom on a body with no free joint is a model MuJoCo
        # accepts, so the reader really does meet these scenes.
        model = mujoco.MjModel.from_xml_path(_write(tmp_path, leg))
        assert model.ngeom == 3

    def test_mujoco_refuses_the_same_class_where_it_checks_one(self, tmp_path: Path) -> None:
        # The format's own disposition: refuse, not measure around. A NaN in a
        # ``fromto`` reaches MuJoCo's size derivation and is rejected there.
        leg = '<geom name="t_leg" type="capsule" size="0.03" fromto="0.35 0.25 0 nan 0.25 0.75"/>'
        with pytest.raises(ValueError, match="nan size in geom"):
            mujoco.MjModel.from_xml_path(_write(tmp_path, leg))

    def test_a_moving_body_is_refused_by_mujoco_so_only_fixtures_reach_the_reader(self, tmp_path: Path) -> None:
        # Scoping the claim: with a free joint the derived inertia is not
        # finite and MuJoCo refuses, which is why the finding is about fixtures.
        leg = '<geom name="t_leg" type="box" size="0.03 0.03 0.375" pos="nan 0.25 0.375"/>'
        scene = _scene(leg).replace('<geom name="t_top"', '<freejoint/><geom name="t_top"')
        path = tmp_path / "moving.xml"
        path.write_text(scene)
        with pytest.raises(ValueError, match="mass and inertia"):
            mujoco.MjModel.from_xml_path(str(path))

    def test_a_running_min_max_drops_a_nan_rather_than_propagating_it(self) -> None:
        # The mechanism itself, so a reader need not take it on trust.
        assert min(float("inf"), float("nan")) == float("inf")
        assert max(float("-inf"), float("nan")) == float("-inf")
        # And the outer accumulator's input: ``inf - inf`` is a NaN, which is
        # why an infinite extent lands at the floor rather than unbounded.
        assert math.isnan(float("inf") - float("inf"))

    def test_the_orientation_spellings_are_the_modules_own_vocabulary(self) -> None:
        # The refusal names the attribute the file used, taken from the list the
        # reader already resolves orientation through rather than a second copy.
        named = {attribute for _, attribute, _ in NON_FINITE_LEGS}
        assert set(_ORIENTATION_SPELLINGS) <= named | {"quat"}
        assert "quat" in _ORIENTATION_SPELLINGS


class TestANonFiniteGeomIsRefusedByName:
    """The regression: every spelling refuses, naming the attribute at fault."""

    @pytest.mark.parametrize(("_label", "attribute", "leg"), NON_FINITE_LEGS, ids=_IDS)
    def test_the_public_loader_refuses(self, tmp_path: Path, _label: str, attribute: str, leg: str) -> None:
        with pytest.raises(ValueError, match="is not finite") as excinfo:
            load_mjcf_scene_objects(_write(tmp_path, leg))
        message = str(excinfo.value)
        assert attribute in message, message
        # The locator, so a reader can find the geom in a scene of hundreds.
        assert "'t_leg'" in message, message

    @pytest.mark.parametrize(("_label", "_attribute", "leg"), NON_FINITE_LEGS, ids=_IDS)
    def test_the_body_bound_refuses_too(self, _label: str, _attribute: str, leg: str) -> None:
        # Refused at the geom parse, so every consumer of the bound inherits it
        # rather than each having to guard separately.
        with pytest.raises(ValueError, match="is not finite"):
            _bound(leg)

    def test_the_fixture_is_no_longer_measured_as_its_tabletop_alone(self, tmp_path: Path) -> None:
        # The headline. Pre-fix this reported the leg-absent bound exactly.
        absent = _table(tmp_path, "")
        assert absent.size == pytest.approx((0.8, 0.6, 0.04), abs=1e-9)
        healthy = _table(tmp_path, HEALTHY_LEG)
        assert healthy.size == pytest.approx((0.8, 0.6, 0.77), abs=1e-9)
        # ... and the two differ, so "same as absent" is a real accusation.
        assert healthy.size != pytest.approx(absent.size, abs=1e-9)
        nan_leg = dict((label, leg) for label, _, leg in NON_FINITE_LEGS)["euler-nan"]
        with pytest.raises(ValueError, match="euler has a component that is not finite"):
            _table(tmp_path, nan_leg)

    def test_an_infinite_extent_is_not_reported_at_the_floor(self, tmp_path: Path) -> None:
        # The inverted case: the widest geom the file can declare had sized the
        # proxy at 1e-4 on that axis, the narrowest the reader can emit.
        size_inf = dict((label, leg) for label, _, leg in NON_FINITE_LEGS)["size-inf"]
        with pytest.raises(ValueError, match="size has a component that is not finite"):
            _table(tmp_path, size_inf)

    def test_the_refusal_says_what_the_alternative_would_have_been(self, tmp_path: Path) -> None:
        # A refusal a caller cannot act on is half a refusal, so the message
        # states the consequence being avoided rather than only the fact.
        with pytest.raises(ValueError) as excinfo:
            _table(tmp_path, dict((label, leg) for label, _, leg in NON_FINITE_LEGS)["pos-nan"])
        assert "does not declare" in str(excinfo.value)


class TestWhatIsDeliberatelyUnchanged:
    """Controls. Each of these passes before and after the fix."""

    def test_a_healthy_fixture_reports_exactly_what_it_did(self, tmp_path: Path) -> None:
        table = _table(tmp_path, HEALTHY_LEG)
        assert table.position == pytest.approx((0.4, 0.1, 0.385), abs=1e-9)
        assert table.size == pytest.approx((0.8, 0.6, 0.77), abs=1e-9)
        assert table.is_static is True

    def test_an_unparseable_attribute_still_falls_back_to_its_default(self) -> None:
        # ``_parse_xyz`` tolerates a value it cannot read, and that is the
        # documented behaviour. Only a value that *parsed* is graded, so this
        # keeps reading as the body origin rather than becoming a refusal.
        leg = '<geom name="t_leg" type="box" size="0.03 0.03 0.375" pos="garbage"/>'
        centre, size = _bound(leg)
        assert all(math.isfinite(v) for v in (*centre, *size))
        # The leg reads at the body origin, spanning z -0.375..0.375, and the
        # top spans 0.73..0.77, so the union is 1.145 m tall.
        assert size == pytest.approx((0.8, 0.6, 1.145), abs=1e-9)

    def test_a_geom_with_no_analytic_aabb_still_returns_none(self) -> None:
        # ``None`` means "fall back to another geom" and must not be confused
        # with a refusal; a mesh geom keeps that answer.
        root = ET.fromstring(_scene('<geom name="t_leg" type="mesh" mesh="m"/>'))
        body = root.find(".//body[@name='kitchen_table']")
        assert body is not None
        mesh_geom = body.findall("geom")[1]
        assert _geom_aabb(mesh_geom, _mjcf_class_defaults(root, ".", "geom"), "") is None

    def test_an_extreme_but_finite_extent_is_accepted(self) -> None:
        # The guard tests finiteness, not magnitude: a implausibly large but
        # representable value is a measurement, and stays one.
        leg = '<geom name="t_leg" type="box" size="1e30 0.03 0.375" pos="0.35 0.25 0.375"/>'
        _centre, size = _bound(leg)
        assert size[0] == pytest.approx(2e30, rel=1e-12)

    def test_the_helper_is_silent_on_finite_input(self) -> None:
        attrs = {"name": "g", "type": "box"}
        # Finite input raises nothing, and an empty tuple - a geom spelling no
        # size at all - is vacuously finite rather than a refusal. That the
        # helper returns nothing is mypy's to state, not a runtime assertion's.
        _refuse_non_finite_geom(attrs, "pos", (0.0, -1.5, 2.0))
        _refuse_non_finite_geom(attrs, "size", ())


class TestTheRefusalLocatesAnUnnamedGeom:
    """MJCF geoms are frequently unnamed, so the locator cannot rely on a name."""

    def test_an_unnamed_geom_is_located_by_its_resolved_type(self, tmp_path: Path) -> None:
        leg = '<geom type="box" size="0.03 0.03 0.375" pos="nan 0.25 0.375"/>'
        with pytest.raises(ValueError, match=r'unnamed <geom type="box">'):
            _table(tmp_path, leg)

    def test_the_type_comes_from_default_inheritance_not_the_element(self) -> None:
        # A geom need not spell its own type; the locator must still be right.
        scene = """<mujoco>
          <default><default class="leg"><geom type="ellipsoid"/></default></default>
          <worldbody><body name="kitchen_table">
            <geom class="leg" size="0.1 0.2 0.3" pos="nan 0 0"/>
          </body></worldbody>
        </mujoco>"""
        root = ET.fromstring(scene)
        body = root.find(".//body")
        assert body is not None
        with pytest.raises(ValueError, match=r'unnamed <geom type="ellipsoid">'):
            _body_collision_aabb(body, _mjcf_class_defaults(root, ".", "geom"), "")


class TestOneOwnerForTheWording:
    """One guard for every parse path, so no spelling drifts from the rest.

    Four geom paths (``pos``, ``size``, the orientation spelling, ``fromto``)
    and the body placements the same bound is composed from.
    """

    def test_every_finiteness_refusal_in_the_module_is_the_shared_one(self) -> None:
        import ast
        import inspect

        from strands_robots.simulation.isaac import loaders

        source = inspect.getsource(loaders)
        owners: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.FunctionDef):
                continue
            # ``math.isfinite`` is passed to ``map`` rather than called, so the
            # reference is an attribute load, not a callee. Grade any mention.
            for inner in ast.walk(node):
                if isinstance(inner, ast.Attribute) and inner.attr == "isfinite":
                    owners.add(node.name)
        assert owners, "the scan found no finiteness test at all - it is looking in the wrong place"
        assert owners == {"_refuse_non_finite_placement"}, owners

    def test_every_locator_delegates_to_that_one_owner(self) -> None:
        # Geoms and bodies are located differently but share the wording, so
        # each locator must reach the single owner rather than raise its own.
        import ast
        import inspect

        from strands_robots.simulation.isaac import loaders

        for wrapper in ("_refuse_non_finite_geom", "_refuse_non_finite_body"):
            tree = ast.parse(inspect.getsource(getattr(loaders, wrapper)).strip())
            called = {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            assert "_refuse_non_finite_placement" in called, (wrapper, called)
            raises = [n for n in ast.walk(tree) if isinstance(n, ast.Raise)]
            assert raises == [], (wrapper, "a locator must not carry its own wording")

    def test_the_guard_is_reached_from_every_parse_path(self) -> None:
        import ast
        import inspect

        from strands_robots.simulation.isaac.loaders import _geom_aabb as target

        calls = [
            ast.unparse(node.args[1])
            for node in ast.walk(ast.parse(inspect.getsource(target).strip()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_refuse_non_finite_geom"
        ]
        assert "'pos'" in calls and "'size'" in calls and "'fromto'" in calls
        # The orientation call names the spelling the file used rather than a
        # literal, which is what lets one call cover five attributes.
        assert "spelling" in calls
        assert len(calls) == 4, calls
