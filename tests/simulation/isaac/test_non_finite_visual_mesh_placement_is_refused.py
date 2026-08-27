"""A visual mesh geom whose own placement is not finite is refused, not reported.

``load_mjcf_scene_objects`` reports two placements per object. The collision
proxy - ``position`` and ``size`` - comes from ``_recursive_collision_aabb``.
Where the *visual asset* is hung on that proxy comes from ``_find_body_mesh``,
which returns the ``mesh_pos`` and ``mesh_quat`` a
:class:`~strands_robots.simulation.isaac.loaders.SceneObject` carries.

``_geom_aabb`` refuses a non-finite geom ``pos``, ``size``, orientation or
``fromto`` at the parse. It never sees the geom this reader picks. Two
independent reasons keep it away:

* ``_body_collision_aabb`` walks ``for collidable_only in (True, False)`` and
  **returns as soon as a pass finds a bound**, so on a body that owns a
  collision primitive the second pass never runs and a contact-free geom is
  never handed to ``_geom_aabb`` at all.
* A ``type="mesh"`` geom has no analytic AABB, so even when it *is* handed over
  the answer is ``None`` and it contributes nothing to the bound.

``_find_body_mesh`` prefers precisely the geom the first reason excludes. Its
strongest rank is ``_MESH_VISUAL_RANK_NON_COLLIDING``, because MuJoCo Menagerie
marks a visual geom ``contype="0" conaffinity="0"`` and declares the collision
geom first. So the one geom whose placement is *reported* was the one geom whose
placement was *unchecked*, and the two conditions the leak needs - a contact-free
mesh geom beside a collidable one - are the dominant convention rather than a
corner:

    body shape                          mesh pos    before
    visual-only mesh + collidable box   0 0 nan     ACCEPTED mesh_pos=(0.0, 0.0, nan)
    collidable  mesh + collidable box   0 0 nan     refused
    visual-only mesh, no other geom     0 0 nan     refused
    collidable  mesh, no other geom     0 0 nan     refused

One row of four, and it is the Menagerie one. ``position`` and ``size`` stay
finite and correct in that row, because the accumulator supplies them from the
collision primitive - so the object's physics reads healthy while its visual
asset is hung at a coordinate that is not a coordinate. That is the wrong half to
lose: ``_find_body_mesh`` exists to prefer the visual asset precisely because
"the visual asset is the one a pixel-conditioned policy was trained on".

A single non-finite component is not contained, either. ``_parse_orientation``
normalizes, so ``quat="1 0 nan 0"`` reported ``mesh_quat=(nan, nan, nan, nan)``:
one bad component destroys all four.

Over the 570 MJCF files cached on this machine, 2053 of 3054 bodies are in the
leaking shape - including every link of the ``so101`` the registry ships - and
none carries a non-finite value today. The defect is latent, and 60 of 60
resolvable registry assets load unchanged after the fix.

Deliberately unchanged, and pinned below:

* An attribute that cannot be *parsed* keeps falling back to its documented
  default, so ``pos="garbage"`` still reads as the origin.
* Finite values are untouched, extreme ones included - the guard tests
  finiteness, not magnitude.
* The nested-body ``pos`` this reader folds into its offset is *not* re-guarded
  here. ``_recursive_collision_aabb`` refuses it on the same
  ``findall("body")`` traversal, so a second test would be unreachable; see
  ``test_non_finite_body_placement_is_refused_not_measured_around.py``.
"""

from __future__ import annotations

import ast
import inspect
import math
import struct
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation.isaac import loaders

# ``_ORIENTATION_SPELLINGS`` is the module's own vocabulary; state the members
# locally so a cell grades the module rather than agreeing with it.
ORIENTATION_SPELLINGS = ("quat", "euler", "axisangle", "xyaxes", "zaxis")

# The two conditions the leak needs, spelled the way MuJoCo Menagerie does.
CONTACT_FREE = 'contype="0" conaffinity="0"'


def _binary_stl(path: Path) -> None:
    """Write a well-formed four-vertex binary STL, the smallest MuJoCo accepts."""
    tris = (
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    blob = b"\x00" * 80 + struct.pack("<I", len(tris))
    for tri in tris:
        rec: list[float] = [0.0, 0.0, 0.0]
        for vert in tri:
            rec.extend(vert)
        blob += struct.pack("<12f", *rec) + struct.pack("<H", 0)
    path.write_bytes(blob)


def _scene(
    tmp_path: Path,
    *,
    mesh_attrs: str = CONTACT_FREE,
    mesh_placement: str = 'pos="0 0 0.02"',
    collidable_sibling: bool = True,
    free_joint: bool = False,
) -> Path:
    """An MJCF fixture body: a collision primitive plus a visual mesh shell.

    No free joint by default, because that is the shape that reaches the reader:
    MuJoCo refuses a non-finite geom on a *moving* body (the inertia it derives
    is then not finite) and compiles the same geom on a body without one. A body
    without a free joint is what this loader reports as a fixture.
    """
    _binary_stl(tmp_path / "shell.stl")
    sibling = '<geom name="hull" type="box" size="0.1 0.1 0.1"/>' if collidable_sibling else ""
    joint = "<freejoint/>" if free_joint else ""
    path = tmp_path / "scene.xml"
    path.write_text(
        f"""<mujoco model="probe">
  <asset><mesh name="shell" file="shell.stl"/></asset>
  <worldbody>
    <body name="widget" pos="0 0 0.5">
      {joint}
      {sibling}
      <geom name="shell_visual" type="mesh" mesh="shell" {mesh_attrs} {mesh_placement}/>
    </body>
  </worldbody>
</mujoco>"""
    )
    return path


def _only_object(path: Path) -> Any:
    objects = loaders.load_mjcf_scene_objects(str(path))
    assert len(objects) == 1, objects
    return objects[0]


class TestThePremisesTheFindingRestsOn:
    """These hold before and after the fix; the report is unfounded without them."""

    def test_mujoco_compiles_the_leaking_fixture(self, tmp_path: Path) -> None:
        """The input reaches the reader, so the reader cannot defer the question."""
        mujoco = pytest.importorskip("mujoco")
        path = _scene(tmp_path, mesh_placement='pos="0 0 nan"')
        model = mujoco.MjModel.from_xml_path(str(path))
        assert model.nbody >= 2

    def test_mujoco_refuses_the_same_geom_on_a_moving_body(self, tmp_path: Path) -> None:
        """So only fixtures reach the reader - which is what this loader is for."""
        mujoco = pytest.importorskip("mujoco")
        path = _scene(tmp_path, mesh_placement='pos="0 0 nan"', free_joint=True)
        with pytest.raises(ValueError, match="mass and inertia of moving bodies"):
            mujoco.MjModel.from_xml_path(str(path))

    def test_a_healthy_fixture_compiles_either_way(self, tmp_path: Path) -> None:
        """The refusal above is about the non-finite value, not about the shape."""
        mujoco = pytest.importorskip("mujoco")
        for moving in (False, True):
            path = _scene(tmp_path, free_joint=moving)
            assert mujoco.MjModel.from_xml_path(str(path)).nbody >= 2

    def test_a_contact_free_geom_is_what_the_ranking_prefers(self) -> None:
        """The leak sits on the geom the reader is written to pick."""
        attrs = {"contype": "0", "conaffinity": "0", "mesh": "shell", "type": "mesh"}
        assert loaders._geom_cannot_collide(attrs)
        assert loaders._mesh_geom_visual_rank(attrs) == loaders._MESH_VISUAL_RANK_NON_COLLIDING

    def test_a_mesh_geom_has_no_analytic_aabb(self) -> None:
        """So it contributes nothing to the bound even when it is handed over."""
        import xml.etree.ElementTree as ET

        mesh_geom = ET.fromstring('<geom type="mesh" mesh="shell" pos="0 0 0"/>')
        assert loaders._geom_aabb(mesh_geom, {}, "") is None
        box_geom = ET.fromstring('<geom type="box" size="0.1 0.1 0.1"/>')
        assert loaders._geom_aabb(box_geom, {}, "") is not None

    def test_the_bound_returns_on_its_first_pass_so_the_visual_geom_is_never_offered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mechanism: _geom_aabb is asked about the hull and never the shell."""
        offered: list[str | None] = []
        real = loaders._geom_aabb

        def spy(geom: Any, *args: Any, **kwargs: Any) -> Any:
            offered.append(geom.get("name"))
            return real(geom, *args, **kwargs)

        monkeypatch.setattr(loaders, "_geom_aabb", spy)
        _only_object(_scene(tmp_path))
        assert offered == ["hull"], offered

    def test_the_aabb_invisible_set_is_exactly_the_strongest_ranked_set(self) -> None:
        """Which is why the guard belongs at the parse and not at the return.

        ``_geom_cannot_collide`` is true for precisely the geoms
        ``_mesh_geom_visual_rank`` ranks strongest, and the reader returns on the
        first of those - so the geom the bound cannot vouch for is always the one
        the reader picks, and there is no "checked when it wins" placement that
        would cover a different set.
        """
        candidates = (
            {"mesh": "m", "contype": "0", "conaffinity": "0"},
            {"mesh": "m", "group": "1"},
            {"mesh": "m"},
        )
        for attrs in candidates:
            invisible_to_the_bound = loaders._geom_cannot_collide(attrs)
            strongest_rank = loaders._mesh_geom_visual_rank(attrs) == loaders._MESH_VISUAL_RANK_NON_COLLIDING
            assert invisible_to_the_bound == strongest_rank, attrs

    def test_the_accumulator_never_parses_a_geom_orientation(self) -> None:
        """So the guarded walk cannot answer for this quantity."""
        source = inspect.getsource(loaders._recursive_collision_aabb)
        assert "_parse_orientation" not in source

    def test_the_orientation_spellings_are_the_modules_own_vocabulary(self) -> None:
        assert set(loaders._ORIENTATION_SPELLINGS) == set(ORIENTATION_SPELLINGS)

    def test_one_non_finite_component_poisons_a_whole_quaternion(self) -> None:
        """Which is why the orientation is refused at the parse, not per component."""
        quat = loaders._parse_orientation({"quat": "1 0 nan 0"}, own={"quat": "1 0 nan 0"})
        assert all(not math.isfinite(component) for component in quat), quat


class TestANonFiniteVisualMeshPlacementIsRefused:
    """The regression: every spelling refuses, naming the geom and the attribute."""

    @pytest.mark.parametrize("component", ["nan", "inf", "-inf"])
    def test_a_non_finite_position_is_refused(self, tmp_path: Path, component: str) -> None:
        with pytest.raises(ValueError, match="pos has a component that is not finite"):
            _only_object(_scene(tmp_path, mesh_placement=f'pos="0 0 {component}"'))

    @pytest.mark.parametrize(
        "spelling,declaration",
        [
            ("quat", 'quat="1 0 nan 0"'),
            ("euler", 'euler="nan 0 0"'),
            ("axisangle", 'axisangle="0 0 1 nan"'),
            ("zaxis", 'zaxis="0 nan 1"'),
            ("xyaxes", 'xyaxes="1 0 0 0 nan 0"'),
        ],
    )
    def test_a_non_finite_orientation_is_refused_by_its_own_spelling(
        self, tmp_path: Path, spelling: str, declaration: str
    ) -> None:
        """The refusal names the attribute a reader can go and look at."""
        with pytest.raises(ValueError, match=f"{spelling} has a component that is not finite"):
            _only_object(_scene(tmp_path, mesh_placement=declaration))

    def test_the_refusal_names_the_geom_rather_than_the_body(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError) as excinfo:
            _only_object(_scene(tmp_path, mesh_placement='pos="0 0 nan"'))
        message = str(excinfo.value)
        assert "geom 'shell_visual'" in message
        assert "body 'widget'" not in message

    def test_an_unnamed_visual_geom_is_located_by_its_resolved_type(self, tmp_path: Path) -> None:
        """MJCF geoms are frequently unnamed, so the locator cannot rely on a name."""
        _binary_stl(tmp_path / "shell.stl")
        path = tmp_path / "scene.xml"
        path.write_text(
            f"""<mujoco model="probe">
  <asset><mesh name="shell" file="shell.stl"/></asset>
  <worldbody>
    <body name="widget" pos="0 0 0.5">
      <geom name="hull" type="box" size="0.1 0.1 0.1"/>
      <geom type="mesh" mesh="shell" {CONTACT_FREE} pos="0 0 nan"/>
    </body>
  </worldbody>
</mujoco>"""
        )
        with pytest.raises(ValueError, match=r'unnamed <geom type="mesh">'):
            _only_object(path)

    def test_the_object_no_longer_reports_a_placement_that_is_not_a_placement(self, tmp_path: Path) -> None:
        """The harm, stated as the reader's own output: a reported non-finite field."""
        healthy = _only_object(_scene(tmp_path))
        assert all(math.isfinite(value) for value in healthy.mesh_pos)
        with pytest.raises(ValueError):
            _only_object(_scene(tmp_path, mesh_placement='pos="0 0 nan"'))

    def test_a_geom_that_inherits_the_bad_value_from_a_default_class_is_refused(self, tmp_path: Path) -> None:
        """The quantities are read through _class_attrs, so the guard must be too."""
        _binary_stl(tmp_path / "shell.stl")
        path = tmp_path / "scene.xml"
        path.write_text(
            f"""<mujoco model="probe">
  <default><default class="visual"><geom {CONTACT_FREE} pos="0 0 nan"/></default></default>
  <asset><mesh name="shell" file="shell.stl"/></asset>
  <worldbody>
    <body name="widget" pos="0 0 0.5">
      <geom name="hull" type="box" size="0.1 0.1 0.1"/>
      <geom name="shell_visual" class="visual" type="mesh" mesh="shell"/>
    </body>
  </worldbody>
</mujoco>"""
        )
        with pytest.raises(ValueError, match="pos has a component that is not finite"):
            _only_object(path)


class TestTheInputsTheGeomGuardAlreadyReached:
    """Controls. Each of these passes before and after the fix.

    Three of the four body shapes, plus a collidable loser beside a good winner:
    in every one of them ``_geom_aabb`` is offered the offending geom and its own
    guard answers. Only the fourth shape - a contact-free mesh geom beside a
    collidable sibling - reaches nothing.
    """

    def test_a_bad_geom_that_loses_the_ranking_is_still_refused(self, tmp_path: Path) -> None:
        """A *collidable* bad geom beside a good visual one: the bound refuses it.

        Passes before and after this change. The loser is collidable, so
        ``_body_collision_aabb``'s first pass does offer it to ``_geom_aabb``,
        whose own guard refuses it. Recorded because it is the shape a reader
        would reach for to argue the leak, and it is not the leak.
        """
        _binary_stl(tmp_path / "shell.stl")
        path = tmp_path / "scene.xml"
        path.write_text(
            f"""<mujoco model="probe">
  <asset><mesh name="shell" file="shell.stl"/><mesh name="pretty" file="shell.stl"/></asset>
  <worldbody>
    <body name="widget" pos="0 0 0.5">
      <geom name="hull" type="box" size="0.1 0.1 0.1"/>
      <geom name="loser" type="mesh" mesh="shell" pos="0 0 nan"/>
      <geom name="winner" type="mesh" mesh="pretty" {CONTACT_FREE} pos="0 0 0.02"/>
    </body>
  </worldbody>
</mujoco>"""
        )
        with pytest.raises(ValueError, match="geom 'loser'"):
            _only_object(path)

    def test_a_collidable_mesh_beside_a_collidable_sibling(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not finite"):
            _only_object(_scene(tmp_path, mesh_attrs="", mesh_placement='pos="0 0 nan"'))

    def test_a_contact_free_mesh_that_is_the_bodys_only_geom(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not finite"):
            _only_object(_scene(tmp_path, mesh_placement='pos="0 0 nan"', collidable_sibling=False))

    def test_a_collidable_mesh_that_is_the_bodys_only_geom(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not finite"):
            _only_object(_scene(tmp_path, mesh_attrs="", mesh_placement='pos="0 0 nan"', collidable_sibling=False))


class TestWhatIsDeliberatelyUnchanged:
    """A finite fixture, and an unreadable attribute, behave exactly as before."""

    def test_a_healthy_fixture_reports_exactly_what_it_did(self, tmp_path: Path) -> None:
        obj = _only_object(_scene(tmp_path))
        assert obj.mesh_pos == (0.0, 0.0, 0.02)
        assert obj.mesh_quat == (1.0, 0.0, 0.0, 0.0)
        assert obj.position == (0.0, 0.0, 0.5)
        assert obj.size == (0.2, 0.2, 0.2)
        assert obj.is_static is True

    def test_a_healthy_orientation_is_still_normalized_and_kept(self, tmp_path: Path) -> None:
        obj = _only_object(_scene(tmp_path, mesh_placement='quat="1 0 0.2 0"'))
        assert all(math.isfinite(component) for component in obj.mesh_quat)
        assert math.isclose(sum(component**2 for component in obj.mesh_quat), 1.0, rel_tol=1e-9)

    def test_an_unparseable_placement_still_falls_back_to_its_default(self, tmp_path: Path) -> None:
        """Only a value that parsed and is non-finite is refused."""
        obj = _only_object(_scene(tmp_path, mesh_placement='pos="garbage"'))
        assert obj.mesh_pos == (0.0, 0.0, 0.0)

    def test_an_extreme_but_finite_placement_is_accepted(self, tmp_path: Path) -> None:
        """The guard tests finiteness, not magnitude."""
        obj = _only_object(_scene(tmp_path, mesh_placement='pos="0 0 1e30"'))
        assert obj.mesh_pos == (0.0, 0.0, 1e30)

    def test_the_collision_proxy_is_untouched_by_the_guard(self, tmp_path: Path) -> None:
        """The bound comes from the collidable sibling either way."""
        obj = _only_object(_scene(tmp_path, mesh_placement='pos="0 0 0.02"'))
        assert obj.size == (0.2, 0.2, 0.2)

    def test_a_body_with_no_mesh_geom_still_reports_the_default_mesh_placement(self, tmp_path: Path) -> None:
        path = tmp_path / "scene.xml"
        path.write_text(
            """<mujoco model="probe"><worldbody>
  <body name="widget" pos="0 0 0.5">
    <geom name="hull" type="box" size="0.1 0.1 0.1"/>
  </body></worldbody></mujoco>"""
        )
        obj = _only_object(path)
        assert obj.mesh_pos == (0.0, 0.0, 0.0)
        assert obj.mesh_quat == (1.0, 0.0, 0.0, 0.0)


class TestOneOwnerForTheWording:
    """The new refusals go through the module's single finiteness owner."""

    def test_the_reader_delegates_to_the_shared_geom_locator(self) -> None:
        source = inspect.getsource(loaders._find_body_mesh)
        tree = ast.parse(source.strip())
        called = {
            node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "_refuse_non_finite_geom" in called
        assert not any(name.startswith("_refuse_non_finite_placement") for name in called)

    def test_the_reader_raises_no_wording_of_its_own(self) -> None:
        source = inspect.getsource(loaders._find_body_mesh)
        raises = [node for node in ast.walk(ast.parse(source.strip())) if isinstance(node, ast.Raise)]
        assert raises == []

    def test_the_guard_precedes_the_offset_it_would_poison(self) -> None:
        """A guard after the composition would report the sum, not the attribute."""
        source = inspect.getsource(loaders._find_body_mesh)
        guard = source.index('_refuse_non_finite_geom(attrs, "pos", gpos)')
        composition = source.index("pos = (offset[0] + gpos[0]")
        assert guard < composition
