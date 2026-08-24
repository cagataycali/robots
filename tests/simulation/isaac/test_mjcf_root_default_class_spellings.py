"""MJCF's root default class is one class under two spellings, and both reach it.

A model's top-level ``<default>`` may be written unnamed or as
``<default class="main">``, and those are the same class: MuJoCo names the root
``main`` and refuses to let a model rename it ("top-level default class 'main'
cannot be renamed"), so those two spellings are the only ones a file has. A geom
reaches that class by the same two names - its own ``class="main"``, an enclosing
body's ``childclass="main"`` - or by naming no class at all.

Keyed on whichever spelling the file happened to use, the resolver loses every
geom that arrives by the other one. In one direction that is the gap noted when
:func:`_mjcf_class_defaults` landed: an unnamed root plus ``<geom class="main"/>``.
In the other it is the whole model, and it is what shipped assets actually do -
Menagerie's ``pal_tiago_dual`` writes ``<default class="main">`` declaring
``type="mesh" group="1"`` and gives none of its 46 geoms a ``class``, so 34 of
them (the ones carrying no ``type`` or ``fromto`` of their own) reported the
0.05 m fallback box for a mesh, under ``load_mjcf`` reporting success.

MuJoCo is the oracle throughout: every expected shape is read back off a compiled
``MjModel`` rather than restated. The refusals that make aliasing safe - a nested
class cannot be named ``main``, and cannot be unnamed - are asserted here too, so
the justification is pinned rather than remembered.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from strands_robots.simulation.isaac import loaders as mod
from strands_robots.simulation.isaac.loaders import load_mjcf, load_mjcf_scene_objects

mujoco = pytest.importorskip("mujoco")

_GEOM_TYPE_NAMES = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}


def _write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _compiled_geom(path, geom_name):
    """MuJoCo's own ``(type, size)`` for a named geom - the oracle."""
    model = mujoco.MjModel.from_xml_path(str(path))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    assert gid >= 0, f"premise: MuJoCo compiled no geom named {geom_name!r}"
    # int(): a pybind11 enum does not match a numpy integer as a dict key.
    return _GEOM_TYPE_NAMES[int(model.geom_type[gid])], tuple(float(x) for x in model.geom_size[gid])


def _body(path, name):
    return next(b for b in load_mjcf(str(path)).bodies if b.name == name)


# The root class carries the shape; the geom carries nothing but its name. The
# two files differ only in whether the root ``<default>`` spells its own name,
# which MJCF says is the same class either way.
_ROOT_SHAPE = '<geom type="capsule" size="0.06 0.21"/>'
_UNNAMED_ROOT = f"""<mujoco model="m">
  <default>{_ROOT_SHAPE}</default>
  <worldbody>
    <body name="seg" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="seg_g" class="main"/>
    </body>
  </worldbody>
</mujoco>
"""
_NAMED_ROOT = f"""<mujoco model="m">
  <default class="main">{_ROOT_SHAPE}</default>
  <worldbody>
    <body name="seg" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="seg_g"/>
    </body>
  </worldbody>
</mujoco>
"""


class TestTheRootClassAnswersToBothOfItsSpellings:
    """The regression: the root class is reached by either name, and by neither."""

    def test_a_named_root_supplies_a_geom_that_names_no_class(self, tmp_path):
        """The shipped-asset case: every geom in the model omits ``class``."""
        path = _write(tmp_path, "named.xml", _NAMED_ROOT)
        true_type, true_size = _compiled_geom(path, "seg_g")
        assert true_type == "capsule", "premise: MuJoCo reads the named root class"

        seg = _body(path, "seg")
        assert seg.shape == true_type, (
            f"the root class is spelled class='main' and the geom names no class, so the loader "
            f"reported {seg.shape!r} with size {seg.shape_size} - the fallback box - for a capsule"
        )
        # MuJoCo's capsule size is (radius, half-length).
        assert seg.shape_size == pytest.approx((true_size[0], true_size[1]), abs=1e-9)

    def test_an_unnamed_root_supplies_a_geom_that_names_it_main(self, tmp_path):
        """The other direction: the file omits the name, the geom spells it."""
        path = _write(tmp_path, "unnamed.xml", _UNNAMED_ROOT)
        true_type, true_size = _compiled_geom(path, "seg_g")
        assert true_type == "capsule", "premise: MuJoCo resolves class='main' to the unnamed root"

        seg = _body(path, "seg")
        assert seg.shape == true_type, (
            f"class='main' names the unnamed root <default>, and the loader reported "
            f"{seg.shape!r} with size {seg.shape_size} instead"
        )
        assert seg.shape_size == pytest.approx((true_size[0], true_size[1]), abs=1e-9)

    def test_both_spellings_are_read_as_the_same_model(self, tmp_path):
        """Neither file is privileged: they describe one model, so they read alike.

        Asserted against MuJoCo rather than only against each other - two
        readings that are both the fallback box are also equal, so agreement
        alone is satisfied by the bug this file pins.
        """
        a_path = _write(tmp_path, "a.xml", _UNNAMED_ROOT)
        b_path = _write(tmp_path, "b.xml", _NAMED_ROOT)
        oracle = _compiled_geom(a_path, "seg_g")
        assert oracle == _compiled_geom(b_path, "seg_g"), "premise: MuJoCo reads one model from either spelling"

        unnamed = _body(a_path, "seg")
        named = _body(b_path, "seg")
        assert (unnamed.shape, unnamed.shape_size) == (named.shape, named.shape_size)
        assert unnamed.shape == oracle[0]
        assert unnamed.shape_size == pytest.approx((oracle[1][0], oracle[1][1]), abs=1e-9)

    def test_a_named_root_reaches_a_subtree_through_childclass(self, tmp_path):
        path = _write(
            tmp_path,
            "cc.xml",
            """<mujoco model="m">
  <default class="main"><geom type="cylinder" size="0.02 0.11"/></default>
  <worldbody>
    <body name="upper" childclass="main" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom name="upper_g"/>
      <body name="lower" pos="0 0 -0.3">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom name="lower_g"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
        )
        assert _compiled_geom(path, "lower_g")[0] == "cylinder", "premise"
        for name in ("upper", "lower"):
            body = _body(path, name)
            assert body.shape == "cylinder", f"{name} did not reach the root class by childclass ({body.shape})"
            assert body.shape_size == pytest.approx((0.02, 0.11), abs=1e-9)

    def test_a_nested_class_under_a_named_root_still_inherits_it(self, tmp_path):
        """Aliasing the root must not detach the tree hanging off it."""
        path = _write(
            tmp_path,
            "nested.xml",
            """<mujoco model="m">
  <default class="main"><geom type="capsule"/>
    <default class="leg"><geom size="0.03 0.09"/></default>
  </default>
  <worldbody>
    <body name="seg" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="seg_g" class="leg"/>
    </body>
  </worldbody>
</mujoco>
""",
        )
        assert _compiled_geom(path, "seg_g")[0] == "capsule", "premise"
        seg = _body(path, "seg")
        assert seg.shape == "capsule", f"'leg' did not inherit type from the named root ({seg.shape})"
        assert seg.shape_size == pytest.approx((0.03, 0.09), abs=1e-9)

    def test_the_scene_object_reader_reads_a_named_root_too(self, tmp_path):
        """The second caller of the resolver, on the same model shape."""
        path = _write(
            tmp_path,
            "scene.xml",
            """<mujoco model="m">
  <default class="main"><geom type="box" size="0.30 0.20 0.02"/></default>
  <worldbody>
    <body name="shelf" pos="0 0 0.5"><geom name="shelf_g"/></body>
  </worldbody>
</mujoco>
""",
        )
        assert _compiled_geom(path, "shelf_g")[0] == "box", "premise"
        shelf = next(o for o in load_mjcf_scene_objects(str(path)) if o.name == "shelf")
        # 0.60 x 0.40 x 0.04 full extent, not the mesh-less fallback.
        assert shelf.size == pytest.approx((0.60, 0.40, 0.04), abs=1e-6), (
            f"the proxy for a named-root box came back {shelf.size}"
        )


class TestTheResolverPublishesBothSpellings:
    """The mapping itself, so the alias is pinned where it is made."""

    @pytest.mark.parametrize("root_open", ["<default>", '<default class="main">'])
    def test_the_root_class_is_registered_under_both_names(self, root_open):
        xml = f'<mujoco model="m">{root_open}<geom type="sphere" size="0.04"/></default></mujoco>'
        defaults = mod._mjcf_class_defaults(ET.fromstring(xml), ".", "geom")
        assert defaults[""] == defaults["main"], (
            f'the root class differs by spelling: "" -> {defaults[""]}, "main" -> {defaults["main"]}'
        )
        # Non-vacuity: both keys carry the root's attributes, not two empty dicts.
        assert defaults[""] == {"type": "sphere", "size": "0.04"}

    def test_a_model_with_no_default_reports_both_names_as_empty(self):
        defaults = mod._mjcf_class_defaults(ET.fromstring('<mujoco model="m"><worldbody/></mujoco>'), ".", "geom")
        assert defaults[""] == {}
        assert defaults["main"] == {}


class TestWhyAliasingTheRootIsSafe:
    """MuJoCo's own refusals, which are what make ``""`` and ``main`` one class.

    If a future MuJoCo let a nested class be named ``main``, or let the root be
    renamed, the alias would map two distinct classes onto one and these tests
    are where that shows up.
    """

    def _refuses(self, xml):
        with pytest.raises(ValueError) as excinfo:
            mujoco.MjModel.from_xml_string(xml)
        return str(excinfo.value)

    def test_mujoco_refuses_to_rename_the_root_class(self):
        message = self._refuses(
            '<mujoco model="m"><default class="base"><geom type="capsule" size="0.1 0.2"/></default>'
            '<worldbody><body><geom class="base"/></body></worldbody></mujoco>'
        )
        assert "cannot be renamed" in message, message

    def test_mujoco_refuses_a_nested_class_named_main(self):
        message = self._refuses(
            '<mujoco model="m"><default><geom type="capsule" size="0.1 0.2"/>'
            '<default class="main"><geom size="0.05 0.3"/></default></default>'
            "<worldbody><body><geom/></body></worldbody></mujoco>"
        )
        assert "repeated default class name" in message, message

    def test_mujoco_refuses_a_nested_default_with_no_class(self):
        message = self._refuses(
            '<mujoco model="m"><default><geom type="capsule" size="0.1 0.2"/>'
            '<default><geom size="0.05 0.3"/></default></default>'
            "<worldbody><body><geom/></body></worldbody></mujoco>"
        )
        assert "empty class name" in message, message


class TestWhatTheAliasMustNotChange:
    """Controls, pinned in both directions."""

    @pytest.mark.parametrize("root_open", ["<default>", '<default class="main">'])
    def test_the_geoms_own_attribute_still_beats_the_root_class(self, tmp_path, root_open):
        path = _write(
            tmp_path,
            "own.xml",
            f"""<mujoco model="m">
  {root_open}<geom type="capsule" size="0.06 0.21"/></default>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="ball_g" class="main" type="sphere" size="0.09"/>
    </body>
  </worldbody>
</mujoco>
""",
        )
        true_type, true_size = _compiled_geom(path, "ball_g")
        assert true_type == "sphere", "premise: the geom's own type wins"
        ball = _body(path, "ball")
        assert ball.shape == "sphere", f"the root class overrode the geom's own type ({ball.shape})"
        # The reader spells a sphere ``(radius,)``; MuJoCo pads its size triple.
        assert ball.shape_size == pytest.approx((true_size[0],), abs=1e-9)

    def test_an_unnamed_root_still_supplies_an_unclassed_geom(self, tmp_path):
        """The path that already worked, so the alias is additive."""
        path = _write(
            tmp_path,
            "plain.xml",
            """<mujoco model="m">
  <default><geom type="sphere" size="0.04"/></default>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="ball_g"/>
    </body>
  </worldbody>
</mujoco>
""",
        )
        assert _compiled_geom(path, "ball_g")[0] == "sphere", "premise"
        assert _body(path, "ball").shape == "sphere"

    def test_a_class_a_model_never_declares_still_contributes_nothing(self, tmp_path):
        """``main`` is now always a key, so ``.get`` must still miss other names."""
        defaults = mod._mjcf_class_defaults(
            ET.fromstring('<mujoco model="m"><default><geom type="sphere" size="0.04"/></default></mujoco>'),
            ".",
            "geom",
        )
        assert "wheel" not in defaults
