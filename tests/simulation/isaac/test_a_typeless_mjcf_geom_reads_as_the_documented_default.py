"""A ``<geom>`` that states no ``type`` is a sphere, and both readers say so.

MJCF gives ``<geom>`` a default type, and it is ``sphere``: ``<geom size="0.03"/>``
compiles to a 0.03 m ball, with ``size`` holding the radius. Two functions in
:mod:`strands_robots.simulation.isaac.loaders` resolve that attribute -
``_extract_mjcf_shape`` for a robot link's ``BodyDef``, and ``_geom_aabb`` for a
scene object's bound - and they disagreed about the default: the AABB reader used
``sphere``, the link reader used ``box``.

Read as a box, such a geom loses its ``size`` as well as its shape, because the
box branch needs three components and a ball declares one: a link written
``<geom size="0.03"/>`` reported a 0.05 m box, so a 60 mm ball spawned as a
100 mm cube and nothing in the load reported a problem. The two readers also
answered the same element two different ways, which is the shape this file pins
shut - both resolve the default through one name now, so neither can drift.

MuJoCo is the oracle throughout: the expected primitive and size are read back
off a compiled ``MjModel`` rather than restated, and the default itself is
asserted against what the compiler does with a typeless geom.
"""

from __future__ import annotations

import inspect

import pytest

from strands_robots.simulation.isaac import loaders as mod
from strands_robots.simulation.isaac.loaders import load_mjcf

mujoco = pytest.importorskip("mujoco")

_GEOM_TYPE_NAMES = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
}

_MODEL = """<mujoco model="m">
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="link_g" {type_attr} size="{size}"/>
    </body>
  </worldbody>
</mujoco>
"""

# (label, ``type`` spelling, ``size``, expected reading). The expected primitive
# is checked against MuJoCo as well; it is stated here so a row that stops
# exercising what it names fails rather than following the code.
_ROWS = [
    ("type unstated, one size", "", "0.03", ("sphere", (0.03,))),
    ("type unstated, three sizes", "", "0.03 0.04 0.05", ("sphere", (0.03,))),
    ('type="sphere"', 'type="sphere"', "0.03", ("sphere", (0.03,))),
    ('type="box"', 'type="box"', "0.03 0.04 0.05", ("box", (0.03, 0.04, 0.05))),
    ('type="capsule"', 'type="capsule"', "0.02 0.1", ("capsule", (0.02, 0.1))),
]


def _write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _compiled_geom(path, geom_name="link_g"):
    """MuJoCo's own ``(type, size)`` for a named geom - the oracle."""
    model = mujoco.MjModel.from_xml_path(str(path))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    assert gid >= 0, f"premise: MuJoCo compiled no geom named {geom_name!r}"
    # int(): a pybind11 enum does not match a numpy integer as a dict key.
    return _GEOM_TYPE_NAMES[int(model.geom_type[gid])], tuple(float(x) for x in model.geom_size[gid])


class TestATypelessGeomTakesTheFormatsDefault:
    """One element, one shape - whichever reader is asked, and MuJoCo agrees."""

    @pytest.mark.parametrize(("label", "type_attr", "size", "expected"), _ROWS, ids=[r[0] for r in _ROWS])
    def test_the_link_reads_as_the_shape_mujoco_compiles(self, tmp_path, label, type_attr, size, expected):
        path = _write(tmp_path, "m.xml", _MODEL.format(type_attr=type_attr, size=size))
        compiled_type, compiled_size = _compiled_geom(path)
        assert compiled_type == expected[0], f"premise: MuJoCo reads {label} as {compiled_type!r}"

        body = next(b for b in load_mjcf(str(path)).bodies if b.name == "link")
        assert body.shape == compiled_type, (
            f"{label}: MuJoCo compiles a {compiled_type!r} and the link reported "
            f"{body.shape!r} with size {body.shape_size}"
        )
        assert body.shape_size == pytest.approx(expected[1]), (
            f"{label}: the file states size={size!r} and the link reported {body.shape_size}"
        )
        # The declared extent has to survive, not just the primitive's name: a
        # ball read as a box fell back to 0.05 m on every axis.
        assert body.shape_size[0] == pytest.approx(compiled_size[0])

    def test_both_readers_answer_one_element_the_same_way(self, tmp_path):
        """The link extent and the scene-object bound describe the same geom."""
        path = _write(tmp_path, "agree.xml", _MODEL.format(type_attr="", size="0.03"))
        root = mod.ET.parse(str(path)).getroot()
        body_el = root.find(".//body")

        shape, shape_size = mod._extract_mjcf_shape(body_el, {}, "")
        aabb = mod._geom_aabb(body_el.find("geom"), {}, "")
        assert aabb is not None, "premise: the AABB reader resolves a typeless geom"
        _, half = aabb

        assert shape == "sphere"
        # A ball is its own rotation: the half-extent is the radius on each axis.
        assert half == pytest.approx((shape_size[0],) * 3), (
            f"the link reader reported {(shape, shape_size)} and the AABB reader {half} "
            "for one <geom size='0.03'/>"
        )

    def test_the_default_is_the_one_the_compiler_applies(self, tmp_path):
        """The constant is graded against MuJoCo, not against its own spelling."""
        path = _write(tmp_path, "default.xml", _MODEL.format(type_attr="", size="0.03"))
        assert mod._MJCF_DEFAULT_GEOM_TYPE == _compiled_geom(path)[0]

    def test_neither_reader_spells_the_default_itself(self):
        """Two spellings of one default is how the two readers came to disagree."""
        readers = (mod._extract_mjcf_shape, mod._geom_aabb, mod._refuse_non_finite_geom)
        for fn in readers:
            src = inspect.getsource(fn)
            assert 'attrs.get("type", _MJCF_DEFAULT_GEOM_TYPE)' in src, (
                f"{fn.__name__} resolves the geom type; it must do so through the shared default"
            )
            assert 'get("type", "' not in src, f"{fn.__name__} spells the MJCF geom default itself"

    def test_a_body_with_no_geom_keeps_the_no_geometry_proxy(self, tmp_path):
        """An absent element is not an absent attribute: there is no shape to name.

        A ``<geom>`` that omits ``type`` still describes geometry, so the format's
        default applies. A body with no ``<geom>`` at all describes none, and keeps
        the module's no-geometry box proxy - the same reading a URDF link with
        neither ``<visual>`` nor ``<collision>`` gets.
        """
        mjcf = """<mujoco model="m"><worldbody>
          <body name="empty"><joint name="j" type="hinge"/></body>
        </worldbody></mujoco>"""
        body = next(b for b in load_mjcf(str(_write(tmp_path, "empty.xml", mjcf))).bodies if b.name == "empty")
        assert (body.shape, body.shape_size) == ("box", (0.05, 0.05, 0.05))

    def test_a_class_supplied_type_still_wins_over_the_default(self, tmp_path):
        """The default applies to an unstated type, not to an unstated attribute."""
        mjcf = """<mujoco model="m">
          <default>
            <default class="seg"><geom type="capsule" size="0.02 0.1"/></default>
          </default>
          <worldbody>
            <body name="link"><geom name="link_g" class="seg"/></body>
          </worldbody>
        </mujoco>"""
        path = _write(tmp_path, "cls.xml", mjcf)
        compiled_type, _ = _compiled_geom(path)
        assert compiled_type == "capsule", "premise: the class supplies the type"
        body = next(b for b in load_mjcf(str(path)).bodies if b.name == "link")
        assert (body.shape, body.shape_size) == ("capsule", (0.02, 0.1))
