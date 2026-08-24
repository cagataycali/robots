"""A geom's shape is read from the ``<default>`` class it inherits.

MJCF lets a ``<geom>`` spell almost nothing itself: ``<default class="X">``
supplies every attribute the element omits, ``<body childclass="X">`` names that
class for a whole subtree, and the classes form an inheritance tree rooted at
the unnamed top-level ``<default>``. So ``type``, ``size`` and ``fromto`` - the
three attributes that decide what shape a geom is - need not be on the geom.

Read as the geom's own attributes alone, ``asimov_v0``'s six leg links report the
0.05 m fallback box for capsules whose ``<default class="body_capsule">``
declares ``type="capsule" size="0.05"`` and whose ``fromto`` runs 0.25 m: a 10 cm
cube for a 25 cm shin, on a shipped registry robot, under ``load_mjcf`` reporting
success. It also makes the endpoint reading ``_extract_mjcf_shape`` performs
unreachable for exactly those links, because that branch is only taken once the
geom is known to be a capsule or a cylinder.

MuJoCo is the oracle throughout: every expected shape here is read back off a
compiled ``MjModel`` rather than restated, so a fixture that is not a model
MuJoCo compiles cannot silently become the contract.
"""

from __future__ import annotations

import ast
import inspect
import xml.etree.ElementTree as ET

import pytest

from strands_robots.simulation.isaac import loaders as mod
from strands_robots.simulation.isaac.loaders import load_mjcf, load_mjcf_scene_objects

mujoco = pytest.importorskip("mujoco")

_GEOM_TYPE_NAMES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}


def _compiled_geom(path, geom_name):
    """MuJoCo's own ``(type, size)`` for a named geom - the oracle."""
    model = mujoco.MjModel.from_xml_path(str(path))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    assert gid >= 0, f"premise: MuJoCo compiled no geom named {geom_name!r}"
    # int(): a pybind11 enum does not match a numpy integer as a dict key.
    return _GEOM_TYPE_NAMES[int(model.geom_type[gid])], tuple(float(x) for x in model.geom_size[gid])


def _write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# The asimov_v0 shape: the class carries the type and the radius, the geom
# carries only the endpoints.
_CLASS_CAPSULE = """<mujoco model="m">
  <default>
    <default class="collision"><geom group="3" type="capsule"/></default>
    <default class="body_capsule"><geom type="capsule" size="0.05"/></default>
  </default>
  <worldbody>
    <body name="shin" pos="0 0 1">
      <joint name="knee" type="hinge" axis="0 1 0"/>
      <geom name="shin_col" class="body_capsule" fromto="0 0 0 0 0 -0.25"/>
    </body>
  </worldbody>
</mujoco>
"""


class TestAClassDeclaredShapeIsTheShape:
    """The regression: an attribute the geom inherits decides its shape."""

    def test_a_class_declared_capsule_is_read_as_a_capsule(self, tmp_path):
        path = _write(tmp_path, "shin.xml", _CLASS_CAPSULE)
        true_type, true_size = _compiled_geom(path, "shin_col")
        assert true_type == "capsule", "premise: MuJoCo reads the class's type"

        robot = load_mjcf(str(path))
        shin = next(b for b in robot.bodies if b.name == "shin")
        assert shin.shape == true_type, (
            f"the geom inherits type='capsule' from <default class='body_capsule'> and the loader "
            f"reported {shin.shape!r} with size {shin.shape_size}, the fallback box, for a 0.25 m segment"
        )
        # MuJoCo's capsule size is (radius, half-length).
        assert shin.shape_size == pytest.approx((true_size[0], true_size[1]), abs=1e-9)

    def test_childclass_supplies_the_class_for_a_whole_subtree(self, tmp_path):
        path = _write(
            tmp_path,
            "cc.xml",
            """<mujoco model="m">
  <default><default class="link"><geom type="cylinder" size="0.02 0.11"/></default></default>
  <worldbody>
    <body name="upper" childclass="link" pos="0 0 1">
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
        robot = load_mjcf(str(path))
        for name in ("upper", "lower"):
            body = next(b for b in robot.bodies if b.name == name)
            assert body.shape == "cylinder", f"{name} did not inherit childclass='link' ({body.shape})"
            assert body.shape_size == pytest.approx((0.02, 0.11 + 0.0), abs=1e-9)

    def test_a_nested_class_inherits_its_enclosing_classs_attributes(self, tmp_path):
        # ``inner`` declares only the size; the type comes from ``outer``.
        path = _write(
            tmp_path,
            "nested.xml",
            """<mujoco model="m">
  <default>
    <default class="outer"><geom type="capsule"/>
      <default class="inner"><geom size="0.03 0.09"/></default>
    </default>
  </default>
  <worldbody>
    <body name="seg" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="seg_g" class="inner"/>
    </body>
  </worldbody>
</mujoco>
""",
        )
        assert _compiled_geom(path, "seg_g")[0] == "capsule", "premise"
        seg = next(b for b in load_mjcf(str(path)).bodies if b.name == "seg")
        assert seg.shape == "capsule", f"the nested class did not inherit type from its parent ({seg.shape})"

    def test_the_unnamed_root_default_supplies_an_unclassed_geom(self, tmp_path):
        path = _write(
            tmp_path,
            "root.xml",
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
        true_type, true_size = _compiled_geom(path, "ball_g")
        assert true_type == "sphere", "premise"
        ball = next(b for b in load_mjcf(str(path)).bodies if b.name == "ball")
        assert ball.shape == "sphere", f"the root <default> was not read ({ball.shape})"
        # The reader spells a sphere ``(radius,)``; MuJoCo pads its size triple.
        assert ball.shape_size == pytest.approx((true_size[0],), abs=1e-9)

    def test_a_nested_body_childclass_overrides_the_outer_one(self, tmp_path):
        path = _write(
            tmp_path,
            "cc2.xml",
            """<mujoco model="m">
  <default>
    <default class="outer"><geom type="capsule" size="0.05 0.10"/></default>
    <default class="inner"><geom type="box" size="0.01 0.02 0.03"/></default>
  </default>
  <worldbody>
    <body name="a" childclass="outer" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom name="ag"/>
      <body name="b" childclass="inner" pos="0 0 -0.2">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom name="bg"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
        )
        assert _compiled_geom(path, "ag")[0] == "capsule", "premise"
        assert _compiled_geom(path, "bg")[0] == "box", "premise"
        bodies = {x.name: x for x in load_mjcf(str(path)).bodies}
        assert bodies["a"].shape == "capsule"
        assert bodies["b"].shape == "box", "the inner childclass did not override the outer one"

    def test_a_class_declared_in_an_include_fragment_is_model_global(self, tmp_path):
        # ``<default>`` is a top-level element, so a spliced fragment declares it
        # for the whole model - the same splice <compiler> and <asset> get.
        (tmp_path / "frag.xml").write_text(
            """<mujoco>
  <default><default class="rod"><geom type="capsule" size="0.01 0.07"/></default></default>
</mujoco>
""",
            encoding="utf-8",
        )
        path = _write(
            tmp_path,
            "top.xml",
            """<mujoco model="m">
  <include file="frag.xml"/>
  <worldbody>
    <body name="rod" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="rod_g" class="rod"/>
    </body>
  </worldbody>
</mujoco>
""",
        )
        assert _compiled_geom(path, "rod_g")[0] == "capsule", "premise: MuJoCo splices the fragment"
        rod = next(b for b in load_mjcf(str(path)).bodies if b.name == "rod")
        assert rod.shape == "capsule", f"a class from an <include> fragment was not read ({rod.shape})"


class TestTheSceneObjectReaderReadsTheClassToo:
    """The same rule in the collision-proxy reader, and in mesh selection."""

    def test_a_scene_object_proxy_reads_a_class_declared_box(self, tmp_path):
        path = _write(
            tmp_path,
            "scene.xml",
            """<mujoco model="s">
  <default><default class="fixture"><geom type="box" size="0.30 0.20 0.02" group="0"/></default></default>
  <worldbody>
    <body name="shelf" pos="0.5 0 0.4"><geom name="shelf_g" class="fixture"/></body>
  </worldbody>
</mujoco>
""",
        )
        assert _compiled_geom(path, "shelf_g")[0] == "box", "premise"
        shelf = next(o for o in load_mjcf_scene_objects(str(path)) if o.name == "shelf")
        # 0.60 x 0.40 x 0.04 full extent, not the mesh-less fallback.
        assert shelf.size == pytest.approx((0.60, 0.40, 0.04), abs=1e-6), (
            f"the proxy for a class-declared box came back {shelf.size}"
        )

    def test_a_mesh_named_only_by_a_class_is_found(self, tmp_path):
        # The abh_* hands' shape: every geom is ``<geom class="X"/>`` and the
        # class carries the mesh name, so asking the element sees no mesh at all.
        (tmp_path / "m.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\n", encoding="utf-8")
        path = _write(
            tmp_path,
            "meshscene.xml",
            """<mujoco model="s">
  <asset><mesh name="widget" file="m.obj"/></asset>
  <default><default class="vis"><geom type="mesh" mesh="widget" group="2"/></default></default>
  <worldbody>
    <body name="widget" pos="0.4 0 0.1"><geom name="widget_g" class="vis"/></body>
  </worldbody>
</mujoco>
""",
        )
        widget = next(o for o in load_mjcf_scene_objects(str(path)) if o.name == "widget")
        assert widget.mesh_path is not None, "a mesh named only by its <default> class was not found"
        assert widget.mesh_path.endswith("m.obj")


class TestThePrecedenceMJCFDefines:
    """Controls. Both hold on ``upstream/main`` too: they pin that resolving a
    class does not displace what the geom spells itself, and that a model with
    no ``<default>`` at all is read exactly as before. Each fails for one of the
    tempting shortcuts - letting the class win over the element, or making the
    resolver's answer mandatory where there is nothing to resolve.
    """

    def test_a_geoms_own_attribute_overrides_its_class(self, tmp_path):
        path = _write(
            tmp_path,
            "override.xml",
            """<mujoco model="m">
  <default><default class="c"><geom type="capsule" size="0.05 0.10"/></default></default>
  <worldbody>
    <body name="b" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="g" class="c" type="box" size="0.01 0.02 0.03"/>
    </body>
  </worldbody>
</mujoco>
""",
        )
        assert _compiled_geom(path, "g")[0] == "box", "premise: the geom's own type wins"
        b = next(x for x in load_mjcf(str(path)).bodies if x.name == "b")
        assert b.shape == "box"
        assert b.shape_size == pytest.approx((0.01, 0.02, 0.03), abs=1e-9)

    def test_a_model_with_no_defaults_is_read_exactly_as_before(self, tmp_path):
        path = _write(
            tmp_path,
            "plain.xml",
            """<mujoco model="m">
  <worldbody>
    <body name="b" pos="0 0 1">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="g" type="capsule" size="0.02 0.08"/>
    </body>
  </worldbody>
</mujoco>
""",
        )
        b = next(x for x in load_mjcf(str(path)).bodies if x.name == "b")
        assert (b.shape, b.shape_size) == ("capsule", pytest.approx((0.02, 0.08), abs=1e-9))


class TestOneRuleAnswersWhatAGeomDeclared:
    """One rule answers "what did this geom declare", for every reader.

    A reader that asks the element directly sees only the half the geom spells
    itself, which is how the two shape readers came to disagree with MuJoCo. The
    scan is derived from the module rather than a list of names, so a fifth
    reader is held to the rule on arrival.
    """

    def test_every_geom_attribute_read_goes_through_the_resolver(self):
        tree = ast.parse(inspect.getsource(mod))
        offenders = []
        scanned = 0
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef) or fn.name == "_class_attrs":
                continue
            scanned += 1
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                if (
                    isinstance(f, ast.Attribute)
                    and f.attr == "get"
                    and isinstance(f.value, ast.Name)
                    and f.value.id.startswith("geom")
                ):
                    offenders.append(f"{fn.name}:{node.lineno} {ast.unparse(node)}")
        # A scan that reached nothing would report clean forever.
        assert scanned >= 20, f"the scan reached only {scanned} functions in the module"
        assert not offenders, (
            "these read a geom attribute off the element, so a value its <default> class supplies "
            f"is invisible to them: {offenders}"
        )

    def test_an_undeclared_class_contributes_nothing(self, tmp_path):
        # MuJoCo refuses such a model itself, so naming the offending class is
        # its report to make; the reader must not fail the load over it.
        from strands_robots.simulation.isaac.loaders import _class_attrs, _mjcf_class_defaults

        root = ET.fromstring(
            '<mujoco><worldbody><body name="b"><geom name="g" class="nope"/></body></worldbody></mujoco>'
        )
        defaults = _mjcf_class_defaults(root, str(tmp_path), "geom")
        geom = root.find(".//geom")
        assert geom is not None
        assert _class_attrs(geom, defaults, "") == {"name": "g", "class": "nope"}

    def test_the_scan_reaches_the_readers_it_grades(self):
        # A scan that reaches nothing would report clean forever.
        src = inspect.getsource(mod)
        for name in ("_extract_mjcf_shape", "_geom_aabb", "_body_collision_aabb", "_find_body_mesh"):
            assert f"def {name}(" in src, f"premise: {name} is no longer in the scanned module"
            assert "_class_attrs(" in inspect.getsource(getattr(mod, name)), (
                f"{name} does not resolve its geom attributes through the shared rule"
            )
