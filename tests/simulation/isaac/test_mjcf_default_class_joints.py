"""A joint's attributes are its ``<default>`` class's, overridden by its own.

MJCF lets a ``<default>`` class supply the attributes that decide what a joint
*is* - ``type``, ``axis``, ``range`` - so a ``<joint>`` need not spell any of
them. The MJCF loader read all five joint attributes off the element, so a
class-declared joint fell to the loader's own defaults: ``hinge`` turning about
``(0, 0, 1)`` between ``-3.14159`` and ``3.14159``. A gripper finger declared
``type="slide"`` by its class was reported as a revolute joint, and a shoulder
whose class declares its axis and its travel was reported turning about the
wrong axis with no useful limit - both under a successful load.

The geom side of the same rule is settled by
:mod:`tests.simulation.isaac.test_mjcf_default_class_geometry`, whose resolver
docstring already states the contract this file extends to joints: every
attribute this module reads off either element kind goes through one rule, so no
reader sees half of what the element declared.

Where MuJoCo can compile the fixture it is the oracle: expectations are read out
of ``jnt_type`` / ``jnt_axis`` / ``jnt_range`` rather than restated, so a fixture
that is not a model MuJoCo accepts cannot quietly pin the wrong answer.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from strands_robots.simulation.isaac import loaders as mod
from strands_robots.simulation.isaac.loaders import load_mjcf

# MuJoCo's own numbering for mjtJoint, and the loader's mapping onto the
# articulation vocabulary ``JointDef`` speaks.
_MJ_JOINT_KIND = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
_KIND_TO_JOINT_TYPE = {"hinge": "revolute", "slide": "prismatic", "ball": "fixed", "free": "fixed"}

# The loader's own fallbacks - what a joint gets when nothing declares otherwise.
_FALLBACK_TYPE = "revolute"
_FALLBACK_AXIS = (0.0, 0.0, 1.0)
_FALLBACK_LIMIT = (-3.14159, 3.14159)
_FALLBACK_DAMPING = 0.1
_FALLBACK_ARMATURE = 0.01


def _write(tmp_path, name: str, xml: str) -> str:
    path = tmp_path / name
    path.write_text(xml)
    return str(path)


def _only_joint(path: str):
    robot = load_mjcf(path)
    assert len(robot.joints) == 1, f"premise: fixture declares one joint, loader reported {len(robot.joints)}"
    return robot.joints[0]


def _mujoco_joints(path: str) -> dict[str, dict[str, object]]:
    """What MuJoCo compiled, keyed by joint name - the oracle for a fixture."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(path)
    out: dict[str, dict[str, object]] = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
        dof = int(model.jnt_dofadr[i])
        out[name] = {
            "joint_type": _KIND_TO_JOINT_TYPE[_MJ_JOINT_KIND[int(model.jnt_type[i])]],
            "axis": tuple(round(float(v), 9) for v in model.jnt_axis[i]),
            "limits": (tuple(round(float(v), 9) for v in model.jnt_range[i]) if model.jnt_limited[i] else None),
            "damping": round(float(model.dof_damping[dof]), 9),
            "armature": round(float(model.dof_armature[dof]), 9),
        }
    return out


# A whole arm whose joints declare nothing themselves: every attribute that
# decides what each joint is comes from its class. MuJoCo compiles it, so it is
# the oracle for all five attributes at once.
_CLASS_DECLARED_ARM = """<mujoco model="class_declared">
  <compiler angle="radian"/>
  <default>
    <default class="shoulder">
      <joint type="hinge" axis="0 1 0" range="-1.85005 1.25664" damping="4.2" armature="0.07"/>
    </default>
    <default class="finger">
      <joint type="slide" axis="1 0 0" range="0 0.041" damping="0.9" armature="0.003"/>
    </default>
  </default>
  <worldbody>
    <body name="upper_arm" pos="0 0 0.4">
      <joint name="shoulder" class="shoulder"/>
      <geom name="upper_arm_geom" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.03"/>
      <body name="left_finger" pos="0 0 -0.25">
        <joint name="left_finger" class="finger"/>
        <geom name="left_finger_geom" type="box" size="0.01 0.005 0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class TestAJointTakesItsDefaultClassAttributes:
    """Every joint attribute a ``<default>`` class supplies reaches the joint."""

    def test_a_class_declared_kind_is_the_joints_kind(self, tmp_path):
        path = _write(
            tmp_path,
            "kind.xml",
            """<mujoco>
  <default><default class="f"><joint type="slide"/></default></default>
  <worldbody><body name="b"><joint name="j" class="f"/><geom name="g" size="0.01"/></body></worldbody>
</mujoco>
""",
        )
        assert _mujoco_joints(path)["j"]["joint_type"] == "prismatic", "premise: MuJoCo compiles a sliding joint"
        assert _only_joint(path).joint_type == "prismatic"

    def test_a_class_declared_axis_is_the_joints_axis(self, tmp_path):
        path = _write(
            tmp_path,
            "axis.xml",
            """<mujoco>
  <default><default class="s"><joint type="hinge" axis="0 1 0"/></default></default>
  <worldbody><body name="b"><joint name="j" class="s"/><geom name="g" size="0.01"/></body></worldbody>
</mujoco>
""",
        )
        expected = _mujoco_joints(path)["j"]["axis"]
        assert expected != _FALLBACK_AXIS, "premise: the fixture's axis differs from the loader's fallback"
        assert _only_joint(path).axis == pytest.approx(expected, abs=1e-9)

    def test_a_class_declared_range_bounds_the_joint(self, tmp_path):
        path = _write(
            tmp_path,
            "range.xml",
            """<mujoco>
  <compiler angle="radian"/>
  <default><default class="s"><joint type="hinge" range="-1.85005 1.25664"/></default></default>
  <worldbody><body name="b"><joint name="j" class="s"/><geom name="g" size="0.01"/></body></worldbody>
</mujoco>
""",
        )
        limits = _mujoco_joints(path)["j"]["limits"]
        assert limits is not None, "premise: MuJoCo limits this joint"
        joint = _only_joint(path)
        assert (joint.limit_lower, joint.limit_upper) == pytest.approx(limits, abs=1e-9)

    def test_a_class_declared_damping_and_armature_reach_the_joint(self, tmp_path):
        path = _write(
            tmp_path,
            "dyn.xml",
            """<mujoco>
  <default><default class="s"><joint type="hinge" damping="4.2" armature="0.07"/></default></default>
  <worldbody><body name="b"><joint name="j" class="s"/><geom name="g" size="0.01"/></body></worldbody>
</mujoco>
""",
        )
        truth = _mujoco_joints(path)["j"]
        assert (truth["damping"], truth["armature"]) != (_FALLBACK_DAMPING, _FALLBACK_ARMATURE), (
            "premise: the fixture differs from the loader's fallbacks"
        )
        joint = _only_joint(path)
        assert joint.damping == pytest.approx(truth["damping"], abs=1e-9)
        assert joint.armature == pytest.approx(truth["armature"], abs=1e-9)

    def test_a_body_childclass_reaches_a_joint_that_names_no_class(self, tmp_path):
        # ``childclass`` applies to every descendant that names no class of its
        # own, so the joint need not mention the class at all.
        path = _write(
            tmp_path,
            "childclass.xml",
            """<mujoco>
  <default><default class="f"><joint type="slide" axis="0 1 0"/></default></default>
  <worldbody>
    <body name="b" childclass="f"><joint name="j"/><geom name="g" size="0.01"/></body>
  </worldbody>
</mujoco>
""",
        )
        truth = _mujoco_joints(path)["j"]
        joint = _only_joint(path)
        assert joint.joint_type == truth["joint_type"] == "prismatic"
        assert joint.axis == pytest.approx(truth["axis"], abs=1e-9)

    def test_a_nested_class_inherits_its_enclosing_class(self, tmp_path):
        # ``inner`` spells only the range; ``type`` and ``axis`` come from the
        # class that encloses it.
        path = _write(
            tmp_path,
            "nested.xml",
            """<mujoco>
  <compiler angle="radian"/>
  <default>
    <default class="outer">
      <joint type="slide" axis="1 0 0"/>
      <default class="inner"><joint range="0 0.05"/></default>
    </default>
  </default>
  <worldbody><body name="b"><joint name="j" class="inner"/><geom name="g" size="0.01"/></body></worldbody>
</mujoco>
""",
        )
        truth = _mujoco_joints(path)["j"]
        joint = _only_joint(path)
        assert joint.joint_type == truth["joint_type"] == "prismatic"
        assert joint.axis == pytest.approx(truth["axis"], abs=1e-9)
        assert (joint.limit_lower, joint.limit_upper) == pytest.approx(truth["limits"], abs=1e-9)

    def test_a_class_declared_in_an_included_fragment_still_applies(self, tmp_path):
        # ``<default>`` is a top-level element, so it is model-global: the
        # fragment declaring the class need not be the one declaring the joint.
        (tmp_path / "classes.xml").write_text(
            """<mujoco>
  <default><default class="f"><joint type="slide" axis="0 1 0"/></default></default>
</mujoco>
"""
        )
        path = _write(
            tmp_path,
            "top.xml",
            """<mujoco>
  <include file="classes.xml"/>
  <worldbody><body name="b"><joint name="j" class="f"/><geom name="g" size="0.01"/></body></worldbody>
</mujoco>
""",
        )
        truth = _mujoco_joints(path)["j"]
        joint = _only_joint(path)
        assert joint.joint_type == truth["joint_type"] == "prismatic"
        assert joint.axis == pytest.approx(truth["axis"], abs=1e-9)

    def test_every_joint_matches_what_mujoco_compiled(self, tmp_path):
        # An arm whose joints declare nothing themselves: all five attributes
        # graded at once against the compiler, for both joints.
        path = _write(tmp_path, "arm.xml", _CLASS_DECLARED_ARM)
        truth = _mujoco_joints(path)
        got = {j.name: j for j in load_mjcf(path).joints}
        assert set(got) == set(truth), f"premise: the loader reports MuJoCo's joints, got {sorted(got)}"
        for name, want in truth.items():
            joint = got[name]
            assert joint.joint_type == want["joint_type"], f"{name}: joint kind"
            assert joint.axis == pytest.approx(want["axis"], abs=1e-9), f"{name}: axis"
            assert want["limits"] is not None, f"premise: MuJoCo limits {name}"
            assert (joint.limit_lower, joint.limit_upper) == pytest.approx(want["limits"], abs=1e-9), f"{name}: range"
            assert joint.damping == pytest.approx(want["damping"], abs=1e-9), f"{name}: damping"
            assert joint.armature == pytest.approx(want["armature"], abs=1e-9), f"{name}: armature"

    def test_one_class_carries_a_shape_and_a_degree_of_freedom(self, tmp_path):
        path = _write(
            tmp_path,
            "both.xml",
            """<mujoco>
  <compiler angle="radian"/>
  <default>
    <default class="c">
      <geom type="capsule" size="0.02 0.08"/>
      <joint type="slide" axis="1 0 0" range="0 0.05"/>
    </default>
  </default>
  <worldbody><body name="b"><joint name="j" class="c"/><geom name="g" class="c"/></body></worldbody>
</mujoco>
""",
        )
        robot = load_mjcf(path)
        body = next(b for b in robot.bodies if b.name == "b")
        assert (body.shape, body.shape_size) == ("capsule", pytest.approx((0.02, 0.08), abs=1e-9))
        joint = _only_joint(path)
        assert joint.joint_type == "prismatic"
        assert joint.axis == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


class TestTheJointsOwnAttributesWin:
    """A joint that spells an attribute keeps it - the class is the fallback."""

    def test_a_joint_that_spells_its_own_kind_keeps_it(self, tmp_path):
        path = _write(
            tmp_path,
            "own_kind.xml",
            """<mujoco>
  <default><default class="f"><joint type="slide"/></default></default>
  <worldbody>
    <body name="b"><joint name="j" class="f" type="hinge"/><geom name="g" size="0.01"/></body>
  </worldbody>
</mujoco>
""",
        )
        assert _mujoco_joints(path)["j"]["joint_type"] == "revolute"
        assert _only_joint(path).joint_type == "revolute"

    def test_a_joint_that_spells_its_own_range_keeps_it(self, tmp_path):
        path = _write(
            tmp_path,
            "own_range.xml",
            """<mujoco>
  <compiler angle="radian"/>
  <default><default class="s"><joint type="hinge" range="-1 1"/></default></default>
  <worldbody>
    <body name="b"><joint name="j" class="s" range="-0.25 0.5"/><geom name="g" size="0.01"/></body>
  </worldbody>
</mujoco>
""",
        )
        joint = _only_joint(path)
        assert (joint.limit_lower, joint.limit_upper) == pytest.approx((-0.25, 0.5), abs=1e-9)


class TestTheGeomSideIsUntouched:
    """Reading a joint's class changes nothing about how a geom's is read.

    ``<geom>`` and ``<joint>`` attribute sets are collected separately, so a
    class that declares only one of them leaves the other element exactly where
    it was: on its own attributes.
    """

    def test_a_class_with_no_joint_child_leaves_the_joint_on_its_own_attributes(self, tmp_path):
        path = _write(
            tmp_path,
            "geom_only.xml",
            """<mujoco>
  <compiler angle="radian"/>
  <default><default class="c"><geom type="capsule" size="0.02 0.08"/></default></default>
  <worldbody>
    <body name="b">
      <joint name="j" type="slide" axis="0 1 0" range="0 0.03"/>
      <geom name="g" class="c"/>
    </body>
  </worldbody>
</mujoco>
""",
        )
        robot = load_mjcf(path)
        body = next(b for b in robot.bodies if b.name == "b")
        assert (body.shape, body.shape_size) == ("capsule", pytest.approx((0.02, 0.08), abs=1e-9))
        joint = _only_joint(path)
        assert joint.joint_type == "prismatic"
        assert joint.axis == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)

    def test_a_class_with_no_geom_child_leaves_the_geom_on_its_own_attributes(self, tmp_path):
        path = _write(
            tmp_path,
            "joint_only.xml",
            """<mujoco>
  <compiler angle="radian"/>
  <default><default class="c"><joint type="slide" axis="1 0 0"/></default></default>
  <worldbody>
    <body name="b">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="g" type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
""",
        )
        robot = load_mjcf(path)
        body = next(b for b in robot.bodies if b.name == "b")
        assert body.shape == "sphere"
        # A sphere keeps the one component it declares.
        assert body.shape_size == pytest.approx((0.05,), abs=1e-9)


class TestTheContractIsNotWidened:
    """What a ``<default>`` class does not supply, and what this rule leaves alone."""

    def test_a_class_cannot_name_an_instance(self, tmp_path):
        # MuJoCo refuses ``name`` inside a ``<default>`` outright, so a joint's
        # name is always the element's own: it names one joint, not a kind.
        mujoco = pytest.importorskip("mujoco")
        with pytest.raises(Exception, match="name"):
            mujoco.MjModel.from_xml_string(
                '<mujoco><default><joint name="nope"/></default>'
                '<worldbody><body><joint/><geom size="0.01"/></body></worldbody></mujoco>'
            )
        path = _write(
            tmp_path,
            "named.xml",
            """<mujoco>
  <default><default class="f"><joint type="slide"/></default></default>
  <worldbody><body name="b"><joint name="my_joint" class="f"/><geom name="g" size="0.01"/></body></worldbody>
</mujoco>
""",
        )
        assert _only_joint(path).name == "my_joint"

    def test_an_undeclared_class_contributes_nothing(self, tmp_path):
        # MuJoCo refuses such a model itself, so naming the offending class is
        # its report to make; the reader falls back rather than failing.
        path = _write(
            tmp_path,
            "undeclared.xml",
            '<mujoco><worldbody><body name="b">'
            '<joint name="j" class="nope"/><geom name="g" size="0.01"/></body></worldbody></mujoco>',
        )
        joint = _only_joint(path)
        assert joint.joint_type == _FALLBACK_TYPE
        assert joint.axis == pytest.approx(_FALLBACK_AXIS, abs=1e-9)

    def test_compiler_angle_units_are_out_of_scope(self, tmp_path):
        # MJCF's angles are degrees unless ``<compiler angle="radian">`` says
        # otherwise, and this loader has never converted them. Reading a range
        # off a class does not change that: whichever element supplies the
        # value, the loader reports the number as written. Measured on the
        # shipped registry, this is the only remaining disagreement with the
        # compiler on a joint range, and it is a contract of its own.
        path = _write(
            tmp_path,
            "degrees.xml",
            """<mujoco>
  <worldbody><body name="b"><joint name="j" type="hinge" range="-15 22.5"/><geom name="g" size="0.01"/></body></worldbody>
</mujoco>
""",
        )
        compiled = _mujoco_joints(path)["j"]["limits"]
        assert compiled == pytest.approx((-0.261799, 0.392699), abs=1e-5), "premise: MuJoCo reads these as degrees"
        joint = _only_joint(path)
        assert (joint.limit_lower, joint.limit_upper) == pytest.approx((-15.0, 22.5), abs=1e-9)

    def test_limited_false_is_out_of_scope(self, tmp_path):
        # ``limited`` decides whether a range binds at all, and this loader has
        # never read it. A class-declared range is reported as the joint's
        # limits even where MuJoCo leaves the joint unlimited.
        path = _write(
            tmp_path,
            "unlimited.xml",
            """<mujoco>
  <compiler angle="radian" autolimits="false"/>
  <worldbody>
    <body name="b"><joint name="j" type="hinge" range="-1 1" limited="false"/><geom name="g" size="0.01"/></body>
  </worldbody>
</mujoco>
""",
        )
        assert _mujoco_joints(path)["j"]["limits"] is None, "premise: MuJoCo leaves this joint unlimited"
        joint = _only_joint(path)
        assert (joint.limit_lower, joint.limit_upper) == pytest.approx((-1.0, 1.0), abs=1e-9)


class TestOneRuleAnswersWhatAJointDeclared:
    """One rule answers "what did this joint declare", for every MJCF reader.

    The graded set is derived from the module: every function inside
    :func:`~strands_robots.simulation.isaac.loaders.load_mjcf` - its nested walk
    included - plus every module-level function whose name marks it an MJCF
    reader. So a second MJCF joint reader is held to the rule on arrival, and it
    mirrors the geom-side scan in
    :mod:`tests.simulation.isaac.test_mjcf_default_class_geometry`.

    ``load_urdf`` is outside the rule by format rather than by exemption: URDF
    has no ``<default>`` classes, so a URDF joint's attributes are only ever its
    element's own.
    """

    @staticmethod
    def _graded_mjcf_functions() -> dict[str, ast.FunctionDef]:
        tree = ast.parse(inspect.getsource(mod))
        graded: dict[str, ast.FunctionDef] = {}
        for top in tree.body:
            if not isinstance(top, ast.FunctionDef):
                continue
            if top.name == "load_mjcf" or "mjcf" in top.name:
                for node in ast.walk(top):
                    if isinstance(node, ast.FunctionDef):
                        graded[f"{top.name}.{node.name}" if node is not top else top.name] = node
        return graded

    def test_no_mjcf_reader_asks_a_joint_element_for_an_attribute(self):
        offenders = []
        graded = self._graded_mjcf_functions()
        for label, fn in graded.items():
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                if (
                    isinstance(f, ast.Attribute)
                    and f.attr == "get"
                    and isinstance(f.value, ast.Name)
                    and "joint" in f.value.id
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    # ``name`` is the one attribute a class cannot supply: MuJoCo
                    # refuses it inside a ``<default>`` outright, which
                    # ``test_a_class_cannot_name_an_instance`` pins.
                    and node.args[0].value != "name"
                ):
                    offenders.append(f"{label}:{node.lineno} {ast.unparse(node)}")
        # A scan that reached nothing would report clean forever.
        assert len(graded) >= 3, f"the scan reached only {sorted(graded)}"
        assert "load_mjcf" in graded, "the scan does not reach the MJCF reader it grades"
        assert not offenders, (
            "these read a joint attribute off the element, so a value its <default> class supplies "
            f"is invisible to them: {offenders}"
        )

    def test_the_scan_reaches_the_reader_it_grades(self):
        src = inspect.getsource(mod.load_mjcf)
        assert "_class_attrs(joint_el" in src, "load_mjcf does not resolve its joint attributes through the shared rule"
        assert '_mjcf_class_defaults(root, mjcf_dir, "joint")' in src, "load_mjcf collects no joint defaults"

    def test_the_urdf_reader_is_outside_the_rule_by_format(self):
        # URDF declares no default classes, so its joint attributes are only
        # ever the element's own. Grading it would report a correct reader.
        assert "load_urdf" not in self._graded_mjcf_functions()
        assert "<default" not in inspect.getsource(mod.load_urdf)
