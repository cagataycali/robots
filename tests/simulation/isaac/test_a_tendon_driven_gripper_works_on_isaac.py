"""A tendon-driven MJCF gripper - the Panda's - works on Isaac: resolved, driven and coupled.

Measured on one L40S, Isaac Sim 6.1, the Panda from the registry:

* ``set_gripper`` and ``move_to`` were refused outright: the registry names the
  gripper by its MuJoCo ACTUATOR, ``actuator8``, and an Isaac articulation has
  joints only (``finger_joint1``, ``finger_joint2``);
* resolved by hand, the fingers still did not move: ``actuator8`` is a servo on
  the ``split`` tendon, and the importer left both finger joints at
  ``stiffness=0`` (open read 0.0069 / 0.0, close 0.0228 / 0.0 - drift, not a
  command);
* the converter marks ``finger_joint1`` with ``NewtonMimicAPI`` but authors no
  ``newton:mimicJoint`` relationship, so USD Physics logged "must have exactly 1
  newton:mimicJoint relationship" on every reset.

After: open 0.0396 / 0.0396, close 0.0004 / 0.0004, ``move_to`` reaches, and the
error is gone. No GPU here: the MJCF is compiled by MuJoCo, the USD is a
hand-built stand-in of the importer's layout.
"""

from __future__ import annotations

import pathlib
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")
mujoco = pytest.importorskip("mujoco")
Usd = pytest.importorskip("pxr.Usd")
from pxr import UsdPhysics  # type: ignore[import-not-found]  # noqa: E402

from strands_robots.simulation.isaac import mjcf_assets  # noqa: E402
from strands_robots.simulation.isaac.motion_primitives import IsaacMotionPrimitivesMixin  # noqa: E402

# The Panda gripper, cut down: two slide fingers, a fixed tendon over both, an
# equality coupling, and a general actuator acting as a servo on the tendon.
_MJCF = """
<mujoco>
  <worldbody>
    <body name="hand">
      <geom type="box" size="0.02 0.02 0.02" mass="0.5"/>
      <body name="left_finger" pos="0 0.01 0">
        <joint name="finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>
        <geom type="box" size="0.01 0.01 0.02" mass="0.05"/>
      </body>
      <body name="right_finger" pos="0 -0.01 0">
        <joint name="finger_joint2" type="slide" axis="0 -1 0" range="0 0.04"/>
        <geom type="box" size="0.01 0.01 0.02" mass="0.05"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="split">
      <joint joint="finger_joint1" coef="0.5"/>
      <joint joint="finger_joint2" coef="0.5"/>
    </fixed>
  </tendon>
  <equality>
    <joint joint1="finger_joint1" joint2="finger_joint2"/>
  </equality>
  <actuator>
    <general name="actuator8" tendon="split" forcerange="-100 100" ctrlrange="0 255" biastype="affine"
      gainprm="0.01568627451 0 0" biasprm="0 -100 -10"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def mjcf(tmp_path: pathlib.Path) -> str:
    path = tmp_path / "hand.xml"
    path.write_text(_MJCF, encoding="utf-8")
    return str(path)


def _converted_usd(tmp_path: pathlib.Path) -> str:
    """The importer's 6.1 layout: joints behind the ``Physics`` variant, zero drives, a bare mimic API."""
    path = tmp_path / "hand.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = stage.DefinePrim("/hand", "Xform")
    stage.SetDefaultPrim(root)
    vset = root.GetVariantSets().AddVariantSet("Physics")
    for name in ("mujoco", "physx"):
        vset.AddVariant(name)
    vset.SetVariantSelection("physx")
    with vset.GetVariantEditContext():
        for joint in ("finger_joint1", "finger_joint2"):
            prim = UsdPhysics.PrismaticJoint.Define(stage, f"/hand/Physics/{joint}").GetPrim()
            drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
            drive.CreateStiffnessAttr().Set(0.0)
            drive.CreateDampingAttr().Set(0.0)
            if joint == "finger_joint1":
                prim.AddAppliedSchema("NewtonMimicAPI")
    vset.ClearVariantSelection()
    stage.GetRootLayer().Save()
    return str(path)


def _composed(path: str) -> Any:
    stage = Usd.Stage.Open(path)
    stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
    return stage


class TestTheActuatorIsTranslatedToJoints:
    def test_a_tendon_actuator_names_every_coupled_joint(self, mjcf: str) -> None:
        assert mjcf_assets.mjcf_actuator_joints(mjcf) == {"actuator8": ("finger_joint1", "finger_joint2")}

    def test_no_file_no_map(self, tmp_path: pathlib.Path) -> None:
        assert mjcf_assets.mjcf_actuator_joints(str(tmp_path / "missing.xml")) == {}
        assert mjcf_assets.mjcf_actuator_joints(None) == {}

    def test_set_gripper_resolves_the_registry_actuator_to_both_fingers(self, mjcf: str) -> None:
        engine = types.SimpleNamespace(
            _registry_gripper_metadata=lambda robot: (
                {"actuators": ["actuator8"], "closed": "low", "open": "high"},
                None,
            ),
            _short_joint_name=IsaacMotionPrimitivesMixin._short_joint_name,
        )
        engine._gripper_joint_vocabulary = types.MethodType(
            IsaacMotionPrimitivesMixin._gripper_joint_vocabulary, engine
        )
        robot = types.SimpleNamespace(
            name="panda",
            data_config="panda",
            joint_names=["joint1", "joint7", "finger_joint1", "finger_joint2"],
            description_path=mjcf,
        )
        dofs, meta, error = IsaacMotionPrimitivesMixin._resolve_gripper_dofs(engine, robot)  # type: ignore[arg-type]
        assert error is None and dofs == [2, 3] and meta is not None

    def test_without_the_mjcf_the_refusal_is_unchanged(self) -> None:
        engine = types.SimpleNamespace(
            _registry_gripper_metadata=lambda robot: (
                {"actuators": ["actuator8"], "closed": "low", "open": "high"},
                None,
            ),
            _short_joint_name=IsaacMotionPrimitivesMixin._short_joint_name,
        )
        engine._gripper_joint_vocabulary = types.MethodType(
            IsaacMotionPrimitivesMixin._gripper_joint_vocabulary, engine
        )
        robot = types.SimpleNamespace(
            name="panda", data_config="panda", joint_names=["finger_joint1"], description_path=None
        )
        dofs, _, error = IsaacMotionPrimitivesMixin._resolve_gripper_dofs(engine, robot)  # type: ignore[arg-type]
        assert dofs == [] and error is not None and "none match a joint" in error["content"][0]["text"]


class TestTheTendonServoBecomesJointDrives:
    def test_each_finger_gets_the_tendon_servo_scaled_by_its_coefficient(self, mjcf: str) -> None:
        gains = mjcf_assets._position_servo_gains(mjcf)
        # kp 100 N/m on the tendon length, coef 0.5 each, sum 1 -> 50 N/m per joint.
        assert gains["finger_joint1"] == pytest.approx((50.0, 5.0, 50.0))
        assert gains["finger_joint2"] == pytest.approx((50.0, 5.0, 50.0))

    def test_the_drives_and_the_mimic_relationship_are_authored(self, mjcf: str, tmp_path: pathlib.Path) -> None:
        usd = _converted_usd(tmp_path)
        written = mjcf_assets._author_position_drives(usd, mjcf)
        assert {"finger_joint1", "finger_joint2", "finger_joint1->mimic:finger_joint2"} <= set(written)
        stage = _composed(usd)
        for joint in ("finger_joint1", "finger_joint2"):
            drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath(f"/hand/Physics/{joint}"), "linear")
            assert drive.GetStiffnessAttr().Get() == pytest.approx(50.0)
            assert drive.GetDampingAttr().Get() == pytest.approx(5.0)
            assert drive.GetMaxForceAttr().Get() == pytest.approx(50.0)
        rel = stage.GetPrimAtPath("/hand/Physics/finger_joint1").GetRelationship("newton:mimicJoint")
        assert [str(t) for t in rel.GetTargets()] == ["/hand/Physics/finger_joint2"]

    def test_the_equality_names_follower_and_leader(self, mjcf: str) -> None:
        assert mjcf_assets._mimic_leaders(mjcf) == {"finger_joint1": "finger_joint2"}

    def test_the_cache_key_moved_so_old_conversions_are_not_reused(self) -> None:
        assert mjcf_assets._POSTPROCESS_VERSION not in ("drives-v1", "drives-v2", "drives-v3")
