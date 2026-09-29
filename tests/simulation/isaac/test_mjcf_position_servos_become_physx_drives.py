"""An MJCF position servo becomes a PhysX drive with its gains, in USD's units.

The Isaac MJCF importer (6.0.1 and 6.1.0) converts a ``<position>`` actuator only
when ``biasprm[2] < 0``. Menagerie servos state ``dampratio``, which MuJoCo's XML
layer stores as ``biasprm[2] = +dampratio``, so every one of them fell through and
the drive stayed ``stiffness=0, damping=0``. Measured on so100
(live GPU probes on an L40S): every joint limp -
Wrist_Pitch 1.3 rad off a 0.3 rad target, Jaw resting on its LOWER limit when
commanded to 10 rad, ``move_to`` failing with a 0.39 m residual. After: tracking
errors within 0.05 rad of MuJoCo's on every joint, limit commands saturate at
the declared limits, ``move_to`` reaches.
"""

from __future__ import annotations

import math
import pathlib

import pytest

pytest.importorskip("strands_robots.simulation.isaac")
mujoco = pytest.importorskip("mujoco")
Usd = pytest.importorskip("pxr.Usd")
from pxr import Sdf, UsdPhysics  # type: ignore[import-not-found]  # noqa: E402

from strands_robots.simulation.isaac import mjcf_assets  # noqa: E402

_MJCF = """
<mujoco>
  <default>
    <default class="servo"><position kp="50" dampratio="1" forcerange="-3.5 3.5"/></default>
  </default>
  <worldbody>
    <body name="link">
      <joint name="pan" type="hinge" axis="0 0 1" range="-1 1"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
      <body name="slider" pos="0 0 0.1">
        <joint name="lift" type="slide" axis="0 0 1" range="0 0.1"/>
        <geom type="box" size="0.02 0.02 0.02" mass="0.2"/>
      </body>
      <body name="spun" pos="0.1 0 0">
        <joint name="spin" type="hinge" axis="1 0 0"/>
        <geom type="sphere" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position class="servo" joint="pan"/>
    <position joint="lift" kp="200" kv="5"/>
    <motor joint="spin" gear="1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def mjcf(tmp_path: pathlib.Path) -> str:
    path = tmp_path / "arm.xml"
    path.write_text(_MJCF, encoding="utf-8")
    return str(path)


def _converted_usd(tmp_path: pathlib.Path) -> str:
    """The vendor 6.1 layout: joints behind an unselected ``Physics`` variant, drives zeroed."""
    path = tmp_path / "arm.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = stage.DefinePrim("/arm", "Xform")
    stage.SetDefaultPrim(root)
    vset = root.GetVariantSets().AddVariantSet("Physics")
    for name in ("mujoco", "physx"):
        vset.AddVariant(name)
    vset.SetVariantSelection("physx")
    with vset.GetVariantEditContext():
        for joint, kind in (("pan", "angular"), ("lift", "linear"), ("spin", "angular")):
            cls = UsdPhysics.RevoluteJoint if kind == "angular" else UsdPhysics.PrismaticJoint
            prim = cls.Define(stage, f"/arm/Physics/{joint}").GetPrim()
            drive = UsdPhysics.DriveAPI.Apply(prim, kind)
            drive.CreateStiffnessAttr().Set(0.0)
            drive.CreateDampingAttr().Set(0.0)
    vset.ClearVariantSelection()
    stage.GetRootLayer().Save()
    return str(path)


def _drive(path: str, joint: str, kind: str) -> tuple[float, float, float | None]:
    stage = Usd.Stage.Open(path)
    stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
    drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f"/arm/Physics/{joint}"), kind)
    return drive.GetStiffnessAttr().Get(), drive.GetDampingAttr().Get(), drive.GetMaxForceAttr().Get()


class TestTheGainsAreReadFromTheCompiledModel:
    def test_the_vendor_premise_dampratio_is_a_positive_biasprm(self, mjcf) -> None:
        spec = mujoco.MjSpec.from_file(mjcf)
        assert spec.actuators[0].biasprm[2] > 0  # what the importer's `< 0` check refuses

    def test_dampratio_resolves_to_a_real_kd(self, mjcf) -> None:
        gains = mjcf_assets._position_servo_gains(mjcf)
        kp, kd, force = gains["pan"]
        assert kp == 50 and force == 3.5
        assert kd > 0  # compiled 2*sqrt(kp*inertia)*dampratio, not +1

    def test_an_explicit_kv_servo_and_a_motor(self, mjcf) -> None:
        gains = mjcf_assets._position_servo_gains(mjcf)
        assert gains["lift"][:2] == (200.0, 5.0)
        assert "spin" not in gains  # a motor is not a position servo

    def test_an_uncompilable_model_answers_empty(self, tmp_path) -> None:
        bad = tmp_path / "bad.xml"
        bad.write_text("<mujoco><worldbody><body><joint/></body></worldbody>", encoding="utf-8")
        assert mjcf_assets._position_servo_gains(str(bad)) == {}


class TestTheDrivesAreAuthored:
    def test_revolute_gains_are_converted_to_per_degree(self, mjcf, tmp_path) -> None:
        usd = _converted_usd(tmp_path)
        kp, kd, _ = mjcf_assets._position_servo_gains(mjcf)["pan"]

        written = mjcf_assets._author_position_drives(usd, mjcf)

        assert sorted(written) == ["lift", "pan"]
        stiffness, damping, force = _drive(usd, "pan", "angular")
        assert stiffness == pytest.approx(kp * math.pi / 180)
        assert damping == pytest.approx(kd * math.pi / 180)
        assert force == pytest.approx(3.5)

    def test_prismatic_gains_are_si(self, mjcf, tmp_path) -> None:
        usd = _converted_usd(tmp_path)
        mjcf_assets._author_position_drives(usd, mjcf)
        assert _drive(usd, "lift", "linear")[:2] == pytest.approx((200.0, 5.0))

    def test_a_motor_joint_keeps_the_vendor_drive(self, mjcf, tmp_path) -> None:
        usd = _converted_usd(tmp_path)
        mjcf_assets._author_position_drives(usd, mjcf)
        assert _drive(usd, "spin", "angular")[:2] == (0.0, 0.0)

    def test_the_opinion_lives_in_the_root_layer_outside_the_variant(self, mjcf, tmp_path) -> None:
        usd = _converted_usd(tmp_path)
        mjcf_assets._author_position_drives(usd, mjcf)
        layer = Sdf.Layer.FindOrOpen(usd)
        assert layer.GetPropertyAtPath("/arm/Physics/pan.drive:angular:physics:stiffness") is not None

    def test_the_cache_key_names_the_postprocess(self) -> None:
        assert mjcf_assets._POSTPROCESS_VERSION
