# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-gated integration: ``remove_robot`` deletes the robot's prim from the stage.

The issue recipe (#4083), on a real Kit runtime: build a one-arm scene, remove
the robot, drop a 0.1 m cube from ``z=1.0`` onto the spot the arm occupied and
settle. If ``remove_robot`` left the articulation on the stage, the cube comes to
rest on the leftover collision (high); with the prim actually deleted it falls to
the ground (cube half-height, ``z ~ 0.05``). A re-add under the same name then
composes a single articulation rather than stacking onto the leftover.

Run with::

    STRANDS_GPU_TEST=1 hatch run test-integ \
        tests_integ/simulation/test_isaac_remove_robot_deletes_prim_gpu.py -m gpu -v
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

_GPU_ENABLED = os.environ.get("STRANDS_GPU_TEST", "0") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _GPU_ENABLED,
        reason="Requires an NVIDIA GPU + Isaac Sim 6.0. Set STRANDS_GPU_TEST=1 to enable.",
    ),
]

# A one-joint arm whose link1 is a tall vertical column standing over the base,
# so a cube dropped straight down onto the base spot rests on the column while
# the arm is present and falls to the ground once the arm's prim is deleted.
ARM_URDF = """<?xml version="1.0"?>
<robot name="prim_column_arm">
  <link name="base_link">
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision>
      <geometry><cylinder radius="0.06" length="0.04"/></geometry>
    </collision>
  </link>
  <joint name="shoulder_pan" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="10"/>
  </joint>
  <link name="link1">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.2"/>
      <geometry><box size="0.1 0.1 0.4"/></geometry>
    </collision>
  </link>
</robot>
"""


def _skip_if_isaac_unavailable() -> None:
    from strands_robots.simulation.isaac import IsaacSimulation

    available, reason = IsaacSimulation.is_available()
    if not available:
        pytest.skip(f"Isaac Sim not available: {reason}")


def _json_payload(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if "json" in c)


class TestRemoveRobotDeletesPrimGPU:
    def test_the_recipe(self, tmp_path):
        """One Kit boot: settle the arm, remove it, drop a cube onto the spot,
        and confirm the cube reaches the ground and a re-add composes one arm."""
        from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()

        urdf_path = tmp_path / "prim_column_arm.urdf"
        urdf_path.write_text(ARM_URDF)

        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
        try:
            assert sim.create_world()["status"] == "success"
            assert sim.add_robot("arm", urdf_path=str(urdf_path))["status"] == "success"
            sim.step(30)

            # The arm's column occupies the drop spot while it is present.
            r = sim.get_body_state(body_name="arm/link1")
            if r["status"] != "success":
                r = sim.get_body_state(body_name="link1")
            assert r["status"] == "success", f"get_body_state(link1): {r}"
            column_z = _json_payload(r)["position"][2]
            assert column_z > 0.15, f"arm column should stand above the base, got z={column_z}"

            # Remove the robot; a dynamic-body mutation, so the view is stale.
            assert sim.remove_robot("arm")["status"] == "success"
            assert sim._physics_view_stale is True

            # Drop a 0.1 m cube from z=1.0 straight onto the vacated base spot.
            assert (
                sim.add_object(
                    name="cube",
                    shape="box",
                    position=[0.0, 0.0, 1.0],
                    size=[0.1, 0.1, 0.1],
                    mass=0.5,
                    is_static=False,
                )["status"]
                == "success"
            )
            # Rebuild the tensor view the removal + add invalidated, then settle.
            assert sim.reset()["status"] == "success"
            assert sim._physics_view_stale is False
            sim.step(240)

            rest_z = _json_payload(sim.get_body_state(body_name="cube"))["position"][2]
            # A 0.1 m cube resting on the ground plane sits at its half-height.
            # A leftover arm column would hold it far higher (~0.45+).
            assert rest_z == pytest.approx(0.05, abs=0.03), (
                f"cube rested at z={rest_z}; the arm's prim was not deleted from the stage"
            )

            # Re-add under the same name: add_robot permits it, and the scene
            # holds a single articulation (one observation set, not a doubled one).
            assert sim.add_robot("arm", urdf_path=str(urdf_path))["status"] == "success"
            assert sim.reset()["status"] == "success"
            sim.step(10)
            obs = sim.get_observation("arm", skip_images=True)
            assert "shoulder_pan" in obs, f"re-added arm missing its joint: {sorted(obs)}"
        finally:
            sim.destroy()
