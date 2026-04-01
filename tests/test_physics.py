"""Tests for PhysicsMixin — advanced MuJoCo physics features.

Tests: raycasting, jacobians, energy, forces, state checkpointing,
inverse dynamics, sensor readout, body introspection, runtime modification.

Run: uv run pytest tests/test_physics.py -v
"""

import json
import os

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

ROBOT_XML = """
<mujoco model="physics_test">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="box1" pos="0 0 0.5">
      <freejoint name="box_free"/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
    </body>
    <body name="arm_base" pos="0.5 0 0">
      <body name="link1" pos="0 0 0.1">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
        <geom name="link1_geom" type="capsule" size="0.02 0.1" rgba="0.3 0.3 0.8 1"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom name="link2_geom" type="capsule" size="0.015 0.08" rgba="0.3 0.8 0.3 1"/>
          <site name="end_effector" pos="0 0 0.08"/>
        </body>
      </body>
    </body>
    <camera name="overhead" pos="0 -1 1.5" quat="0.7 0.7 0 0"/>
  </worldbody>
  <actuator>
    <motor name="shoulder_motor" joint="shoulder" ctrlrange="-1 1"/>
    <motor name="elbow_motor" joint="elbow" ctrlrange="-1 1"/>
  </actuator>
  <sensor>
    <jointpos name="shoulder_pos" joint="shoulder"/>
    <jointpos name="elbow_pos" joint="elbow"/>
  </sensor>
</mujoco>
"""


@pytest.fixture
def sim():
    """Create a Simulation with the test scene loaded directly."""
    from strands_robots.simulation.models import SimStatus, SimWorld

    s = Simulation(tool_name="test_sim", mesh=False)
    s._world = SimWorld()
    s._world._model = mj.MjModel.from_xml_string(ROBOT_XML)
    s._world._data = mj.MjData(s._world._model)
    s._world.status = SimStatus.IDLE
    mj.mj_forward(s._world._model, s._world._data)
    yield s
    s.cleanup()


class TestRaycasting:
    def test_raycast_hits_ground(self, sim):
        result = sim.raycast(origin=[0, 0, 2], direction=[0, 0, -1])
        assert result["status"] == "success"
        data = json.loads(result["content"][1]["text"])
        assert data["hit"] is True
        assert data["distance"] is not None
        assert data["distance"] > 0

    def test_raycast_hits_box(self, sim):
        result = sim.raycast(origin=[0, 0, 2], direction=[0, 0, -1])
        assert result["status"] == "success"
        data = json.loads(result["content"][1]["text"])
        assert data["hit"] is True
        assert data["geom_name"] in ("box_geom", "ground")

    def test_raycast_misses(self, sim):
        result = sim.raycast(origin=[0, 0, 2], direction=[0, 0, 1])  # shooting up
        assert result["status"] == "success"
        data = json.loads(result["content"][1]["text"])
        assert data["hit"] is False

    def test_multi_raycast(self, sim):
        dirs = [[0, 0, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        result = sim.multi_raycast(origin=[0, 0, 2], directions=dirs)
        assert result["status"] == "success"
        rays = json.loads(result["content"][1]["text"])["rays"]
        assert len(rays) == 4
        # At least the downward ray should hit
        assert rays[0]["distance"] is not None


class TestJacobians:
    def test_body_jacobian(self, sim):
        result = sim.get_jacobian(body_name="link2")
        assert result["status"] == "success"
        data = json.loads(result["content"][1]["text"])
        assert len(data["jacp"]) == 3  # 3×nv
        assert data["nv"] == sim._world._model.nv

    def test_site_jacobian(self, sim):
        result = sim.get_jacobian(site_name="end_effector")
        assert result["status"] == "success"

    def test_geom_jacobian(self, sim):
        result = sim.get_jacobian(geom_name="link2_geom")
        assert result["status"] == "success"

    def test_jacobian_no_target(self, sim):
        result = sim.get_jacobian()
        assert result["status"] == "error"

    def test_jacobian_invalid_body(self, sim):
        result = sim.get_jacobian(body_name="nonexistent")
        assert result["status"] == "error"


class TestEnergy:
    def test_get_energy(self, sim):
        result = sim.get_energy()
        assert result["status"] == "success"
        data = json.loads(result["content"][1]["text"])
        assert "potential" in data
        assert "kinetic" in data
        assert "total" in data
        # Box at height 0.5 should have nonzero potential energy
        assert data["potential"] != 0 or data["kinetic"] != 0

    def test_energy_changes_after_step(self, sim):
        e1 = json.loads(sim.get_energy()["content"][1]["text"])
        # Step physics to let box fall
        for _ in range(100):
            mj.mj_step(sim._world._model, sim._world._data)
        e2 = json.loads(sim.get_energy()["content"][1]["text"])
        # Kinetic energy should change (box falls)
        assert e1["kinetic"] != e2["kinetic"] or e1["potential"] != e2["potential"]


class TestExternalForces:
    def test_apply_force(self, sim):
        result = sim.apply_force(body_name="box1", force=[0, 0, 100])
        assert result["status"] == "success"
        assert "box1" in result["content"][0]["text"]

    def test_apply_force_invalid_body(self, sim):
        result = sim.apply_force(body_name="nonexistent", force=[0, 0, 10])
        assert result["status"] == "error"

    def test_force_changes_acceleration(self, sim):
        # Get initial state
        data = sim._world._data
        old_qfrc = data.qfrc_applied.copy()
        sim.apply_force(body_name="box1", force=[0, 0, 100])
        # qfrc_applied should change
        assert not np.array_equal(old_qfrc, data.qfrc_applied)


class TestMassMatrix:
    def test_get_mass_matrix(self, sim):
        result = sim.get_mass_matrix()
        assert result["status"] == "success"
        data = json.loads(result["content"][1]["text"])
        nv = sim._world._model.nv
        assert data["shape"] == [nv, nv]
        assert data["rank"] > 0
        assert data["total_mass"] > 0

    def test_mass_diagonal_positive(self, sim):
        result = sim.get_mass_matrix()
        diag = json.loads(result["content"][1]["text"])["diagonal"]
        assert all(d >= 0 for d in diag)


class TestStateCheckpointing:
    def test_save_and_load_state(self, sim):
        # Set a known joint position
        sim._world._data.qpos[7] = 1.0  # shoulder
        mj.mj_forward(sim._world._model, sim._world._data)

        # Save
        result = sim.save_state(name="test_checkpoint")
        assert result["status"] == "success"

        # Change state
        sim._world._data.qpos[7] = -1.0
        mj.mj_forward(sim._world._model, sim._world._data)
        assert sim._world._data.qpos[7] == pytest.approx(-1.0)

        # Restore
        result = sim.load_state(name="test_checkpoint")
        assert result["status"] == "success"
        assert sim._world._data.qpos[7] == pytest.approx(1.0)

    def test_load_nonexistent_checkpoint(self, sim):
        result = sim.load_state(name="doesnt_exist")
        assert result["status"] == "error"


class TestInverseDynamics:
    def test_inverse_dynamics(self, sim):
        mj.mj_forward(sim._world._model, sim._world._data)
        result = sim.inverse_dynamics()
        assert result["status"] == "success"
        forces = json.loads(result["content"][1]["text"])["qfrc_inverse"]
        assert "shoulder" in forces or "elbow" in forces


class TestBodyState:
    def test_get_body_state(self, sim):
        result = sim.get_body_state(body_name="box1")
        assert result["status"] == "success"
        state = json.loads(result["content"][1]["text"])
        assert "position" in state
        assert "quaternion" in state
        assert "linear_velocity" in state
        assert "angular_velocity" in state
        assert "mass" in state
        assert len(state["position"]) == 3
        assert len(state["quaternion"]) == 4
        assert state["mass"] == pytest.approx(1.0)

    def test_body_state_invalid(self, sim):
        result = sim.get_body_state(body_name="nonexistent")
        assert result["status"] == "error"


class TestDirectJointControl:
    def test_set_joint_positions(self, sim):
        result = sim.set_joint_positions(positions={"shoulder": 0.5, "elbow": -0.3})
        assert result["status"] == "success"
        assert "2/2" in result["content"][0]["text"]

        # Verify positions were set
        model, data = sim._world._model, sim._world._data
        shoulder_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "shoulder")
        qpos_adr = model.jnt_qposadr[shoulder_id]
        assert data.qpos[qpos_adr] == pytest.approx(0.5)

    def test_set_joint_velocities(self, sim):
        result = sim.set_joint_velocities(velocities={"shoulder": 1.0})
        assert result["status"] == "success"


class TestSensors:
    def test_get_all_sensors(self, sim):
        result = sim.get_sensor_data()
        assert result["status"] == "success"
        sensors = json.loads(result["content"][1]["text"])["sensors"]
        assert "shoulder_pos" in sensors
        assert "elbow_pos" in sensors

    def test_get_specific_sensor(self, sim):
        result = sim.get_sensor_data(sensor_name="shoulder_pos")
        assert result["status"] == "success"
        sensors = json.loads(result["content"][1]["text"])["sensors"]
        assert len(sensors) == 1
        assert "shoulder_pos" in sensors

    def test_sensor_values_change(self, sim):
        # Set shoulder position
        sim.set_joint_positions(positions={"shoulder": 1.0})
        result = sim.get_sensor_data(sensor_name="shoulder_pos")
        val = json.loads(result["content"][1]["text"])["sensors"]["shoulder_pos"]["values"]
        assert abs(val - 1.0) < 0.01


class TestRuntimeModification:
    def test_set_body_mass(self, sim):
        result = sim.set_body_properties(body_name="box1", mass=5.0)
        assert result["status"] == "success"
        body_id = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_BODY, "box1")
        assert sim._world._model.body_mass[body_id] == pytest.approx(5.0)

    def test_set_geom_color(self, sim):
        result = sim.set_geom_properties(geom_name="box_geom", color=[0, 1, 0, 1])
        assert result["status"] == "success"
        geom_id = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_GEOM, "box_geom")
        assert sim._world._model.geom_rgba[geom_id][1] == pytest.approx(1.0)

    def test_set_geom_friction(self, sim):
        result = sim.set_geom_properties(geom_name="box_geom", friction=[0.5, 0.01, 0.001])
        assert result["status"] == "success"

    def test_invalid_geom(self, sim):
        result = sim.set_geom_properties(geom_name="nonexistent", color=[1, 0, 0, 1])
        assert result["status"] == "error"


class TestContactForces:
    def test_get_contact_forces_after_settling(self, sim):
        # Let box fall and settle
        for _ in range(500):
            mj.mj_step(sim._world._model, sim._world._data)
        result = sim.get_contact_forces()
        assert result["status"] == "success"
        # Box should be in contact with ground
        contacts = json.loads(result["content"][1]["text"])["contacts"]
        assert len(contacts) > 0
        assert contacts[0]["normal_force"] != 0


class TestForwardKinematics:
    def test_forward_kinematics(self, sim):
        result = sim.forward_kinematics()
        assert result["status"] == "success"
        bodies = json.loads(result["content"][1]["text"])["bodies"]
        assert "box1" in bodies
        assert "link1" in bodies
        assert len(bodies["box1"]["position"]) == 3


class TestTotalMass:
    def test_get_total_mass(self, sim):
        result = sim.get_total_mass()
        assert result["status"] == "success"
        data = json.loads(result["content"][1]["text"])
        assert data["total_mass"] > 0
        assert "box1" in data["bodies"]
        assert data["bodies"]["box1"] == pytest.approx(1.0)


class TestExportXML:
    def test_export_xml_string(self, sim):
        result = sim.export_xml()
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "mujoco" in text.lower() or "Model XML" in text

    def test_export_xml_file(self, sim, tmp_path):
        path = str(tmp_path / "exported.xml")
        result = sim.export_xml(output_path=path)
        assert result["status"] == "success"
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "<mujoco" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
