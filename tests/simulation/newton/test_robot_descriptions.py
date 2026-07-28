"""Newton ``add_robot`` resolution of the ``robot_descriptions`` URDF long tail.

Newton ingests URDF natively, so robots that ship only a URDF (no MJCF) can be
loaded straight from ``robot_descriptions`` - a capability the MuJoCo backend
lacks. These tests exercise the resolution + import wiring end to end on a real
Newton model build, using a self-contained minimal URDF (no network clone) so
they are deterministic in CI:

    - ``source="robot_descriptions"`` routes through the URDF importer and
      discovers the robot's joints,
    - ``source=None`` falls back to a URDF when the registry has no asset,
    - ``source="registry"`` does NOT fall back,
    - an explicit ``urdf_path`` wins over ``source``,
    - bad / unknown selectors produce clear error dicts,
    - ``list_urdfs`` returns the registry + ``robot_descriptions`` URDF union.

The real ``robot_descriptions.panda_description`` flow (URDF clone + 9-joint
build) is the design target; it is validated out-of-band to keep the suite
network-free. The cheap name->module mapping it relies on is covered by
``tests/registry/test_robot_descriptions_urdf.py``.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")

_MINI_URDF = """<?xml version="1.0"?>
<robot name="mini_arm">
  <link name="base_link">
    <inertial><mass value="1.0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <link name="link1">
    <inertial><mass value="0.5"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <collision><geometry><box size="0.05 0.05 0.2"/></geometry></collision>
  </link>
  <link name="link2">
    <inertial><mass value="0.3"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <collision><geometry><box size="0.05 0.05 0.2"/></geometry></collision>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="2"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="2"/>
  </joint>
</robot>
"""


@pytest.fixture
def urdf_file(tmp_path):
    path = tmp_path / "mini_arm.urdf"
    path.write_text(_MINI_URDF)
    return str(path)


@pytest.fixture
def engine():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    sim = NewtonSimEngine(solver="mujoco")
    sim.create_world()
    yield sim
    sim.destroy()


def _stub_discovery(monkeypatch, *, urdf_path):
    """Make ``discover_urdf_path`` resolve any name to *urdf_path* (or None)."""
    from strands_robots.simulation.newton import simulation as sim_mod

    monkeypatch.setattr(sim_mod, "discover_urdf_path", lambda name: urdf_path)


class TestRobotDescriptionsSource:
    def test_loads_urdf_and_discovers_joints(self, engine, urdf_file, monkeypatch):
        _stub_discovery(monkeypatch, urdf_path=urdf_file)
        result = engine.add_robot("mini_arm", source="robot_descriptions")
        assert result["status"] == "success"
        # Joints are discovered from the URDF and exposed under short names.
        assert engine.robot_joint_names("mini_arm") == ["shoulder", "elbow"]
        # The world is a real Newton model that steps.
        assert engine.step(3)["status"] == "success"

    def test_unknown_robot_errors_with_hint(self, engine, monkeypatch):
        _stub_discovery(monkeypatch, urdf_path=None)
        result = engine.add_robot("not_a_robot", source="robot_descriptions")
        assert result["status"] == "error"
        assert "list_urdfs" in result["content"][0]["text"]


class TestSourceSelector:
    def test_unknown_source_rejected(self, engine):
        result = engine.add_robot("anything", source="bogus")
        assert result["status"] == "error"
        assert "Unknown source" in result["content"][0]["text"]

    def test_default_source_falls_back_to_urdf(self, engine, urdf_file, monkeypatch):
        # 'mini_arm' is not in the curated registry, so default resolution must
        # fall back to the robot_descriptions URDF rather than failing.
        _stub_discovery(monkeypatch, urdf_path=urdf_file)
        result = engine.add_robot("mini_arm", source=None)
        assert result["status"] == "success"
        assert engine.robot_joint_names("mini_arm") == ["shoulder", "elbow"]

    def test_registry_source_does_not_fall_back(self, engine, urdf_file, monkeypatch):
        # A URDF is available, but source='registry' must not consult it.
        _stub_discovery(monkeypatch, urdf_path=urdf_file)
        result = engine.add_robot("mini_arm", source="registry")
        assert result["status"] == "error"
        assert "mini_arm" not in engine.list_robots()

    def test_explicit_urdf_path_wins_over_source(self, engine, urdf_file):
        result = engine.add_robot("mini_arm", urdf_path=urdf_file, source="registry")
        assert result["status"] == "success"
        assert engine.robot_joint_names("mini_arm") == ["shoulder", "elbow"]


class TestResolveAsset:
    def test_default_returns_urdf_for_discoverable(self, engine, urdf_file, monkeypatch):
        _stub_discovery(monkeypatch, urdf_path=urdf_file)
        path, error = engine._resolve_asset("mini_arm", None)
        assert error is None
        assert path == urdf_file

    def test_registry_only_no_urdf_fallback(self, engine, urdf_file, monkeypatch):
        _stub_discovery(monkeypatch, urdf_path=urdf_file)
        path, error = engine._resolve_asset("mini_arm", "registry")
        assert path is None
        assert error and "mini_arm" in error

    def test_known_registry_robot_resolves_mjcf(self, engine):
        path, error = engine._resolve_asset("so100", None)
        assert error is None
        assert path is not None and path.endswith(".xml")


class TestListUrdfsUnion:
    def test_includes_robot_descriptions_urdf_long_tail(self, engine):
        result = engine.list_urdfs()
        assert result["status"] == "success"
        urdf_robots = result["content"][1]["json"]["robot_descriptions_urdf"]
        # Real static table (no network): the URDF long tail is non-empty and
        # surfaced for programmatic use alongside the human-readable listing.
        assert isinstance(urdf_robots, list) and len(urdf_robots) > 0
        if urdf_robots:
            assert "robot_descriptions" in result["content"][0]["text"]


class TestDataConfigResolvesTheAsset:
    """``data_config`` is the registry key; ``name`` is the instance label.

    Newton's ``add_robot`` documented ``data_config`` as "Accepted for ABC parity;
    unused by Newton" and resolved purely on ``name``. So the multi-robot form the
    MuJoCo backend actively recommends - its own hint reads "Prefer:
    add_robot(name='<instance_label>', data_config='so101')" - failed here::

        MUJOCO add_robot(name='armA', data_config='so101') -> success, joints 1..6
        NEWTON add_robot(name='armA', data_config='so101') -> error
               "Could not resolve a sim asset for robot 'armA'."

    The single-robot ``Robot("so101", backend="newton")`` only worked because the
    instance label happened to BE the registry key.
    """

    def test_data_config_resolves_an_instance_label(self, engine):
        """The regression: this errored on Newton and succeeded on MuJoCo."""
        result = engine.add_robot(name="armA", data_config="so101")

        assert result["status"] == "success", result
        assert engine.robot_joint_names("armA") == ["1", "2", "3", "4", "5", "6"]

    def test_two_instances_of_one_config_get_distinct_index_maps(self, engine):
        """The point of the form: two arms from one registry entry."""
        assert engine.add_robot(name="armA", data_config="so101")["status"] == "success"
        assert engine.add_robot(name="armB", data_config="so101", position=[0.6, 0.0, 0.0])["status"] == "success"

        first = {j: engine._joint_dof_index[("armA", j)] for j in engine.robot_joint_names("armA")}
        second = {j: engine._joint_dof_index[("armB", j)] for j in engine.robot_joint_names("armB")}

        assert set(first.values()).isdisjoint(second.values()), (first, second)

    def test_actuating_one_instance_leaves_the_other_alone(self, engine):
        """Distinct maps must mean distinct actuation, not just distinct numbers."""
        assert engine.add_robot(name="armA", data_config="so101")["status"] == "success"
        assert engine.add_robot(name="armB", data_config="so101", position=[0.6, 0.0, 0.0])["status"] == "success"

        assert engine.send_action({"1": 0.8}, robot_name="armA")["status"] == "success"
        engine.step(40)

        assert float(engine.get_observation("armA")["1"]) > 0.5
        assert float(engine.get_observation("armB")["1"]) == pytest.approx(0.0, abs=1e-3)

    def test_the_data_config_is_recorded_on_the_robot(self, engine):
        engine.add_robot(name="armA", data_config="so101")

        assert engine._world.robots["armA"].data_config == "so101"

    def test_a_name_only_caller_still_resolves(self, engine):
        """The fallback must stay: ``name`` as the registry key."""
        result = engine.add_robot("so101")

        assert result["status"] == "success", result
        assert engine.robot_joint_names("so101") == ["1", "2", "3", "4", "5", "6"]

    def test_an_unknown_data_config_names_the_config_not_the_label(self, engine):
        """Naming the instance label would send the caller after the wrong key."""
        result = engine.add_robot(name="armC", data_config="not_a_robot")

        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "not_a_robot" in text, text
        assert "armC" not in text, text
        assert text.isascii()

    def test_an_explicit_urdf_path_still_wins(self, engine, urdf_file):
        """Precedence: urdf_path > data_config > name."""
        result = engine.add_robot(name="armD", urdf_path=urdf_file, data_config="so101")

        assert result["status"] == "success", result
        # The mini URDF's joints, not so101's "1".."6".
        assert engine.robot_joint_names("armD") == ["shoulder", "elbow"]

    def test_resolve_asset_prefers_data_config_over_name(self, engine):
        """Unit-level: the resolver itself, independent of add_robot."""
        by_config, error = engine._resolve_asset("armA", None, "so100")

        assert error is None
        assert by_config is not None and by_config.endswith(".xml")
        # And the same label without a data_config does NOT resolve.
        by_name, name_error = engine._resolve_asset("armA", None)
        assert by_name is None
        assert name_error is not None
