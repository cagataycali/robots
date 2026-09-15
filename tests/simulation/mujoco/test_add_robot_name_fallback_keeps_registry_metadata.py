"""``add_robot("so101")`` keeps the registry metadata of the model it loaded.

The deprecated name-as-registry-key fallback resolved the model from the
instance name but left the robot's ``data_config`` ``None``, so everything
keyed on it forgot which registry entry the model came from: ``set_gripper``
reported "the registry carries no gripper metadata for this robot" for an
entry that has it, a recording declared ``robot_type`` from the instance name,
``list_robots_info`` printed ``Config: direct``. ``Robot("so101")`` passes
``data_config`` and worked on the same scene, so the Python-API shape an owner
reaches for first (``Simulation().add_robot("so101")``) was the one that
failed.

The robot's ``data_config`` now records the registry entry the model was
built from whichever argument named it; the deprecation hint on the reply is
unchanged.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


def _text(result) -> str:
    return next(b["text"] for b in result["content"] if "text" in b)


@pytest.fixture
def sim():
    s = Simulation(tool_name="fallback_meta", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


class TestTheFallbackRecordsTheRegistryEntry:
    def test_data_config_is_the_registry_key_the_model_came_from(self, sim) -> None:
        result = sim.add_robot("so101")
        assert result["status"] == "success", _text(result)
        assert "deprecated name-as-registry-key fallback" in _text(result)
        assert sim._world.robots["so101"].data_config == "so101"

    def test_it_matches_what_the_documented_form_records(self, sim) -> None:
        assert sim.add_robot(name="arm", data_config="so101")["status"] == "success"
        assert sim.add_robot("so101")["status"] == "success"
        assert sim._world.robots["arm"].data_config == sim._world.robots["so101"].data_config == "so101"

    def test_set_gripper_resolves_the_registry_gripper_after_a_name_only_add(self, sim) -> None:
        assert sim.add_robot("so101")["status"] == "success"
        result = sim.set_gripper(robot_name="so101", state="close", steps=5)
        assert result["status"] == "success", _text(result)
        assert "carries no gripper metadata" not in _text(result)

    def test_list_robots_info_names_the_config(self, sim) -> None:
        assert sim.add_robot("so101")["status"] == "success"
        assert "Config: so101" in _text(sim.list_robots_info())
        assert "Config: direct" not in _text(sim.list_robots_info())

    def test_a_urdf_path_add_still_records_no_registry_entry(self, sim, tmp_path) -> None:
        """Only a model that came from the registry gets a registry key."""
        from strands_robots.simulation.model_registry import resolve_model

        path = resolve_model("so101")
        assert path
        result = sim.add_robot(name="direct", urdf_path=path)
        assert result["status"] == "success", _text(result)
        assert sim._world.robots["direct"].data_config is None
