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

``resolve_model`` resolves a decorated variant of a registry key to its model
as a documented friction fix (``"so101_arm"`` loads so101's model), so the
string that named the model is not always the key its entry is filed under -
recording it verbatim lost the entry the same way. The recorded key is the one
the entry is under; a name that names no entry at all is kept as passed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.model_registry import (  # noqa: E402
    _URDF_REGISTRY,
    register_urdf,
    registry_entry_key,
    resolve_model,
)
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
        path = resolve_model("so101")
        assert path
        result = sim.add_robot(name="direct", urdf_path=path)
        assert result["status"] == "success", _text(result)
        assert sim._world.robots["direct"].data_config is None


class TestTheRecordedKeyNamesAnEntry:
    """A decorated variant of a key resolves to its model; record the key."""

    @pytest.mark.parametrize("kwargs", [{"name": "so101_arm"}, {"name": "arm", "data_config": "so101_arm"}])
    def test_a_decorated_name_records_the_key_its_entry_is_under(self, sim, kwargs) -> None:
        result = sim.add_robot(**kwargs)
        assert result["status"] == "success", _text(result)
        assert sim._world.robots[kwargs["name"]].data_config == "so101"
        gripper = sim.set_gripper(robot_name=kwargs["name"], state="close", steps=5)
        assert gripper["status"] == "success", _text(gripper)

    def test_a_name_the_registry_does_not_carry_is_recorded_as_passed(self, sim) -> None:
        """A model registered outside the robot registry still names itself."""
        path = resolve_model("so101")
        assert path
        register_urdf("widget_no_registry_entry", path)
        try:
            assert sim.add_robot(name="w", data_config="widget_no_registry_entry")["status"] == "success"
            assert sim._world.robots["w"].data_config == "widget_no_registry_entry"
        finally:
            _URDF_REGISTRY.pop("widget_no_registry_entry", None)

    @pytest.mark.parametrize(
        ("named", "entry"),
        [
            ("so101", "so101"),  # a key names itself
            ("so101_follower", "so101_follower"),  # an alias names an entry
            ("so101_arm", "so101"),  # a decorated variant names the bare key
            ("so101_sim", "so101"),
            ("panda_robot", "panda"),
            ("google_robot", "google_robot"),  # a key that ENDS in a suffix
            ("nonesuch", None),  # names no entry
            ("nonesuch_arm", None),  # stripping it still names no entry
            ("_arm", None),  # a suffix alone strips to nothing
            ("", None),
        ],
    )
    def test_registry_entry_key_follows_the_ladder_resolve_model_resolves_through(self, named, entry) -> None:
        assert registry_entry_key(named) == entry
