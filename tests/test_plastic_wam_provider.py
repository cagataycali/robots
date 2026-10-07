"""plastic_wam provider: registry entry resolves and the class is a strands-robots Policy (no weights loaded)."""
import json
from pathlib import Path

from strands_robots.policies.base import Policy


def test_registry_entry_and_class():
    reg = json.loads((Path(__file__).parents[1] / "strands_robots/registry/policies.json").read_text())
    entry = reg["providers"]["plastic_wam"]
    assert entry["class"] == "PlasticWAMPolicy" and "plastic_wam" in entry["shorthands"]
    from strands_robots.policies.plastic_wam import PlasticWAMPolicy
    assert issubclass(PlasticWAMPolicy, Policy)
    assert PlasticWAMPolicy.provider_name.fget(object.__new__(PlasticWAMPolicy)) == "plastic_wam"
