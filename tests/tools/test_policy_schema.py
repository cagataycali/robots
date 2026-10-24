"""An agent reads the shipped presets and every wire form of a policy spec from the tool spec, before its first call."""

from __future__ import annotations

import pytest

from strands_robots import Robot, Simulation
from strands_robots.policy import PolicyLike
from strands_robots.policy.factory import FORMS
from strands_robots.policy.onnx import presets
from strands_robots.tools import parameters

WIRE_FORMS = [f.split()[0] for f in FORMS if not f.startswith("<callable>")]


def _robot_policy() -> dict:
    return Robot("so101").tool.tool_spec["inputSchema"]["json"]["properties"]["run"]["properties"]["policy"]


def _world_policy() -> dict:
    run = Simulation.default().tool.tool_spec["inputSchema"]["json"]["properties"]["run"]
    return run["properties"]["policies"]["additionalProperties"]


@pytest.mark.parametrize("schema", [_robot_policy, _world_policy], ids=["Robot.run", "Simulation.run"])
def test_the_policy_schema_names_every_shipped_preset_and_wire_form(schema):
    s = schema()
    assert s["type"] == "string"
    for name in presets.preset_names():
        assert repr(name) in s["description"]
    for form in WIRE_FORMS:
        assert form in s["description"]
    assert "<callable>" not in s["description"]  # a function does not cross a wire


def test_a_preset_added_to_the_registry_reaches_the_schema(monkeypatch):
    table = {**presets._table(), "newduck": {"class": "microduck:MicroduckPolicy"}}
    monkeypatch.setattr(presets, "_table", lambda: table)

    def run(policy: PolicyLike) -> None: ...

    assert "'newduck'" in parameters(run, joints=())["properties"]["policy"]["description"]
