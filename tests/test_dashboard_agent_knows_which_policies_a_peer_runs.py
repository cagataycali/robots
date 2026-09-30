"""The dashboard agent knows which policies each robot on the mesh can run.

Asked to make a Unitree G1 walk, the agent ran ``mock``: the peer's tool offered
``execute`` with a free-text ``policy_provider`` and nothing said which
providers apply to which robot, what each needs, or that ``mock`` is a sine
test. Now every ``fleet`` row carries ``policies`` (the robot the peer is and
the providers it can run, with the kwargs the wire carries, typed and bounded),
the system prompt says to choose from it, and the wire carries what a
whole-body controller needs: ``model_path`` (spelled ``checkpoint`` by the
provider), ``walk`` as a real boolean, ``target_velocity`` inside the
locomotion envelope. Anything else is refused on the robot host.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.dashboard import agent_console, peer_policies, peer_tools, routes_record
from strands_robots.mesh import Mesh, security
from strands_robots.policies.factory import _resolve_policy_class
from strands_robots.registry.policies import get_policy_provider, list_policy_providers
from tests._docs_hooks import docs_hook

G1_CHILD = ("g1-sim-1__unitree_g1", {"presence": {"robot_type": "sim", "parent": "g1-sim-1"}, "state": {"joints": {}}})
SO101_CHILD = ("lane__so101", {"presence": {"robot_type": "sim", "parent": "lane"}})
REAL_ARM = ("arm-1", {"presence": {"robot_type": "robot", "hw": "feetech"}, "state": {"joints": {"j1": 0.0}}})


def _providers(row: dict[str, Any]) -> list[str]:
    return [p["provider"] for p in row["can_run"]]


# -- which robot a peer is -------------------------------------------------------------------------


def test_the_robot_is_read_from_the_child_id_the_sim_world_the_managed_table_or_the_driver() -> None:
    assert peer_policies.robot_of_peer(*G1_CHILD) == "unitree_g1"
    assert peer_policies.robot_of_peer("g1-sim-1", {"presence": {"sim_robots": ["unitree_g1"]}}) == "unitree_g1"
    assert peer_policies.robot_of_peer("g1-sim-1", {"presence": {"sim_robots": ["a", "b"]}}) is None
    managed = [{"peer_id": "spawned-1", "robot_name": "franka_panda"}]
    assert peer_policies.robot_of_peer("spawned-1", {"presence": {}}, managed) == "franka_panda"
    assert peer_policies.robot_of_peer(*REAL_ARM) == "feetech"
    assert peer_policies.robot_of_peer("host-1", {"presence": {"robot_type": "robot"}}) is None


# -- which providers apply -------------------------------------------------------------------------


def test_a_g1_is_offered_the_whole_body_controllers_first_and_mock_last() -> None:
    row = peer_policies.peer_policies(*G1_CHILD)
    names = _providers(row)
    assert row["robot"] == "unitree_g1"
    assert names[:2] == ["wbc", "wbc_gait"], names
    assert names[-1] == "mock"
    wbc = row["can_run"][0]
    assert wbc["requires"] == ["model_path"], "wbc loads nothing without a checkpoint directory"
    assert set(wbc["kwargs"]) == {"model_path", "walk", "target_velocity"}
    assert wbc["kwargs"]["walk"]["type"] == "bool"
    bounds = wbc["kwargs"]["target_velocity"]["bounds"]
    assert bounds["vx"] == [-2.0, 2.0] and bounds["wz"] == [-2.0, 2.0]


def test_an_arm_is_never_offered_a_humanoid_controller() -> None:
    for peer in (SO101_CHILD, REAL_ARM):
        names = _providers(peer_policies.peer_policies(*peer))
        assert "wbc" not in names and "wbc_gait" not in names and "microduck" not in names
        assert "lerobot_local" in names and names[-1] == "mock"


def test_an_unknown_robot_is_offered_only_the_unbound_providers() -> None:
    names = _providers(peer_policies.peer_policies("host-1", {"presence": {"robot_type": "robot"}}))
    assert names and not (set(names) & set(peer_policies.EMBODIMENT_BOUND))


def test_removed_and_server_only_providers_are_not_offered() -> None:
    offered = set(_providers(peer_policies.peer_policies(*G1_CHILD)))
    assert not (offered & peer_policies.NOT_OFFERED)


def test_every_offered_provider_is_in_the_registry_with_a_hint() -> None:
    registry = set(list_policy_providers())
    for peer in (G1_CHILD, SO101_CHILD):
        for entry in peer_policies.peer_policies(*peer)["can_run"]:
            assert entry["provider"] in registry
            assert entry["hint"].strip(), f"{entry['provider']} has no line for the agent"


def test_the_embodiment_table_agrees_with_the_docs_coverage_witnesses() -> None:
    """One truth about which provider loads which body; the docs hook is loaded once per process by tests._docs_hooks."""
    module = docs_hook("coverage")
    witnesses = {provider: tuple(robots) for provider, robots, _path, _needle in module.EMBODIMENT_WITNESSES}
    assert peer_policies.EMBODIMENT_BOUND == witnesses


# -- the wire carries what the offered kwargs promise ----------------------------------------------


def test_every_offered_kwarg_is_one_the_wire_validates() -> None:
    wire = set(routes_record.WIRE_POLICY_CONFIG_KEYS) | {"target_velocity"}
    for peer in (G1_CHILD, SO101_CHILD):
        for entry in peer_policies.peer_policies(*peer)["can_run"]:
            unknown = set(entry["kwargs"]) - wire
            assert not unknown, f"{entry['provider']} offers {sorted(unknown)}, which the wire would refuse"
            assert set(entry["requires"]) <= set(entry["kwargs"]) | {"pretrained_name_or_path", "server_address"}


@pytest.mark.parametrize("kind", [peer_tools.KIND_SIM, peer_tools.KIND_REAL])
def test_both_proxy_rails_forward_the_whole_body_kwargs_and_the_robot_host_accepts_them(kind: str) -> None:
    tool_input = {
        "action": "execute",
        "instruction": "walk forward",
        "policy_provider": "wbc",
        "model_path": "/ckpt/grootwbc-g1",
        "walk": True,
        "target_velocity": [0.5, 0.0, 0.0],
        "not_a_wire_key": 1,
    }
    cmd, error = peer_tools.map_invocation("g1-sim-1__unitree_g1", kind, tool_input)
    assert error is None and cmd is not None
    assert (
        cmd["model_path"] == "/ckpt/grootwbc-g1" and cmd["walk"] is True and cmd["target_velocity"] == [0.5, 0.0, 0.0]
    )
    assert "not_a_wire_key" not in cmd
    validated = security.validate_command(cmd)
    assert validated["walk"] is True and validated["model_path"] == "/ckpt/grootwbc-g1"
    assert validated["target_velocity"] == [0.5, 0.0, 0.0]


def test_the_proxy_schemas_offer_the_same_policy_keys_the_rails_forward() -> None:
    for kind, tool_name in ((peer_tools.KIND_SIM, "g1"), (peer_tools.KIND_REAL, "arm")):
        spec = peer_tools.peer_tool_spec("p", kind, tool_name)
        assert spec is not None
        assert set(peer_tools._POLICY_FIELDS) <= set(spec["inputSchema"]["json"]["properties"])


def test_walk_crosses_the_wire_only_as_a_boolean() -> None:
    base = {"action": "execute", "instruction": "walk", "policy_provider": "wbc"}
    assert security.validate_command({**base, "walk": False})["walk"] is False
    for bad in ("false", "true", 1, 0, None, [True]):
        with pytest.raises(security.ValidationError, match="walk must be a bool"):
            security.validate_command({**base, "walk": bad})


def test_a_velocity_outside_the_locomotion_envelope_is_refused_not_clamped() -> None:
    base = {"action": "execute", "instruction": "walk", "policy_provider": "wbc", "walk": True}
    with pytest.raises(security.ValidationError):
        security.validate_command({**base, "target_velocity": [5.0, 0.0, 0.0]})
    with pytest.raises(security.ValidationError):
        security.validate_command({**base, "target_velocity": [0.0, 0.0, 9.0]})


def test_the_dispatcher_hands_walk_to_the_policy_constructor_and_the_velocity_to_every_call() -> None:
    class _FakeSim:
        def __init__(self) -> None:
            self._world: Any = object()
            self.calls: list[dict[str, Any]] = []
            self.tool_name_str = "fakesim"

        def list_robots(self) -> list[str]:
            return ["unitree_g1"]

        def run_policy(self, robot_name: str, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            return {"status": "success", "content": [{"text": robot_name}]}

    sim = _FakeSim()
    mesh = Mesh(sim, peer_id="g1-sim-1")
    mesh._dispatch(
        {
            "action": "execute",
            "instruction": "walk forward",
            "policy_provider": "wbc",
            "model_path": "/ckpt/grootwbc-g1",
            "walk": False,
            "target_velocity": [0.5, 0.0, 0.0],
        }
    )
    kwargs = sim.calls[0]
    assert kwargs["policy_config"] == {"model_path": "/ckpt/grootwbc-g1", "walk": False}
    assert kwargs["policy_kwargs"] == {"target_velocity": [0.5, 0.0, 0.0]}


# -- the provider receives the checkpoint under its own name --------------------------------------


def test_the_wire_model_path_becomes_the_checkpoint_a_whole_body_controller_declares() -> None:
    for provider in ("wbc", "wbc_gait"):
        keys = set((get_policy_provider(provider) or {})["config_keys"])
        assert "checkpoint" in keys and "model_path" not in keys, "premise: the provider spells it checkpoint"
        _name, _cls, kwargs = _resolve_policy_class(provider, model_path="/ckpt/grootwbc-g1", walk=True)
        assert kwargs == {"checkpoint": "/ckpt/grootwbc-g1", "walk": True}
    # an explicit checkpoint wins; a provider that declares model_path keeps it
    _n, _c, kwargs = _resolve_policy_class("wbc", model_path="/a", checkpoint="/b")
    assert kwargs == {"checkpoint": "/b"}
    _n, _c, kwargs = _resolve_policy_class("mock", model_path="/a")
    assert kwargs == {"model_path": "/a"}


# -- the agent is told ------------------------------------------------------------------------------


def test_the_fleet_row_carries_the_policies_block_and_the_prompt_points_at_it() -> None:
    row = agent_console.peer_summary(*G1_CHILD)
    assert row["policies"]["robot"] == "unitree_g1"
    assert _providers(row["policies"])[0] == "wbc"
    prompt = agent_console.SYSTEM_PROMPT
    assert "can_run" in prompt and "`wbc`" in prompt and "target_velocity" in prompt
    assert "mock" in prompt and "only when the operator asks" in prompt
    assert "ask for it" in prompt, "a missing checkpoint is a question to the operator, not a fallback to mock"
