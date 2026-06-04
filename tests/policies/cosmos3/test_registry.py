"""Cosmos 3 registry + factory resolution tests."""

from strands_robots.policies import create_policy, list_providers
from strands_robots.policies.cosmos3 import Cosmos3Policy
from strands_robots.policies.cosmos3.client import Cosmos3WebsocketClient
from strands_robots.registry import get_policy_provider, list_policy_providers, resolve_policy


def test_cosmos3_in_registry():
    assert "cosmos3" in list_policy_providers()
    cfg = get_policy_provider("cosmos3")
    assert cfg["module"] == "strands_robots.policies.cosmos3"
    assert cfg["class"] == "Cosmos3Policy"


def test_shorthands_resolve_to_cosmos3():
    for name in ("cosmos3", "cosmos", "c3"):
        prov, _ = resolve_policy(name)
        assert prov == "cosmos3", name


def test_cosmos3_url_pattern():
    prov, _ = resolve_policy("cosmos3://localhost:8000")
    assert prov == "cosmos3"


def test_model_id_override_disambiguates_from_groot():
    # nvidia/Cosmos3-* must route to cosmos3, not groot
    prov, kwargs = resolve_policy("nvidia/Cosmos3-Nano-Policy-DROID")
    assert prov == "cosmos3"


def test_listed_in_providers():
    assert "cosmos3" in list_providers()


def test_create_policy_constructs_cosmos3():
    # No server needed: client connects lazily, construction must not touch network.
    p = create_policy("cosmos3", embodiment="droid", port=8123)
    assert isinstance(p, Cosmos3Policy)
    assert p.provider_name == "cosmos3"
    assert p.port == 8123
    assert isinstance(p._client, Cosmos3WebsocketClient)
