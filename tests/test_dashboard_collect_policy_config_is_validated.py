"""``POST /api/collect`` forwards a ``policy_config`` the wire would have accepted, nothing else.

The route handed ``policy_provider`` and ``policy_config`` straight to a child process that
calls ``run_policy`` with them, so ``create_policy`` received whatever the page sent: a
provider off the allowlist, a ``model_path`` anywhere on the disk (the loader's own error then
says what is there), a ``server_address`` to any host (an outbound dial from the operator's
machine). The same payload arriving over the mesh goes through
``mesh.security.validate_command`` first, which allowlists the provider, the policy type, the
policy host, the Hub reference and the model path, and drops every other key.

The route now runs the same validator over the same keys, contains ``model_path`` under the
checkpoint homes the dashboard already uses, refuses a key the wire does not know, and
forwards only what came back validated.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402

from strands_robots.dashboard import routes_record  # noqa: E402


@pytest.fixture
def homes(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("STRANDS_TRAIN_OUTPUT_DIR", str(out))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.delenv("STRANDS_MESH_POLICY_TYPE_ALLOW", raising=False)
    monkeypatch.delenv("STRANDS_MESH_POLICY_HOST_ALLOW", raising=False)
    monkeypatch.delenv("STRANDS_MESH_HF_REPO_ALLOW", raising=False)
    from strands_robots.mesh import security

    security._clear_security_caches_for_tests()
    yield {"out": out.resolve()}
    security._clear_security_caches_for_tests()


def _refusal(provider: Any, config: Any) -> HTTPException:
    with pytest.raises(HTTPException) as info:
        routes_record.contained_policy_request(provider, config)
    return info.value


def test_the_default_mock_request_passes_unchanged(homes):
    provider, config = routes_record.contained_policy_request("mock", None)
    assert provider == "mock"
    assert config is None


def test_a_provider_off_the_allowlist_is_refused(homes):
    exc = _refusal("evil_provider", None)
    assert exc.status_code == 422
    assert "policy_provider" in str(exc.detail)


def test_a_model_path_outside_every_checkpoint_home_is_refused(homes, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    exc = _refusal("lerobot_local", {"model_path": str(elsewhere)})
    assert exc.status_code in (400, 422)
    assert str(elsewhere) not in str(exc.detail), "the refusal does not echo the path"
    for word in ("exists", "not found", "no such"):
        assert word not in str(exc.detail).lower()


def test_a_model_path_with_traversal_is_refused(homes):
    exc = _refusal("lerobot_local", {"model_path": str(homes["out"] / ".." / ".." / "etc")})
    assert exc.status_code in (400, 422)


def test_a_model_path_inside_the_output_home_is_forwarded_folded(homes):
    ckpt = homes["out"] / "run" / "checkpoints" / "last" / "pretrained_model"
    provider, config = routes_record.contained_policy_request("lerobot_local", {"model_path": str(ckpt)})
    assert provider == "lerobot_local"
    assert config == {"model_path": str(ckpt)}


def test_a_server_address_off_the_host_allowlist_is_refused(homes):
    exc = _refusal("lerobot_local", {"server_address": "attacker.example:8000"})
    assert exc.status_code == 422
    assert "server_address" in str(exc.detail)


def test_a_policy_host_off_the_allowlist_is_refused(homes):
    exc = _refusal("lerobot_local", {"policy_host": "attacker.example"})
    assert exc.status_code == 422


def test_a_hub_reference_off_the_repo_allowlist_is_refused(homes):
    exc = _refusal("lerobot_local", {"pretrained_name_or_path": "attacker/weights"})
    assert exc.status_code == 422
    assert "pretrained_name_or_path" in str(exc.detail)


def test_a_policy_type_off_the_allowlist_is_refused(homes):
    exc = _refusal("lerobot_local", {"policy_type": "not_a_family"})
    assert exc.status_code == 422
    assert "policy_type" in str(exc.detail)


def test_a_key_the_wire_does_not_know_is_refused_not_dropped(homes):
    """Fail closed: a constructor kwarg the mesh would never forward is named, not silently lost."""
    exc = _refusal("mock", {"trust_remote_code": True})
    assert exc.status_code == 422
    assert "trust_remote_code" in str(exc.detail)


def test_a_non_object_policy_config_is_refused(homes):
    assert _refusal("mock", "model_path=/etc").status_code == 422
    assert _refusal("mock", ["x"]).status_code == 422


def test_the_forwarded_config_is_only_what_the_wire_validated(homes):
    provider, config = routes_record.contained_policy_request(
        "lerobot_local", {"pretrained_name_or_path": "lerobot/act_so101", "policy_type": "act"}
    )
    assert provider == "lerobot_local"
    assert config == {"pretrained_name_or_path": "lerobot/act_so101", "policy_type": "act"}


def test_the_keys_the_route_forwards_are_the_keys_the_mesh_forwards():
    """One vocabulary: the mesh dispatcher's ``extra`` tuple plus the two host knobs the wire grades."""
    assert set(routes_record.WIRE_POLICY_CONFIG_KEYS) == {
        "model_path",
        "server_address",
        "policy_type",
        "pretrained_name_or_path",
        "policy_host",
        "policy_port",
        "walk",
    }


def test_walk_is_carried_as_a_bool_and_nothing_else(homes):
    """The whole-body controllers' posture flag crosses the wire only as a real boolean."""
    _provider, config = routes_record.contained_policy_request("wbc", {"walk": False})
    assert config == {"walk": False}
    exc = _refusal("wbc", {"walk": "false"})
    assert exc.status_code == 422 and "walk must be a bool" in str(exc.detail)
