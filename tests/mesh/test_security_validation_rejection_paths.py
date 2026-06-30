"""Wire-boundary rejection and accept contracts for ``mesh.security``.

The mesh validator is the single chokepoint between an authenticated peer's
payload and the robot's control surface, so every agent-supplied string must be
shape-checked there rather than relying on downstream consumers to be safe
(AGENTS.md > Review Learnings #92). These pin the type / length / charset guards
on the allowlist helpers and the ``validate_command`` fields that prior tests
left unexercised:

* :func:`~strands_robots.mesh.security.is_safe_model_path` type, length, and
  ``hf_only`` ``<org>/<repo>`` allowlist branches;
* :func:`~strands_robots.mesh.security.is_safe_policy_type` /
  :func:`~strands_robots.mesh.security.is_safe_policy_provider` type guards;
* ``execute`` payload fields: the accepted ``model_path``, the ``target_joints`` per-key length cap, and the
  ``world_update`` JSON-serialisability bound;
* the optional ``device_name`` on ``teleop_receive`` (accept) and
  ``teleop_stop`` (type guard + accept), neither of which had an accept-path
  test.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import security as sec


def _execute(**fields: object) -> dict[str, object]:
    """A minimal-valid ``execute`` payload to layer optional fields onto."""
    return {"action": "execute", "instruction": "task", "policy_provider": "mock", **fields}


# --- is_safe_model_path -------------------------------------------------


class TestSafeModelPathGuards:
    @pytest.mark.parametrize("bad", [None, 42, "", b"nvidia/x"])
    def test_non_string_or_empty_rejected(self, bad):
        assert sec.is_safe_model_path(bad) is False  # type: ignore[arg-type]

    def test_oversized_path_rejected(self):
        assert sec.is_safe_model_path("a" * (sec.MAX_MODEL_PATH_LEN + 1)) is False

    def test_hf_only_accepts_org_repo_allowlist_entry(self, monkeypatch):
        # An operator-supplied "<org>/<repo>" allowlist entry matches the exact
        # repo under hf_only (the "/"-in-entry branch), distinct from the
        # bare-org default entries.
        monkeypatch.setenv("STRANDS_MESH_HF_REPO_ALLOW", "myorg/myrepo")
        assert sec.is_safe_model_path("myorg/myrepo", hf_only=True) is True
        # A different repo under the same org is NOT whitelisted by the
        # repo-scoped entry.
        assert sec.is_safe_model_path("myorg/other", hf_only=True) is False


# --- policy type / provider allowlists ----------------------------------


class TestPolicyTypeProviderGuards:
    @pytest.mark.parametrize("bad", [None, 42, ""])
    def test_policy_type_non_string_or_empty_rejected(self, bad):
        assert sec.is_safe_policy_type(bad) is False  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [None, 42, ""])
    def test_policy_provider_non_string_or_empty_rejected(self, bad):
        assert sec.is_safe_policy_provider(bad) is False  # type: ignore[arg-type]


# --- validate_command execute-payload fields ----------------------------


class TestExecutePayloadFields:
    def test_safe_model_path_accepted(self):
        out = sec.validate_command(_execute(model_path="nvidia/gr00t-n1.5"))
        assert out["model_path"] == "nvidia/gr00t-n1.5"

    def test_target_joints_key_over_peer_id_len_rejected(self):
        joints = {"j" * (sec.MAX_PEER_ID_LEN + 1): 0.0}
        with pytest.raises(sec.ValidationError, match="MAX_PEER_ID_LEN"):
            sec.validate_command(_execute(target_joints=joints))

    def test_world_update_non_serialisable_rejected(self):
        with pytest.raises(sec.ValidationError, match="not JSON-serialisable"):
            sec.validate_command(_execute(world_update={"bad": {1, 2, 3}}))


# --- teleop device_name (accept + teleop_stop type guard) ---------------


class TestTeleopDeviceName:
    def test_teleop_receive_valid_device_name_accepted(self):
        out = sec.validate_command(
            {"action": "teleop_receive", "source_peer_id": "operator-1", "device_name": "leader"}
        )
        assert out["device_name"] == "leader"

    def test_teleop_receive_non_string_device_name_rejected(self):
        with pytest.raises(sec.ValidationError, match="device_name must be a string"):
            sec.validate_command({"action": "teleop_receive", "source_peer_id": "operator-1", "device_name": 123})

    def test_teleop_stop_valid_device_name_accepted(self):
        out = sec.validate_command({"action": "teleop_stop", "device_name": "follower"})
        assert out["device_name"] == "follower"

    def test_teleop_stop_none_device_name_accepted(self):
        out = sec.validate_command({"action": "teleop_stop", "device_name": None})
        assert out["device_name"] is None

    def test_teleop_stop_non_string_device_name_rejected(self):
        with pytest.raises(sec.ValidationError, match="string or null"):
            sec.validate_command({"action": "teleop_stop", "device_name": ["leader"]})


# --- teleop input safety-bound env resolver -----------------------------


class TestInputValueAbsEnvFallback:
    """A misconfigured ``STRANDS_MESH_INPUT_VALUE_ABS`` falls back to the safe
    default rather than disabling the teleop input-magnitude bound."""

    def test_non_numeric_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_VALUE_ABS", "not-a-number")
        assert sec._input_value_abs() == sec.DEFAULT_INPUT_VALUE_ABS

    @pytest.mark.parametrize("bad", ["0", "-5", "inf", "nan"])
    def test_non_positive_or_non_finite_env_falls_back_to_default(self, monkeypatch, bad):
        monkeypatch.setenv("STRANDS_MESH_INPUT_VALUE_ABS", bad)
        assert sec._input_value_abs() == sec.DEFAULT_INPUT_VALUE_ABS
