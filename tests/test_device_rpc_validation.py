"""Regression tests for device-native RPC validation (robot_mesh action='rpc').

Closes the gap where robot_mesh could discover a device's functions (e.g. the
Reachy's nod/look/playMove) but tell/send rejected them via the policy-action
ALLOWED_ACTIONS allowlist. The rpc path validates name+params WITHOUT that
allowlist, then invokes the device function directly.
"""

import pytest

from strands_robots.mesh import security


class TestValidateDeviceRpc:
    def test_accepts_bare_function(self):
        assert security.validate_device_rpc("nod") == ("nod", {})

    def test_accepts_function_with_params(self):
        assert security.validate_device_rpc("look", {"yaw": 30, "pitch": -15}) == (
            "look",
            {"yaw": 30, "pitch": -15},
        )

    def test_none_params_become_empty_dict(self):
        assert security.validate_device_rpc("happy", None) == ("happy", {})

    def test_not_gated_by_policy_allowlist(self):
        # The whole point: 'nod' is NOT in ALLOWED_ACTIONS but is a valid RPC.
        assert "nod" not in security.ALLOWED_ACTIONS
        func, _ = security.validate_device_rpc("nod")
        assert func == "nod"

    @pytest.mark.parametrize("bad", ["rm -rf /", "nod;reboot", "../escape", "a.b", "has space", "", "n@d"])
    def test_rejects_unsafe_function_names(self, bad):
        with pytest.raises(security.ValidationError):
            security.validate_device_rpc(bad)

    def test_rejects_overlong_function_name(self):
        with pytest.raises(security.ValidationError):
            security.validate_device_rpc("a" * (security.MAX_DC_RPC_FUNC_LEN + 1))

    def test_rejects_non_string_function(self):
        with pytest.raises(security.ValidationError):
            security.validate_device_rpc(123)  # type: ignore[arg-type]

    def test_rejects_non_dict_params(self):
        with pytest.raises(security.ValidationError):
            security.validate_device_rpc("nod", ["not", "a", "dict"])  # type: ignore[arg-type]

    def test_rejects_unsafe_param_keys(self):
        with pytest.raises(security.ValidationError):
            security.validate_device_rpc("look", {"bad key": 1})

    def test_rejects_oversize_params(self):
        big = {"k": "x" * (security.MAX_DC_RPC_PARAMS_BYTES + 10)}
        with pytest.raises(security.ValidationError):
            security.validate_device_rpc("playMove", big)

    def test_rejects_non_serialisable_params(self):
        with pytest.raises(security.ValidationError):
            security.validate_device_rpc("look", {"obj": object()})

    def test_returns_copy_of_params(self):
        src = {"yaw": 1}
        _, out = security.validate_device_rpc("look", src)
        out["yaw"] = 99
        assert src["yaw"] == 1  # original untouched
