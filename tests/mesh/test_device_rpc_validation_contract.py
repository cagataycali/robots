"""Security contracts for ``security.validate_device_rpc``.

``validate_device_rpc`` is the sanitisation boundary for Device Connect
*native* RPC calls (e.g. a Reachy's ``nod`` / ``look`` / ``playMove``)
invoked directly via the ``robot_mesh`` tool's ``rpc`` action. Unlike
:func:`~strands_robots.mesh.security.validate_command`, it deliberately
does NOT enforce the SO-100/SO-101 policy allowlist - an arbitrary device
legitimately advertises its own function set. But the function name and
params still flow into the device runtime, RPC subjects, and audit logs,
so it MUST enforce defence-in-depth: an identifier-safe charset on the
function name and on every param key, a length cap, a JSON-object shape,
JSON-serialisability, and a bounded encoded size.

These tests pin each guard. Because the function had no direct coverage,
they are written against the documented contract (return shape + every
``ValidationError`` branch) so a regression that loosens any check -
e.g. accepting a function name with shell metacharacters, an oversized
params blob, or a non-string key - fails here loudly.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import security as sec


class TestValidDeviceRpc:
    """Happy paths return a sanitised ``(function, params_dict)`` copy."""

    def test_function_only_returns_empty_params(self) -> None:
        func, params = sec.validate_device_rpc("nod")
        assert func == "nod"
        assert params == {}

    def test_none_params_returns_empty_dict(self) -> None:
        func, params = sec.validate_device_rpc("look", None)
        assert func == "look"
        assert params == {}

    def test_valid_params_returned_as_copy(self) -> None:
        original = {"angle": 30, "speed": 0.5, "loop": True}
        func, params = sec.validate_device_rpc("playMove", original)
        assert func == "playMove"
        assert params == original
        # A defensive copy: mutating the result must not touch the input.
        params["angle"] = 999
        assert original["angle"] == 30

    def test_underscore_leading_function_allowed(self) -> None:
        func, params = sec.validate_device_rpc("_private_fn")
        assert func == "_private_fn"
        assert params == {}

    def test_function_at_length_cap_allowed(self) -> None:
        name = "a" * sec.MAX_DC_RPC_FUNC_LEN
        func, _ = sec.validate_device_rpc(name)
        assert func == name


class TestFunctionNameRejected:
    """The function name is an identifier-safe, length-capped string."""

    @pytest.mark.parametrize("bad", ["", None, 123, b"nod", []])
    def test_non_empty_string_required(self, bad: object) -> None:
        with pytest.raises(sec.ValidationError, match="non-empty function name"):
            sec.validate_device_rpc(bad)  # type: ignore[arg-type]

    def test_over_length_cap_rejected(self) -> None:
        too_long = "a" * (sec.MAX_DC_RPC_FUNC_LEN + 1)
        with pytest.raises(sec.ValidationError, match="MAX_DC_RPC_FUNC_LEN"):
            sec.validate_device_rpc(too_long)

    @pytest.mark.parametrize(
        "bad",
        [
            "nod;rm -rf",  # shell metacharacter
            "look space",  # whitespace
            "play.Move",  # dot
            "../escape",  # path traversal
            "fn$VAR",  # shell var
            "1leading",  # leading digit
            "na\nme",  # control char / newline
            "na\x00me",  # NUL
        ],
    )
    def test_charset_violations_rejected(self, bad: str) -> None:
        with pytest.raises(sec.ValidationError, match="must match"):
            sec.validate_device_rpc(bad)


class TestParamsRejected:
    """Params must be a JSON object with identifier-safe keys, bounded size."""

    @pytest.mark.parametrize("bad", [[1, 2, 3], "string", 42, True])
    def test_non_dict_params_rejected(self, bad: object) -> None:
        with pytest.raises(sec.ValidationError, match="JSON object"):
            sec.validate_device_rpc("nod", bad)

    @pytest.mark.parametrize("bad_key", ["", 123, None])
    def test_non_string_or_empty_key_rejected(self, bad_key: object) -> None:
        with pytest.raises(sec.ValidationError, match="keys must be non-empty strings"):
            sec.validate_device_rpc("nod", {bad_key: 1})

    def test_over_length_key_rejected(self) -> None:
        long_key = "k" * (sec.MAX_DC_RPC_FUNC_LEN + 1)
        with pytest.raises(sec.ValidationError, match="params key length"):
            sec.validate_device_rpc("nod", {long_key: 1})

    @pytest.mark.parametrize("bad_key", ["has space", "dot.key", "../trav", "k;rm"])
    def test_charset_violating_key_rejected(self, bad_key: str) -> None:
        with pytest.raises(sec.ValidationError, match="params key"):
            sec.validate_device_rpc("nod", {bad_key: 1})

    def test_non_json_serialisable_value_rejected(self) -> None:
        with pytest.raises(sec.ValidationError, match="not JSON-serialisable"):
            sec.validate_device_rpc("nod", {"obj": object()})

    def test_oversized_params_rejected(self) -> None:
        # A single string value whose JSON encoding exceeds the byte cap.
        big = {"blob": "x" * (sec.MAX_DC_RPC_PARAMS_BYTES + 1)}
        with pytest.raises(sec.ValidationError, match="encoded size"):
            sec.validate_device_rpc("nod", big)


class TestPublicSurface:
    """The validator and its bounds are part of the documented public API."""

    def test_validator_exported(self) -> None:
        assert "validate_device_rpc" in sec.__all__
        assert "MAX_DC_RPC_FUNC_LEN" in sec.__all__
        assert "MAX_DC_RPC_PARAMS_BYTES" in sec.__all__
