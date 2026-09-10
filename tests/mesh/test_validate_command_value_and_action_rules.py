"""Value and action rules of the two wire validators, pinned against the code.

:func:`~strands_robots.mesh.security.validate_input_frame` refuses a ``bool``
(a Python or numpy one) rather than read it as ``1.0``, refuses a numeric-looking
string and a non-finite value, and re-reads its envelope per call so an operator
can narrow it without a restart. :func:`~strands_robots.mesh.security.validate_command`
holds the teleop identifiers to the ``[A-Za-z0-9_.-]+`` charset, admits a null
``teleop_stop.device_name``, and bounds ``resume.override_code`` by type, length
and charset with an empty default.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from strands_robots.mesh import security


def _refused(value: Any) -> str | None:
    """Drive the real validator with *value*; return its refusal, or ``None``."""
    try:
        security.validate_input_frame({"j1": value})
    except security.ValidationError as refusal:
        return str(refusal)
    return None


class TestTheInputFrameValueDomain:
    """Which values :func:`validate_input_frame` admits, and which it refuses."""

    @pytest.mark.parametrize("value", [True, False])
    def test_a_python_bool_is_refused_rather_than_coerced(self, value: bool) -> None:
        refusal = _refused(value)
        assert refusal is not None and "not bool" in refusal, refusal

    def test_a_numpy_bool_is_refused_too(self) -> None:
        numpy = pytest.importorskip("numpy")
        refusal = _refused(numpy.bool_(True))
        assert refusal is not None and "not bool" in refusal, refusal

    def test_a_numeric_looking_string_is_refused(self) -> None:
        """The type is checked, not the value's coercibility."""
        assert float("1.0") == 1.0, "premise: the string is coercible to float"
        refusal = _refused("1.0")
        assert refusal is not None and "must be numeric" in refusal, refusal

    @pytest.mark.parametrize("value", [1, 1.5, -1.5, 0])
    def test_an_int_or_float_inside_the_envelope_is_admitted(self, value: float) -> None:
        assert _refused(value) is None

    def test_a_numpy_float_is_unwrapped_and_admitted(self) -> None:
        numpy = pytest.importorskip("numpy")
        assert _refused(numpy.float32(1.5)) is None

    def test_a_non_finite_value_is_refused(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            refusal = _refused(value)
            assert refusal is not None and "finite" in refusal, (value, refusal)

    def test_an_empty_key_is_refused(self) -> None:
        with pytest.raises(security.ValidationError, match="key length out of range"):
            security.validate_input_frame({"": 1.0})

    def test_the_envelope_is_re_read_per_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The narrowed envelope refuses a value the wider one admitted."""
        assert _refused(50.0) is None, "premise: 50.0 is inside the default envelope"
        monkeypatch.setenv("STRANDS_MESH_INPUT_VALUE_ABS", "10.0")
        refusal = _refused(50.0)
        assert refusal is not None and "out of range" in refusal, refusal


class TestTheCommandActionRules:
    """The per-action identifier and override-code rules of a mesh command."""

    @pytest.mark.parametrize("identifier", ["**", "*", "a/b", "with space", "semi;colon"])
    def test_a_teleop_source_peer_id_outside_the_charset_is_refused(self, identifier: str) -> None:
        with pytest.raises(security.ValidationError, match=r"\[A-Za-z0-9_\.-\]\+"):
            security.validate_command({"action": "teleop_receive", "source_peer_id": identifier})

    def test_a_teleop_device_name_outside_the_charset_is_refused(self) -> None:
        with pytest.raises(security.ValidationError, match="device_name"):
            security.validate_command({"action": "teleop_receive", "source_peer_id": "leader", "device_name": "**"})

    def test_an_identifier_inside_the_charset_is_admitted(self) -> None:
        out = security.validate_command(
            {"action": "teleop_receive", "source_peer_id": "arm-1.left", "device_name": "leader_2"}
        )
        assert out["source_peer_id"] == "arm-1.left"
        assert out["device_name"] == "leader_2"

    @pytest.mark.parametrize("device", [123, [], {}, 1.5])
    def test_a_non_string_teleop_stop_device_name_is_refused(self, device: Any) -> None:
        with pytest.raises(security.ValidationError, match="must be a string or null"):
            security.validate_command({"action": "teleop_stop", "device_name": device})

    def test_a_null_teleop_stop_device_name_is_admitted(self) -> None:
        assert security.validate_command({"action": "teleop_stop", "device_name": None})["device_name"] is None

    @pytest.mark.parametrize("code", [123, [], {}, True])
    def test_a_non_string_override_code_is_refused(self, code: Any) -> None:
        with pytest.raises(security.ValidationError, match="must be a string"):
            security.validate_command({"action": "resume", "override_code": code})

    def test_an_override_code_past_the_length_bound_is_refused(self) -> None:
        over = "a" * (security.MAX_OVERRIDE_CODE_LEN + 1)
        with pytest.raises(security.ValidationError, match="too long"):
            security.validate_command({"action": "resume", "override_code": over})

    @pytest.mark.parametrize("code", ["ok\ninjected", "ok\r\ninjected", "nul\x00byte", "bell\x07"])
    def test_an_override_code_carrying_a_control_character_is_refused(self, code: str) -> None:
        with pytest.raises(security.ValidationError, match="control characters"):
            security.validate_command({"action": "resume", "override_code": code})

    def test_a_printable_override_code_at_the_length_bound_is_admitted(self) -> None:
        at_bound = "a" * security.MAX_OVERRIDE_CODE_LEN
        assert security.validate_command({"action": "resume", "override_code": at_bound})["override_code"] == at_bound

    def test_an_omitted_override_code_defaults_to_empty(self) -> None:
        assert security.validate_command({"action": "resume"})["override_code"] == ""
