"""The Microduck actuation flags are checked, never read by truthiness.

``active`` (a standing-pose posture flag) and ``enable_torque``'s ``on`` select a
*posture* / energise servos on physical hardware, so they must be rejected unless
they are a real boolean — the convention in ``AGENTS.md`` ("Posture flags are
checked, never read by truthiness"), with ``strands_robots.utils.boolean_flag_error``
as the shared domain. Every non-empty string is truthy, so ``"false"`` read by
``bool()`` would send ``active: true`` / energise the torque — silent wrong
actuation. These tests pin the domain over the spellings an operator reaches for
when opting out.
"""

from __future__ import annotations

import math

import pytest

from strands_robots.drivers.microduck import MicroduckDriver, action_to_wire

#: The opt-out spellings + numbers/None that truthiness reads as the wrong branch.
UNUSABLE = ["false", "no", "off", "0", 1, 0, math.nan, None, [], "true"]


class TestActivePostureFlag:
    """``action_to_wire`` refuses a non-boolean ``active`` before any frame is built."""

    @pytest.mark.parametrize("value", [True, False], ids=["True", "False"])
    def test_a_real_bool_is_accepted(self, value: bool) -> None:
        out = action_to_wire({"active": value, "z": 0.0})
        assert not isinstance(out, str), f"a real bool must not be refused, got {out!r}"

    @pytest.mark.parametrize("value", UNUSABLE, ids=[repr(v)[:12] for v in UNUSABLE])
    def test_a_non_bool_is_refused_naming_the_flag(self, value: object) -> None:
        out = action_to_wire({"active": value})
        assert isinstance(out, str) and "active" in out, f"expected a refusal naming 'active', got {out!r}"


class TestEnableTorqueFlag:
    """``enable_torque`` refuses a non-boolean ``on`` at the read, before touching the socket."""

    @pytest.mark.parametrize("value", UNUSABLE, ids=[repr(v)[:12] for v in UNUSABLE])
    def test_a_non_bool_on_is_refused_naming_the_flag(self, value: object) -> None:
        driver = MicroduckDriver(tool_name="microduck", port="/tmp/robotd-does-not-exist.sock")
        result = driver.enable_torque(on=value)  # type: ignore[arg-type]
        assert result["status"] == "error"
        assert "on" in result["content"][0]["text"]
