"""An unflyable setpoint is refused by name, never clamped and never sent.

``cflib`` imposes no ceiling on a setpoint and the firmware attempts whatever
arrives, so the flight envelope is this driver's to enforce. It refuses rather
than clamping, and the distinction is the whole point of the file: an operator
who asked for 5 m/s and silently got 1 m/s plans the next command around a speed
the aircraft never flew, and the error only surfaces as a position discrepancy
metres later. A refusal naming the bound leaves the caller's model correct.

The rows cover the two ways a number is unusable - outside the envelope, and not
a number at all - because they fail differently: an over-speed is a valid float
the firmware would honour, while ``nan`` serialises into a CRTP packet as a
perfectly well-formed float64 that poisons the controller's state.

A key the driver does not command is the third way, and the quietest of the
three. An unusable *value* is at least read before it is judged; an unusable
*key* is dropped before any check sees it, and the component it named then rests
at zero - indistinguishable, downstream, from a caller who never asked for it.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.drivers.crazyflie import (
    ACTION_KEYS,
    MAX_HEIGHT,
    MAX_HORIZONTAL_SPEED,
    MAX_VERTICAL_SPEED,
    MAX_YAW_RATE,
    MIN_HEIGHT,
    action_to_setpoint,
    twist_envelope,
    twist_error,
)


class TestTheEnvelopeIsReportedAsItIsEnforced:
    """The discovery surface and the check cannot disagree."""

    def test_every_enforced_bound_is_reported(self) -> None:
        assert twist_envelope() == {
            "max_horizontal_speed": MAX_HORIZONTAL_SPEED,
            "max_vertical_speed": MAX_VERTICAL_SPEED,
            "max_yaw_rate": MAX_YAW_RATE,
            "min_height": MIN_HEIGHT,
            "max_height": MAX_HEIGHT,
        }

    def test_the_hover_band_is_a_band(self) -> None:
        """Non-vacuity: a min above a max would accept nothing and pass silently."""
        assert 0.0 < MIN_HEIGHT < MAX_HEIGHT


class TestAValueOnTheBoundIsFlyable:
    """The bound is inclusive, so the reported ceiling is reachable."""

    @pytest.mark.parametrize(
        "twist",
        [
            {"vx": MAX_HORIZONTAL_SPEED},
            {"vx": -MAX_HORIZONTAL_SPEED},
            {"vy": MAX_HORIZONTAL_SPEED},
            {"vz": MAX_VERTICAL_SPEED},
            {"wz": MAX_YAW_RATE},
            {"wz": -MAX_YAW_RATE},
            {"z": MIN_HEIGHT},
            {"z": MAX_HEIGHT},
        ],
    )
    def test_the_reported_ceiling_is_accepted(self, twist: dict[str, float]) -> None:
        assert not isinstance(action_to_setpoint(twist), str), (
            f"{twist} sits exactly on a bound twist_envelope() advertises; refusing it would "
            "make the advertised ceiling unreachable"
        )


class TestAValueOutsideTheEnvelopeIsRefusedByName:
    """Every refusal names the parameter, the value and the bound."""

    @pytest.mark.parametrize(
        ("twist", "param"),
        [
            ({"vx": 5.0}, "vx"),
            ({"vx": -5.0}, "vx"),
            ({"vy": 1.01}, "vy"),
            ({"vz": 0.6}, "vz"),
            ({"wz": 10.0}, "wz"),
            ({"z": 0.01}, "z"),
            ({"z": 3.0}, "z"),
        ],
    )
    def test_the_reason_names_the_parameter_and_a_bound(self, twist: dict[str, float], param: str) -> None:
        reason = action_to_setpoint(twist)
        assert isinstance(reason, str), f"{twist} is outside the envelope and must be refused"
        assert param in reason
        assert "twist_envelope()" in reason, "a refused caller needs the discovery surface"

    @pytest.mark.parametrize(
        ("twist", "param"),
        [
            ({"vx": float("nan")}, "vx"),
            ({"vx": float("inf")}, "vx"),
            ({"wz": float("-inf")}, "wz"),
            ({"vy": "0.5"}, "vy"),
            ({"vz": None}, "vz"),
            ({"z": float("nan")}, "z"),
            ({"z": -0.5}, "z"),
        ],
    )
    def test_a_value_that_is_not_a_usable_number_is_refused(self, twist: dict[str, Any], param: str) -> None:
        """``nan`` is a valid float64 on the wire, so the door is the only guard."""
        reason = action_to_setpoint(twist)
        assert isinstance(reason, str), f"{twist} is not a usable number and must be refused"
        assert param in reason

    def test_the_yaw_rate_bound_is_quoted_in_radians(self) -> None:
        """The refusal must speak the caller's unit, not the wire's.

        Quoting 143 deg/s at a caller who passed rad/s hands them a number they
        then have to convert back to check their own command.
        """
        reason = twist_error(0.0, 0.0, 99.0, context="set_twist")
        assert reason is not None
        assert "rad/s" in reason
        assert str(MAX_YAW_RATE) in reason
        assert "deg" not in reason


class TestNothingIsClamped:
    """A refusal, not a silently reduced command."""

    def test_an_over_speed_command_produces_no_setpoint_at_all(self) -> None:
        translated = action_to_setpoint({"vx": 5.0, "z": 0.5})
        assert isinstance(translated, str), (
            "clamping to the ceiling would return a setpoint here, and the caller would fly "
            "1 m/s while believing they commanded 5"
        )

    def test_the_first_offender_is_the_one_reported(self) -> None:
        """One reason at a time, in a stated order, so the message stays readable."""
        reason = action_to_setpoint({"vx": 5.0, "wz": 99.0})
        assert isinstance(reason, str)
        assert "vx" in reason


class TestAKeyTheDriverCannotCarryIsRefusedByName:
    """The key half of the same rule: a named component is flown, or refused.

    Every flight component is read with a zero default, so a key this driver
    does not know is dropped in silence. The gate for an action naming *no*
    known key exists precisely because an all-zero setpoint "would latch the
    aircraft into a hover the caller never asked for" - and that outcome is
    reachable straight through the gate, because a single known key satisfies
    it. ``{"vx": 0.0, "height": 1.5}`` asks to hover at 1.5 m using this
    module's own internal spelling of ``z``, and produces the resting setpoint.
    """

    @pytest.mark.parametrize(
        ("action", "dropped"),
        [
            ({"vx": 0.0, "height": 1.5}, "height"),  # this module's own name for z
            ({"wz": 0.0, "altitude": 1.5}, "altitude"),
            ({"vx": 0.5, "vyaw": 2.0}, "vyaw"),  # the yaw spelling robotd uses
            ({"z": 1.0, "vX": 0.6}, "vX"),  # a capitalisation nothing downstream reads
            ({"vx": 0.2, "z": 0.5, "gripper": 1.0}, "gripper"),
        ],
    )
    def test_a_dropped_component_is_refused_not_rested(self, action: dict[str, Any], dropped: str) -> None:
        reason = action_to_setpoint(action)
        assert isinstance(reason, str), (
            f"{action} names {dropped!r}, which no setpoint carries; returning a setpoint here "
            "flies a command the caller never gave and reports success for it"
        )
        assert dropped in reason, "the refusal must name the key that would have been dropped"
        for key in ACTION_KEYS:
            assert key in reason, f"the refusal must name {key!r} as an accepted alternative"

    def test_nothing_reaches_the_commander(self, connected, recorder) -> None:  # type: ignore[no-untyped-def]
        """The envelope door refuses too, so no partial command is latched.

        The flyable keys here are flyable, so a driver that refused only the
        whole action would still have to decide what to do with them; it must
        send nothing and keep no repeater alive.
        """
        driver, _fake, connect_reason = connected()
        assert connect_reason is None, connect_reason

        envelope = driver.send_action({"vx": 0.2, "z": 0.5, "height": 1.5})

        assert envelope["status"] == "error", envelope
        assert "height" in str(envelope)
        assert recorder.count("commander.send_hover_setpoint") == 0
        assert recorder.count("commander.send_velocity_world_setpoint") == 0
