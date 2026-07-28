"""One rule decides which environment-held loop rates are usable.

Every mesh loop rate is operator-tunable through an environment variable, and
each reader has its own documented fallback for a value it cannot use: a sensor
loop keeps its built-in rate, the camera loop stays off, the teleop apply
ceiling reverts to its default. What none of them can use is a non-finite rate,
because each turns the rate into a period with ``1.0 / hz``: ``inf`` makes that
period zero, so a loop that meant to wait never waits, and ``nan`` makes it
compare ``False`` against every bound, so a cap built from it never trips.
``float()`` accepts ``"inf"`` and ``"nan"`` and overflows ``"1e999"`` to
``inf``, so all three reach the readers unless something refuses them.

These tests pin the shared decision in :func:`hz_from_env` and, more
importantly, pin that all three readers agree on it -- the drift this replaces
had the camera reader guarding non-finite input while the seven sensor loops
and the teleop apply ceiling, which compute the same period from the same kind
of value, did not.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import input as mesh_input
from strands_robots.mesh import sensors as mesh_sensors
from strands_robots.mesh.input import INPUT_MAX_HZ_DEFAULT
from strands_robots.mesh.session import hz_from_env

#: Values ``float()`` accepts that no ``1.0 / hz`` consumer can honor.
NON_FINITE = ["inf", "+inf", "-inf", "Infinity", "1e999", "nan", "NaN"]

_ENV = "STRANDS_MESH_TEST_RATE_HZ"


class TestHzFromEnv:
    """The shared decision: usable, absent, or unusable-with-a-reason."""

    def test_unset_reports_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_ENV, raising=False)
        assert hz_from_env(_ENV) == (None, None)

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_reports_absent(self, monkeypatch: pytest.MonkeyPatch, blank: str) -> None:
        monkeypatch.setenv(_ENV, blank)
        assert hz_from_env(_ENV) == (None, None)

    @pytest.mark.parametrize(("raw", "expected"), [("20", 20.0), ("2.5", 2.5), ("0", 0.0), ("-3", -3.0)])
    def test_finite_value_is_returned_verbatim(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: float
    ) -> None:
        """Sign and zero are the caller's decision, not this helper's."""
        monkeypatch.setenv(_ENV, raw)
        assert hz_from_env(_ENV) == (expected, None)

    def test_unparsable_value_reports_a_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_ENV, "not-a-number")
        hz, reason = hz_from_env(_ENV)
        assert hz is None
        assert reason is not None
        assert _ENV in reason and "not-a-number" in reason

    @pytest.mark.parametrize("raw", NON_FINITE)
    def test_non_finite_value_reports_a_reason(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(_ENV, raw)
        hz, reason = hz_from_env(_ENV)
        assert hz is None
        assert reason is not None
        assert _ENV in reason and raw in reason


class TestReadersAgreeOnWhatIsUsable:
    """No reader of an environment-held rate accepts a value it cannot pace with.

    Each reader keeps its own fallback, so the shared property asserted here is
    the weaker one that actually matters: whatever comes back must be a rate the
    ``1.0 / hz`` consumer downstream can use -- finite, and either positive or
    the documented "off" zero.
    """

    @pytest.mark.parametrize("raw", NON_FINITE)
    def test_sensor_loop_rate_falls_back_to_its_default(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(_ENV, raw)
        assert mesh_sensors._resolve_hz(_ENV, 12.5) == 12.5

    @pytest.mark.parametrize("raw", NON_FINITE)
    def test_camera_rate_stays_off(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        from strands_robots.mesh.core import Mesh

        monkeypatch.setenv("STRANDS_MESH_CAMERA_HZ", raw)
        mesh = Mesh(object(), peer_id="rate-parity")
        assert mesh._resolve_camera_hz() == 0.0

    @pytest.mark.parametrize("raw", NON_FINITE)
    def test_teleop_apply_ceiling_falls_back_to_its_default(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", raw)
        assert mesh_input._input_max_hz() == INPUT_MAX_HZ_DEFAULT

    @pytest.mark.parametrize("raw", NON_FINITE)
    def test_every_reader_returns_a_rate_a_period_can_be_built_from(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        """The property the readers must share, asserted on all three at once."""
        import math

        from strands_robots.mesh.core import Mesh

        monkeypatch.setenv(_ENV, raw)
        monkeypatch.setenv("STRANDS_MESH_CAMERA_HZ", raw)
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", raw)
        resolved = {
            "sensor loop": mesh_sensors._resolve_hz(_ENV, 10.0),
            "camera loop": Mesh(object(), peer_id="rate-parity-all")._resolve_camera_hz(),
            "teleop apply ceiling": mesh_input._input_max_hz(),
        }
        for reader, hz in resolved.items():
            assert math.isfinite(hz), f"{reader} returned {hz!r} for {raw!r}"
            assert hz >= 0.0, f"{reader} returned {hz!r} for {raw!r}"
