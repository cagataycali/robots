"""The ``foxglove=`` keyword family is graded before anything opens a socket.

``Robot(name, foxglove=...)``, ``MuJoCoSimEngine(foxglove=...)`` and the
hardware ``Robot`` all hand their three keywords to
:func:`strands_robots.foxglove.options.resolve_foxglove_options`, so the
accepted spellings and every refusal sentence live in one place and are pinned
here once. Nothing in this file needs the ``[foxglove]`` extra.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands_robots.foxglove.options import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    FOXGLOVE_ENV,
    FoxgloveOptions,
    resolve_foxglove_options,
)


@pytest.fixture(autouse=True)
def _no_env_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FOXGLOVE_ENV, raising=False)


class TestOff:
    def test_false_resolves_to_nothing(self) -> None:
        assert resolve_foxglove_options(False, context="Robot") is None

    def test_mcap_without_the_server_is_refused_by_name(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"foxglove_mcap / foxglove_services require foxglove=True"):
            resolve_foxglove_options(False, foxglove_mcap=tmp_path / "run.mcap", context="Robot")

    def test_services_without_the_server_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match=r"require foxglove=True"):
            resolve_foxglove_options(False, foxglove_services=True, context="Robot")


class TestOn:
    def test_true_serves_the_default_address(self) -> None:
        options = resolve_foxglove_options(True, context="Robot")
        assert options == FoxgloveOptions(host=DEFAULT_HOST, port=DEFAULT_PORT)

    @pytest.mark.parametrize(
        "spelling, expected",
        [
            ("0.0.0.0:8765", ("0.0.0.0", 8765)),
            (":0", (DEFAULT_HOST, 0)),
            ("192.168.1.20", ("192.168.1.20", DEFAULT_PORT)),
            ("localhost:9000", ("localhost", 9000)),
        ],
    )
    def test_a_host_port_string_names_the_address(self, spelling: str, expected: tuple[str, int]) -> None:
        options = resolve_foxglove_options(spelling, context="Robot")
        assert options is not None
        assert (options.host, options.port) == expected

    def test_the_mcap_path_and_services_flag_travel(self, tmp_path: Path) -> None:
        options = resolve_foxglove_options(
            True, foxglove_mcap=str(tmp_path / "run.mcap"), foxglove_services=True, context="Robot"
        )
        assert options is not None
        assert options.mcap == tmp_path / "run.mcap"
        assert options.services is True


class TestRefusals:
    @pytest.mark.parametrize("value", [1, 8765, 2.5, ["127.0.0.1", 8765], object()])
    def test_a_value_that_is_neither_flag_nor_address_is_refused(self, value: object) -> None:
        with pytest.raises(ValueError, match=r"MuJoCoSimEngine: foxglove must be True, False or a 'host:port' string"):
            resolve_foxglove_options(value, context="MuJoCoSimEngine")

    @pytest.mark.parametrize("value", ["host:port", "127.0.0.1:99999", "127.0.0.1:-1"])
    def test_an_address_without_a_port_is_refused(self, value: str) -> None:
        with pytest.raises(ValueError, match=r"does not name a port"):
            resolve_foxglove_options(value, context="Robot")

    def test_services_must_be_a_boolean(self) -> None:
        with pytest.raises(ValueError, match=r"foxglove_services must be a boolean"):
            resolve_foxglove_options(True, foxglove_services="false", context="Robot")

    def test_an_existing_mcap_is_refused_rather_than_overwritten(self, tmp_path: Path) -> None:
        existing = tmp_path / "run.mcap"
        existing.write_bytes(b"")
        with pytest.raises(ValueError, match=r"already exists and would be overwritten"):
            resolve_foxglove_options(True, foxglove_mcap=existing, context="Robot")

    def test_an_mcap_in_a_missing_directory_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"its directory does not exist"):
            resolve_foxglove_options(True, foxglove_mcap=tmp_path / "nowhere" / "run.mcap", context="Robot")

    @pytest.mark.parametrize("value", [True, 3, "", "   "])
    def test_an_mcap_that_is_not_a_path_is_refused(self, value: object) -> None:
        with pytest.raises(ValueError, match=r"foxglove_mcap must be a file path"):
            resolve_foxglove_options(True, foxglove_mcap=value, context="Robot")


class TestEnvironmentSwitch:
    @pytest.mark.parametrize("spelling", ["1", "true", "on", "YES"])
    def test_an_on_spelling_starts_the_default_server(self, monkeypatch: pytest.MonkeyPatch, spelling: str) -> None:
        monkeypatch.setenv(FOXGLOVE_ENV, spelling)
        options = resolve_foxglove_options(False, context="Robot")
        assert options == FoxgloveOptions(host=DEFAULT_HOST, port=DEFAULT_PORT, from_env=True)

    @pytest.mark.parametrize("spelling", ["0", "false", "", "off"])
    def test_an_off_spelling_leaves_the_server_off(self, monkeypatch: pytest.MonkeyPatch, spelling: str) -> None:
        monkeypatch.setenv(FOXGLOVE_ENV, spelling)
        assert resolve_foxglove_options(False, context="Robot") is None

    def test_an_address_in_the_variable_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FOXGLOVE_ENV, "0.0.0.0:9100")
        options = resolve_foxglove_options(False, context="Robot")
        assert options is not None
        assert (options.host, options.port, options.from_env) == ("0.0.0.0", 9100, True)

    def test_a_bad_variable_is_refused_by_the_variable_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FOXGLOVE_ENV, "nonsense:port")
        with pytest.raises(ValueError, match=FOXGLOVE_ENV):
            resolve_foxglove_options(False, context="Robot")

    def test_the_keyword_wins_over_the_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FOXGLOVE_ENV, "0.0.0.0:9100")
        options = resolve_foxglove_options("127.0.0.1:9200", context="Robot")
        assert options is not None
        assert (options.port, options.from_env) == (9200, False)


class TestTheEnginesGradeTheSameKeywords:
    """The refusal a caller reads is the same sentence whichever constructor took the keyword."""

    def test_the_mujoco_engine_refuses_before_it_builds(self) -> None:
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        with pytest.raises(ValueError, match=r"MuJoCoSimEngine: foxglove must be True, False or a 'host:port' string"):
            MuJoCoSimEngine(foxglove=8765)  # type: ignore[arg-type]

    def test_the_mujoco_engine_refuses_a_sidecar_without_a_server(self, tmp_path: Path) -> None:
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        with pytest.raises(ValueError, match=r"require foxglove=True"):
            MuJoCoSimEngine(foxglove_mcap=tmp_path / "run.mcap")

    def test_the_sim_engine_base_grades_them_too(self) -> None:
        from tests.foxglove._engine_double import TelemetryEngine

        with pytest.raises(
            ValueError, match=r"TelemetryEngine: foxglove_mcap / foxglove_services require foxglove=True"
        ):
            TelemetryEngine({}, foxglove=False, foxglove_services=True)

    def test_the_hardware_robot_grades_them_before_lerobot_is_imported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.hardware_robot import Robot

        def _never(*args: object, **kwargs: object) -> None:
            raise AssertionError("_initialize_robot must not run for a refused keyword")

        monkeypatch.setattr(Robot, "_initialize_robot", _never)
        with pytest.raises(ValueError, match=r"Robot: foxglove must be True, False or a 'host:port' string"):
            Robot("arm", "so101_follower", foxglove=8765)  # type: ignore[arg-type]
