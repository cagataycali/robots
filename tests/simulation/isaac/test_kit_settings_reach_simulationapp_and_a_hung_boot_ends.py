"""Kit settings reach ``SimulationApp`` through ``IsaacConfig``, and a hung start-up ends.

Measured on one L40S host (32 cores, Isaac Sim 6.1), six ``Robot("so101",
backend="isaac")`` processes started together: with Kit's default task threads
(one per core in every process) 3 of 6 were still inside ``SimulationApp``
start-up after 240 s; with ``IsaacConfig(task_threads=4)`` 12 of 12 across two
runs booted in 15-16 s. ``IsaacConfig.extra`` was documented as the escape
hatch but never forwarded, so the setting could only be passed through a
private function before the first world. And a hung start-up blocked forever:
with ``boot_timeout_s=3`` the process now exits with status 70 and every
thread's stack.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

import strands_robots.simulation.isaac.simulation as sim_module  # noqa: E402
from strands_robots.simulation.isaac.config import IsaacConfig  # noqa: E402


class TestTheConfigFields:
    def test_the_defaults_change_nothing(self) -> None:
        config = IsaacConfig()
        assert config.kit_args == () and config.task_threads is None and config.boot_timeout_s is None
        assert sim_module._kit_args(config) == []

    def test_task_threads_becomes_the_carb_tasking_setting(self) -> None:
        config = IsaacConfig(kit_args=["--/app/foo=1"], task_threads=4)
        assert config.kit_args == ("--/app/foo=1",)
        assert sim_module._kit_args(config) == ["--/app/foo=1", "--/plugins/carb.tasking.plugin/threadCount=4"]

    @pytest.mark.parametrize(
        ("kwargs", "needle"),
        [
            ({"kit_args": "--/app/foo=1"}, "kit_args must be a sequence"),
            ({"kit_args": ["/app/foo=1"]}, "starting with '--'"),
            ({"kit_args": [3]}, "starting with '--'"),
            ({"task_threads": 0}, "task_threads"),
            ({"task_threads": True}, "task_threads"),
            ({"boot_timeout_s": 0}, "boot_timeout_s"),
            ({"boot_timeout_s": float("inf")}, "boot_timeout_s"),
        ],
    )
    def test_an_unusable_value_is_refused_by_name(self, kwargs: dict[str, Any], needle: str) -> None:
        with pytest.raises(ValueError, match=needle):
            IsaacConfig(**kwargs)


class TestTheWatchdog:
    def test_a_start_within_the_timeout_exits_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        exits: list[int] = []
        monkeypatch.setattr(sim_module.os, "_exit", exits.append)
        with sim_module._boot_watchdog(0.5):
            pass
        time.sleep(0.7)
        assert exits == []

    def test_a_hung_start_exits_with_status_70_and_the_remedy(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exited = threading.Event()
        codes: list[int] = []

        def fake_exit(code: int) -> None:
            codes.append(code)
            exited.set()

        monkeypatch.setattr(sim_module.os, "_exit", fake_exit)
        with sim_module._boot_watchdog(0.05):
            assert exited.wait(2.0)
        assert codes == [sim_module.BOOT_TIMEOUT_EXIT_STATUS] == [70]
        assert "task_threads=4" in capsys.readouterr().err

    def test_a_broken_stderr_does_not_defeat_the_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A batch driver that died leaves each child's stderr pipe broken; the watchdog's diagnostics
        must not take the exit with them. Every stderr operation may raise, the exit still happens."""
        import io

        class _Broken(io.StringIO):
            def write(self, _text: str) -> int:
                raise ValueError("I/O operation on closed file")

            def flush(self) -> None:
                raise ValueError("I/O operation on closed file")

        exited = threading.Event()
        codes: list[int] = []

        def fake_exit(code: int) -> None:
            codes.append(code)
            exited.set()

        monkeypatch.setattr(sim_module.os, "_exit", fake_exit)
        monkeypatch.setattr(sys, "stderr", _Broken())
        with sim_module._boot_watchdog(0.05):
            assert exited.wait(2.0), "the watchdog thread died on the broken stderr before os._exit"
        assert codes == [sim_module.BOOT_TIMEOUT_EXIT_STATUS]

    def test_none_watches_nothing(self) -> None:
        before = threading.active_count()
        with sim_module._boot_watchdog(None):
            assert threading.active_count() == before


class TestCreateWorldForwardsThem:
    def test_the_settings_reach_the_simulationapp_launch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tests.simulation._isaac_engine import isaac_engine

        seen: dict[str, Any] = {}

        def fake_app(headless: bool = True, launch_config: Any = None, **kw: Any) -> Any:
            seen["launch"] = dict(launch_config or {})
            raise ImportError("stop here: no Isaac runtime in unit tests")

        monkeypatch.setattr(sim_module, "_get_or_create_simulation_app", fake_app)
        engine = isaac_engine(IsaacConfig(task_threads=4, kit_args=("--/app/foo=1",)))
        engine.create_world()
        assert seen["launch"]["extra_args"] == ["--/app/foo=1", "--/plugins/carb.tasking.plugin/threadCount=4"]
