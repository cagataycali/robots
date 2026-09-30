"""``open_viewer`` on macOS names ``mjpython`` and how to run under it.

Issue #4169: ``mujoco.viewer.launch_passive`` refuses to run on macOS unless the
script is launched by ``mjpython``, the launcher the mujoco wheel installs next
to ``python``. MuJoCo's message named the launcher and nothing else, and no
docs page named it at all, so a reader on a Mac with a display got ``Viewer
failed: ...`` and no next step. The refusal now says what ``mjpython`` is, how
to run the script under it, and that :meth:`render` captures frames without a
window; the MuJoCo page says the same next to ``open_viewer()``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strands_robots.simulation.mujoco.simulation import MJPYTHON_LAUNCHER, viewer_failure_text

REPO = Path(__file__).resolve().parents[3]
MUJOCO_MESSAGE = "`launch_passive` requires that the Python script be run under `mjpython` on macOS"


def test_the_mjpython_refusal_names_the_launcher_and_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """MuJoCo's own sentence is kept, and the remedy follows it."""
    monkeypatch.setattr("sys.argv", [str(Path("scripts") / "rollout.py")])
    text = viewer_failure_text(RuntimeError(MUJOCO_MESSAGE))
    assert text.startswith(f"Viewer failed: {MUJOCO_MESSAGE}")
    assert f"run `{MJPYTHON_LAUNCHER} rollout.py` instead of `python rollout.py`" in text
    assert "installs next to python" in text
    assert "render()" in text


@pytest.mark.parametrize("argv", [[], [""], ["-c"], ["-"]])
def test_an_interactive_session_gets_a_placeholder_script(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    """``python -c`` and a REPL have no script name; the command still reads as one."""
    monkeypatch.setattr("sys.argv", argv)
    text = viewer_failure_text(RuntimeError(MUJOCO_MESSAGE))
    assert f"`{MJPYTHON_LAUNCHER} your_script.py`" in text


def test_another_viewer_failure_is_reported_as_mujoco_phrased_it() -> None:
    """The remedy is for the launcher case only; a GLFW failure is not sent to mjpython."""
    text = viewer_failure_text(RuntimeError("GLFW initialization failed"))
    assert text == "Viewer failed: GLFW initialization failed"
    assert MJPYTHON_LAUNCHER not in text


def test_open_viewer_reports_through_the_same_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine's ``open_viewer`` answer carries the remedy when launch_passive refuses."""
    from types import SimpleNamespace
    from typing import Any

    from strands_robots.simulation.mujoco import simulation as sim_module

    def refuse(model: object, data: object) -> None:
        raise RuntimeError(MUJOCO_MESSAGE)

    monkeypatch.setattr(sim_module, "mujoco_viewer", lambda: SimpleNamespace(launch_passive=refuse))
    monkeypatch.setattr("sys.argv", ["demo.py"])
    engine: Any = sim_module.MuJoCoSimEngine.__new__(sim_module.MuJoCoSimEngine)
    engine._world = SimpleNamespace(_model=object(), _data=object())
    engine._viewer_handle = None
    result = sim_module.MuJoCoSimEngine.open_viewer(engine)
    assert result["status"] == "error"
    assert f"run `{MJPYTHON_LAUNCHER} demo.py`" in result["content"][0]["text"]


def test_the_mujoco_page_names_mjpython_next_to_open_viewer() -> None:
    """The docs half: the sentence about ``open_viewer()`` names the launcher."""
    page = (REPO / "docs" / "learn" / "simulation" / "mujoco.md").read_text(encoding="utf-8")
    line = next(line for line in page.splitlines() if "`open_viewer()`" in line)
    assert f"`{MJPYTHON_LAUNCHER}`" in line
    assert "macOS" in line
