# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A second ``start_recording`` while one is live is refused; the live one is untouched.

It used to fall through: the recorder object was replaced (the frames buffered
since the last ``save_episode`` went with it - never saved, never mentioned), and
when the new dataset then refused (schema mismatch on resume) the caller was left
with ``recording`` False and both sessions' frames gone. Observed on the tool
surface as ``start_recording lab/live -> run_policy (15 steps) -> start_recording
lab/other -> success ... [recording] 0 steps captured``.

The refusal belongs to all THREE backends that implement ``start_recording``
(MuJoCo, Isaac, Newton) - each arms the shared recorder the same way, so each
drops frames the same way - and both figures it quotes are scoped to the live
SESSION, since what the caller is deciding is what THIS session would lose.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from strands_robots.simulation.newton.recording import NewtonRecordingMixin

if TYPE_CHECKING:
    from strands_robots.simulation.models import SimWorld

#: Every backend that implements ``start_recording``. A fourth one that forgets
#: the refusal silently reintroduces the frame loss, which is what the audit
#: cell below grades - the guard is a contract of the surface, not of one engine.
ENGINE_MIXINS = (
    "strands_robots.simulation.mujoco.recording",
    "strands_robots.simulation.isaac.recording",
    "strands_robots.simulation.newton.recording",
)


def _text(result: dict[str, Any]) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _json(result: dict[str, Any]) -> dict[str, Any]:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


@pytest.fixture
def sim():
    """A live MuJoCo engine with one arm, for the end-to-end rollout cases."""
    pytest.importorskip("mujoco")
    pytest.importorskip("lerobot")
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    s = MuJoCoSimEngine(tool_name="two_starts", mesh=False)
    s.create_world()
    assert s.add_robot(name="so101", data_config="so101")["status"] == "success"
    yield s
    s.cleanup()


def test_second_start_is_refused_and_the_first_recording_still_saves(sim, tmp_path):
    root1, root2 = str(tmp_path / "live"), str(tmp_path / "other")
    assert sim.start_recording(repo_id="lab/live", root=root1, fps=30)["status"] == "success"
    r = sim.run_policy("so101", policy_provider="mock", duration=0.5, control_frequency=30.0)
    assert r["status"] == "success", _text(r)
    assert "[recording] 15 steps captured" in _text(sim.get_recording_status())

    second = sim.start_recording(repo_id="lab/other", root=root2, fps=30)
    assert second["status"] == "error"
    assert _text(second) == (
        "start_recording: already recording 'lab/live' (15 frame(s) buffered since the last saved episode, "
        "0 episode(s) saved this session). Call stop_recording first - it saves the buffered frames - then "
        "start_recording 'lab/other'. The live recording is untouched."
    )
    assert _json(second) == {
        "recording": True,
        "repo_id": "lab/live",
        "frames_buffered": 15,
        "episodes_saved_this_session": 0,
        "requested_repo_id": "lab/other",
    }

    # Untouched: the 15 frames are still there and stop saves them.
    assert "[recording] 15 steps captured" in _text(sim.get_recording_status())
    stop = sim.stop_recording()
    assert stop["status"] == "success", _text(stop)
    assert "lab/live -- 15 frames, 1 episode(s)" in _text(stop)

    # And after stop the second dataset starts normally.
    assert sim.start_recording(repo_id="lab/other", root=root2, fps=30)["status"] == "success"
    sim.stop_recording()


def test_same_repo_id_twice_is_the_same_refusal_not_a_schema_error(sim, tmp_path):
    root = str(tmp_path / "ds")
    assert sim.start_recording(repo_id="lab/same", root=root, fps=30, cameras=[])["status"] == "success"
    again = sim.start_recording(repo_id="lab/same", root=root, fps=30)
    assert again["status"] == "error"
    assert _text(again).startswith("start_recording: already recording 'lab/same' (0 frame(s) buffered")
    assert "schema" not in _text(again)
    assert sim._is_recording()
    sim.stop_recording()


def test_the_episode_count_is_this_sessions_not_the_resumed_datasets(sim, tmp_path):
    """A resumed session that has saved nothing is not credited with the dataset's episodes.

    ``DatasetRecorder.resume`` seeds ``episode_count`` with the dataset's
    totals, so reading it raw told a caller "1 episode(s) saved" about a session
    that had saved none - the one reassurance a caller weighing a stop must not
    be given, since the 15 buffered frames were then the only thing at stake.
    Measured against the ``episodes_at_start`` stash, the same way
    ``stop_recording`` reports its own "+N episode(s) this session".
    """
    root = str(tmp_path / "ds")
    assert sim.start_recording(repo_id="lab/resumed", root=root, fps=30)["status"] == "success"
    sim.run_policy("so101", policy_provider="mock", duration=0.5, control_frequency=30.0)
    assert "1 episode(s)" in _text(sim.stop_recording())

    # Second session RESUMES that dataset, captures frames, saves no episode.
    resumed = sim.start_recording(repo_id="lab/resumed", root=root, fps=30)
    assert resumed["status"] == "success", _text(resumed)
    sim.run_policy("so101", policy_provider="mock", duration=0.5, control_frequency=30.0)
    assert int(sim._active_recorder().episode_count) == 1, "the recorder carries the dataset's total"

    refusal = sim.start_recording(repo_id="lab/third", root=str(tmp_path / "third"), fps=30)
    assert refusal["status"] == "error"
    assert "0 episode(s) saved this session" in _text(refusal)
    assert _json(refusal)["episodes_saved_this_session"] == 0
    assert _json(refusal)["frames_buffered"] == 15
    sim.stop_recording()


def test_newton_refuses_a_second_start_and_keeps_the_live_session(tmp_path):
    """The Newton backend answers a second start like MuJoCo does.

    Newton's ``start_recording`` fell through exactly as MuJoCo's did - it
    cleared the trajectory mirror and replaced the recorder, so the live
    session's buffered frames were gone and ``get_recording_status`` read
    ``[recording] 0 steps captured`` - while reporting ``success``. The engine
    needs no ``newton`` install to answer this: the refusal happens on the
    shared recording state, before any scene is read.
    """

    class _Recorder:
        repo_id = "lab/live"
        episode_frame_count = 15
        episode_count = 0
        frame_count = 15

    class _World:
        def __init__(self) -> None:
            self._backend_state: dict[str, Any] = {}
            self.cameras: dict[str, Any] = {}
            self.robots: dict[str, Any] = {}

    class _Engine(NewtonRecordingMixin):
        def __init__(self) -> None:
            self._world = cast("SimWorld", _World())
            self._model = object()

        def _validate_recording_start_rate(self, fps: Any, method: str) -> None:
            return None

    engine = _Engine()
    live = _Recorder()
    state = engine._world._backend_state
    state.update(
        {
            "recording": True,
            "dataset_recorder": live,
            "trajectory": [{"frame": i} for i in range(15)],
            "last_dataset_repo_id": "lab/live",
            "episodes_at_start": 0,
        }
    )

    second = engine.start_recording(repo_id="lab/other", root=str(tmp_path), fps=30)

    assert second["status"] == "error"
    assert _text(second).startswith(
        "start_recording: already recording 'lab/live' (15 frame(s) buffered since the last saved episode, "
        "0 episode(s) saved this session)."
    )
    assert _json(second)["requested_repo_id"] == "lab/other"
    # The live session is exactly as it was: same recorder, same buffered frames.
    assert state["dataset_recorder"] is live
    assert len(state["trajectory"]) == 15
    assert "[recording] 15 steps captured" in _text(engine.get_recording_status())


@pytest.mark.parametrize("module_name", ENGINE_MIXINS)
def test_every_backend_refuses_before_it_touches_the_recording_state(module_name: str) -> None:
    """``start_recording`` asks for the refusal before writing any session state.

    Two of the three backends carried the guard and Newton did not, so the
    silent frame loss survived on the engine whose own docstring says it matches
    MuJoCo. The order is the substance: the call has to come before the first
    write to the recording state, because it is those writes (``recording``,
    ``trajectory``, and the recorder that replaces the live one) that lose the
    frames.
    """
    module = __import__(module_name, fromlist=["*"])
    mixin = next(
        obj
        for _name, obj in vars(module).items()
        if inspect.isclass(obj) and obj.__module__ == module_name and hasattr(obj, "start_recording")
    )
    tree = ast.parse(Path(inspect.getfile(mixin)).read_text())
    func = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "start_recording")

    guard_lines = [
        node.lineno
        for node in ast.walk(func)
        if isinstance(node, ast.Attribute) and node.attr == "_already_recording_error"
    ]
    assert guard_lines, f"{module_name} does not refuse a second start_recording"

    # The first assignment of a session-state key, e.g. state["recording"] = True.
    session_writes = [
        node.lineno
        for node in ast.walk(func)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value in {"recording", "trajectory", "dataset_recorder"}
    ]
    assert session_writes, f"{module_name} writes no session state, so this cell reads nothing"
    assert min(guard_lines) < min(session_writes), (
        f"{module_name} writes recording state at line {min(session_writes)} before asking for the "
        f"refusal at line {min(guard_lines)}, so a second start drops the live session's frames"
    )


def test_no_refusal_when_idle(sim):
    assert sim._already_recording_error("start_recording", "lab/x") is None
