"""``get_recording_status`` reports the dataset that was saved, not a cleared buffer.

Measured on ``Robot("so101", mode="sim")``: ``start_recording(repo_id="lab/so101_ep",
root=/tmp/x)`` -> 30 scripted steps -> ``stop_recording`` answered "37 frames, 1
episode(s)", and ``get_recording_status`` right after it answered ``[idle] Not
recording (last episode: 0 steps)`` - byte-identical to the sentence a sim that
has never recorded gives. It read ``state["trajectory"]``, the mirror
``_release_dataset_recorder`` had just cleared, so the count was structurally
zero after every successful save.

Two readers were pointed here. ``docs/simulation/overview.md`` advertises
"Episode, frame count, output dir" for this method, and
``docs/troubleshooting.md`` sends an operator here to "check
get_recording_status() frame count" when an MP4 is empty - so the zero was the
false confirmation of the very diagnosis ("stopped before any frames") the
operator was sent to rule out.

``stop_recording`` now stashes the four facts of the save it just made, past the
teardown that drops the recorder, and idle reports them with a replay call. That
recipe carries ``root=`` because the bare id is not always enough: a later
session that saves nothing still moves ``last_dataset_repo_id`` /
``last_dataset_root`` (what #3786's replay adoption reads), and a bare-id replay
of the older id then errors while ``root=`` succeeds.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.simulation.recording import DatasetRecordingMixin

_SAVE = {"repo_id": "lab/so101_ep", "root": "/tmp/x", "frame_count": 37, "episode_count": 1}


class _Sim(DatasetRecordingMixin):
    """The production recording lifecycle over a state mapping the test owns.

    Only ``_recording_state`` is overridden - the seam the Isaac backend
    overrides too - so every reader under test (``_active_recorder``,
    ``_active_dataset_repo_id`` / ``_active_dataset_root``) is the real one.
    """

    def __init__(self, state: dict[str, Any] | None) -> None:
        self._state = state

    def _recording_state(self) -> dict[str, Any] | None:
        return self._state


class _Recorder:
    """A recorder with one pending episode, shaped like ``DatasetRecorder``."""

    def __init__(self, repo_id: str = "lab/so101_ep", root: str = "/tmp/x") -> None:
        self.repo_id, self.root = repo_id, root
        self.episode_frame_count, self.frame_count, self.episode_count = 37, 37, 0
        self.dataset = SimpleNamespace(meta=SimpleNamespace(total_episodes=1))

    def save_episode(self) -> dict[str, str]:
        self.episode_count += 1
        return {"status": "success"}

    def finalize(self) -> None:
        self.finalized = True


def _arm(sim: _Sim, recorder: _Recorder, steps: int) -> _Sim:
    """Open a session on ``sim``, the way a backend's ``start_recording`` leaves it."""
    state = sim._recording_state()
    assert state is not None
    state.update({"recording": True, "dataset_recorder": recorder, "trajectory": [0] * steps})
    return sim


def _open_session() -> _Sim:
    return _arm(_Sim({}), _Recorder(), 37)


def _text(reply: dict[str, Any]) -> str:
    return next(block["text"] for block in reply["content"] if "text" in block)


def _payload(reply: dict[str, Any]) -> dict[str, Any]:
    payload = next((block["json"] for block in reply["content"] if "json" in block), None)
    assert payload is not None, "a poller reads the json block, not three sentences"
    return payload


class TestASaveIsReportedAfterTheRecorderIsGone:
    def test_a_real_stop_recording_leaves_the_save_readable(self) -> None:
        """The whole defect, through the production path that closes a session."""
        sim = _open_session()
        assert sim.stop_recording()["status"] == "success"

        reply = sim.get_recording_status()
        assert _payload(reply)["last_save"] == _SAVE
        text = _text(reply)
        assert "37 frames" in text and "1 episode(s)" in text
        assert "lab/so101_ep" in text and "/tmp/x" in text
        assert "0 steps" not in text

    def test_the_teardown_that_clears_the_buffer_does_not_clear_it(self) -> None:
        """``steps`` is 0 by construction after a save, so it cannot be the source."""
        sim = _open_session()
        sim.stop_recording()
        assert _payload(sim.get_recording_status())["steps"] == 0

    def test_a_later_session_that_saves_nothing_neither_claims_nor_erases_it(self) -> None:
        """The empty-capture refusal is not a save, and the real one survives it."""
        sim = _open_session()
        sim.stop_recording()
        empty = _Recorder("lab/empty", "/tmp/e")
        empty.episode_frame_count = empty.frame_count = 0
        assert _arm(sim, empty, 0).stop_recording()["status"] == "error"
        assert _payload(sim.get_recording_status())["last_save"] == _SAVE


class TestEveryLifecycleStateAnswersInOneShape:
    @pytest.mark.parametrize(
        ("sim", "marker", "expected"),
        [
            (_Sim(None), "No world", {"world": False, "recording": False}),
            (_Sim({}), "nothing saved in this session", {"world": True, "recording": False}),
            (_open_session(), "[recording] 37 steps captured", {"world": True, "recording": True}),
            (_Sim({"last_save": _SAVE}), "Last saved: lab/so101_ep", {"world": True, "recording": False}),
        ],
        ids=["no-world", "nothing-saved", "recording", "after-a-save"],
    )
    def test_the_text_distinguishes_it_and_a_json_block_carries_it(
        self, sim: _Sim, marker: str, expected: dict[str, bool]
    ) -> None:
        reply = sim.get_recording_status()
        assert reply["status"] == "success"
        assert marker in _text(reply)
        payload = _payload(reply)
        assert {k: payload[k] for k in expected} == expected
        assert {"world", "recording", "steps", "last_save"} <= set(payload)

    def test_an_open_session_names_the_dataset_it_is_writing(self) -> None:
        """The one fact a caller polling an open session cannot get elsewhere."""
        payload = _payload(_open_session().get_recording_status())
        assert (payload["repo_id"], payload["root"]) == ("lab/so101_ep", "/tmp/x")


class TestTheRecipeIsRunnable:
    def test_the_printed_call_is_one_replay_episode_accepts(self) -> None:
        """A remedy naming a call must name arguments that call takes.

        ``root=`` is part of it deliberately: measured on the real sim, once a
        later session has recorded, replaying the saved id WITHOUT a root errors
        ("No dataset ... at the local default ...") while the printed call with
        ``root=`` replays 37/37 frames.
        """
        import ast
        import inspect
        import re

        from strands_robots.simulation.base import SimEngine

        text = _text(_Sim({"last_save": _SAVE}).get_recording_status())
        printed = re.search(r"replay_episode\([^)]*\)", text)
        assert printed is not None, f"no replay call in {text!r}"
        call = ast.parse(printed.group(0), mode="eval").body
        assert isinstance(call, ast.Call) and not call.args, "keyword arguments only"
        passed = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        assert passed == {"repo_id": _SAVE["repo_id"], "root": _SAVE["root"]}
        assert set(passed) <= set(inspect.signature(SimEngine.replay_episode).parameters)
