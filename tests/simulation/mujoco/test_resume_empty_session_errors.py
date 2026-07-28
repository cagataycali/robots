# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""An append session that captured nothing must fail loudly, like a fresh one does.

``stop_recording`` chose between flush / no-op / error using ``pending =
episode_frame_count`` and ``captured = frame_count``, with "nothing was ever
captured" being ``elif captured == 0``. But ``DatasetRecorder.resume`` seeds
``frame_count`` from the dataset already on disk ("so reporting reflects totals"),
so on ANY append session it starts non-zero. An append that captured zero new
frames therefore had ``pending == 0`` and ``captured > 0``, skipped both branches,
and returned success with the INHERITED counts. Measured::

    session 1 records 5 frames -> success  'local/d48 -- 5 frames, 1 episode(s)'
    session 2 resumes, runs NO rollout
      resumed counters: frame_count=5 episode_count=1 episode_frame_count=0
    session 2 stop_recording -> success  'local/d48 -- 5 frames, 1 episode(s)'

The equivalent FRESH session already errored correctly, so the guard was right
about everything except which counter it read.

The same conflation made the #708 parquet-truth gate a tautology on resume:
it compared ``recorder.episode_count`` against ``dataset.meta.total_episodes``,
and ``resume()`` seeds the former FROM the latter - so the gate could never fire
on a resumed dataset, which is exactly where a silent collapse is hardest to
notice. It now composes the expectation as
``episodes_seeded_at_resume + session_episode_count``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

from strands_robots.policies.base import Policy  # noqa: E402
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402


class _Hold(Policy):
    """State-only policy: no camera renders, so these tests stay fast."""

    def __init__(self, keys) -> None:
        super().__init__()
        self._keys = list(keys)

    @property
    def provider_name(self) -> str:
        return "hold"

    def set_robot_state_keys(self, keys) -> None:
        pass

    @property
    def requires_images(self) -> bool:
        return False

    async def get_actions(self, observation, instruction, **kwargs):
        return [dict.fromkeys(self._keys, 0.0)]


def _session(root, *, n_steps: int):
    """Open a recording session on ``root``, optionally roll, then stop.

    Returns ``(result, payload, recorder)``. The recorder is captured BEFORE
    stop_recording drops it, so its counters can be asserted.
    """
    sim = MuJoCoSimEngine()
    try:
        sim.create_world()
        assert sim.add_robot("so101")["status"] == "success"
        started = sim.start_recording(repo_id="local/resume_guard", root=str(root), fps=50, task="t")
        assert started["status"] == "success", started
        recorder = sim._world._backend_state["dataset_recorder"]  # type: ignore[union-attr]
        if n_steps:
            sim.run_policy(
                policy_object=_Hold(sim.robot_action_keys("so101")),
                robot_name="so101",
                n_steps=n_steps,
                control_frequency=50.0,
            )
        result = sim.stop_recording()
        payload: dict = next((block["json"] for block in result["content"] if "json" in block), {})
        return result, payload, recorder
    finally:
        sim.destroy()


def _text(result) -> str:
    return " ".join(block.get("text", "") for block in result.get("content", []) if "text" in block)


class TestAnEmptyAppendSessionErrors:
    def test_a_resumed_session_that_captured_nothing_is_an_error(self, tmp_path):
        """The regression: this returned success with the inherited counts."""
        root = tmp_path / "ds"
        first, _, _ = _session(root, n_steps=5)
        assert first["status"] == "success", first

        second, _, _ = _session(root, n_steps=0)

        assert second["status"] == "error", second

    def test_the_message_distinguishes_the_session_from_the_dataset_total(self, tmp_path):
        """A bare '0 frames' would be a lie: the dataset has 5."""
        root = tmp_path / "ds"
        _session(root, n_steps=5)

        second, _, _ = _session(root, n_steps=0)

        text = _text(second)
        assert "THIS session" in text, text
        assert "5 frame(s) from earlier sessions" in text, text
        assert text.isascii()

    def test_a_fresh_empty_session_still_errors(self, tmp_path):
        """The behaviour that already worked must be unchanged."""
        result, _, _ = _session(tmp_path / "fresh", n_steps=0)

        assert result["status"] == "error"
        assert "dataset would be empty" in _text(result), _text(result)


class TestLegitimateAppendsStillWork:
    def test_an_append_that_captured_frames_succeeds_and_accumulates(self, tmp_path):
        root = tmp_path / "ds"
        first, first_payload, _ = _session(root, n_steps=5)
        assert first_payload["frame_count"] == 5, first_payload

        second, second_payload, _ = _session(root, n_steps=4)

        assert second["status"] == "success", second
        assert second_payload["frame_count"] == 9, second_payload
        assert second_payload["episode_count"] == 2, second_payload

    def test_recording_still_works_after_an_empty_session_error(self, tmp_path):
        """The error must not poison the dataset for later appends."""
        root = tmp_path / "ds"
        _session(root, n_steps=5)
        assert _session(root, n_steps=0)[0]["status"] == "error"

        third, payload, _ = _session(root, n_steps=3)

        assert third["status"] == "success", third
        assert payload["frame_count"] == 8, payload


class TestSessionCountersAreSessionScoped:
    def test_a_fresh_recorder_starts_both_counters_at_zero(self, tmp_path):
        _, _, recorder = _session(tmp_path / "fresh2", n_steps=0)

        assert recorder.session_frame_count == 0
        assert recorder.session_episode_count == 0
        assert recorder.episodes_seeded_at_resume == 0

    def test_a_resumed_recorder_does_not_inherit_the_session_counters(self, tmp_path):
        """The whole point: totals are seeded, session counts are not."""
        root = tmp_path / "ds"
        _session(root, n_steps=5)

        _, _, recorder = _session(root, n_steps=0)

        assert recorder.frame_count == 5, "the dataset total was not inherited"
        assert recorder.session_frame_count == 0, "the session count was seeded - the defect"
        assert recorder.episodes_seeded_at_resume == 1

    def test_the_session_counters_track_this_sessions_work_only(self, tmp_path):
        root = tmp_path / "ds"
        _session(root, n_steps=5)

        _, _, recorder = _session(root, n_steps=4)

        assert recorder.frame_count == 9, "total"
        assert recorder.session_frame_count == 4, "session"
        assert recorder.session_episode_count == 1
        assert recorder.episodes_seeded_at_resume == 1


class TestTheParquetGateKeepsItsPower:
    def test_no_spurious_mismatch_on_a_clean_append(self, tmp_path):
        """Composing seeded + session must agree with parquet when nothing broke."""
        root = tmp_path / "ds"
        _session(root, n_steps=5)

        _, payload, _ = _session(root, n_steps=4)

        assert payload["episode_count_mismatch"] is False, payload
        assert payload["parquet_episode_count"] == 2, payload

    def test_the_gate_fires_when_the_composed_count_disagrees(self, tmp_path):
        """Pin that the gate is no longer a tautology on the append path.

        Forcing the session count off by one must be DETECTED; comparing the
        seeded ``episode_count`` against the number it was seeded from could
        never detect anything.
        """
        root = tmp_path / "ds"
        _session(root, n_steps=5)

        sim = MuJoCoSimEngine()
        try:
            sim.create_world()
            assert sim.add_robot("so101")["status"] == "success"
            assert (
                sim.start_recording(repo_id="local/resume_guard", root=str(root), fps=50, task="t")["status"]
                == "success"
            )
            recorder = sim._world._backend_state["dataset_recorder"]
            sim.run_policy(
                policy_object=_Hold(sim.robot_action_keys("so101")),
                robot_name="so101",
                n_steps=3,
                control_frequency=50.0,
            )
            # Claim one more saved episode than the parquet will hold.
            recorder.session_episode_count += 1

            result = sim.stop_recording()

            payload = next((block["json"] for block in result["content"] if "json" in block), {})
            assert payload["episode_count_mismatch"] is True, payload
            # Parquet remains the reported truth.
            assert payload["episode_count"] == payload["parquet_episode_count"], payload
        finally:
            sim.destroy()
