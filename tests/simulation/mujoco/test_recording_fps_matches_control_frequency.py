# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A dataset's fps must describe the rate its frames were actually captured at.

``start_recording`` defaults ``fps=30`` while ``run_policy`` defaults
``control_frequency=50.0``, and nothing compared them. lerobot's ``add_frame``
synthesizes ``timestamp = frame_index / meta.fps`` unconditionally, so a rollout
captured every 20 ms was written as if the frames were 33.3 ms apart. Measured on
the two untouched defaults::

    fps=30  cf=50.0 -> sim time/frame 0.0200s  dataset time/frame 0.0333s  1.667x
    fps=50  cf=50.0 -> sim time/frame 0.0200s  dataset time/frame 0.0200s  1.000x

silently, with no log line. That contradicts ``policy_runner``'s own stated
invariant ("the recorded control frequency IS the dataset fps") and it propagates:
``replay_episode`` derives its per-frame physics budget from the dataset fps, so a
mislabelled dataset also replays at the wrong speed.

The fix warns once per recorder, naming both rates, the per-frame times and the
distortion factor. Deliberately a warning, not a refusal: a mismatched dataset is
still readable and a caller may be recording a deliberately decimated stream, so
this must not break an in-flight rollout.

Note for anyone extending these: give each case its OWN ``root``.
``start_recording`` RESUMES an existing dataset directory and inherits its fps
from disk, so sharing one root across cases makes every case report the first
one's fps - which looks exactly like the fps argument being ignored.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

from strands_robots.policies.base import Policy  # noqa: E402
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402

_WARNING_MARKER = "does not match control_frequency"


class _Hold(Policy):
    """A state-only policy: no camera renders, so these tests stay fast."""

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


def _recording_sim(tmp_path, fps: int, case: str):
    """A world recording at ``fps`` into its OWN dataset root."""
    sim = MuJoCoSimEngine()
    sim.create_world()
    assert sim.add_robot("so101")["status"] == "success"
    result = sim.start_recording(repo_id=f"local/{case}", root=str(tmp_path / case), fps=fps, task="t")
    assert result["status"] == "success", result
    return sim


def _dataset_fps(sim) -> float:
    recorder = sim._world._backend_state["dataset_recorder"]
    return float(recorder.dataset.meta.fps)


def _roll(sim, control_frequency: float, n_steps: int = 4):
    return sim.run_policy(
        policy_object=_Hold(sim.robot_action_keys("so101")),
        robot_name="so101",
        n_steps=n_steps,
        control_frequency=control_frequency,
    )


def _warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if _WARNING_MARKER in record.getMessage()]


class TestMismatchIsReported:
    def test_the_two_defaults_warn(self, tmp_path, caplog):
        """The regression: fps=30 vs control_frequency=50.0, silently 1.667x off."""
        sim = _recording_sim(tmp_path, 30, "defaults")
        try:
            assert _dataset_fps(sim) == pytest.approx(30.0), "fixture did not record at 30fps"
            with caplog.at_level("WARNING"):
                _roll(sim, 50.0)

            warnings = _warnings(caplog)
            assert warnings, [r.getMessage() for r in caplog.records]
        finally:
            sim.stop_recording()
            sim.destroy()

    def test_the_warning_names_both_rates_and_the_distortion(self, tmp_path, caplog):
        sim = _recording_sim(tmp_path, 30, "text")
        try:
            with caplog.at_level("WARNING"):
                _roll(sim, 50.0)

            warning = _warnings(caplog)[0]
            assert "30.0" in warning and "50.0" in warning, warning
            assert "1.667x" in warning, warning
            # Actionable: it must say what to pass instead.
            assert "start_recording(fps=50)" in warning, warning
            assert warning.isascii()
        finally:
            sim.stop_recording()
            sim.destroy()

    @pytest.mark.parametrize("fps,control_frequency", [(30, 50.0), (60, 50.0), (30, 25.0)])
    def test_any_mismatch_warns(self, tmp_path, caplog, fps, control_frequency):
        sim = _recording_sim(tmp_path, fps, f"m{fps}_{int(control_frequency)}")
        try:
            assert _dataset_fps(sim) == pytest.approx(float(fps))
            with caplog.at_level("WARNING"):
                _roll(sim, control_frequency)

            assert _warnings(caplog), f"fps={fps} vs cf={control_frequency} did not warn"
        finally:
            sim.stop_recording()
            sim.destroy()

    def test_run_multi_policy_warns_too(self, tmp_path, caplog):
        """The multi-robot loop records as well, so it needs the same check."""
        sim = _recording_sim(tmp_path, 30, "multi")
        try:
            policies = {"so101": _Hold(sim.robot_action_keys("so101"))}
            with caplog.at_level("WARNING"):
                sim.run_multi_policy(policies, n_steps=3, control_frequency=50.0, action_horizon=1)

            assert _warnings(caplog), [r.getMessage() for r in caplog.records]
        finally:
            sim.stop_recording()
            sim.destroy()


class TestMatchingRatesStaySilent:
    @pytest.mark.parametrize("rate", [50, 30, 25])
    def test_equal_rates_do_not_warn(self, tmp_path, caplog, rate):
        sim = _recording_sim(tmp_path, rate, f"eq{rate}")
        try:
            assert _dataset_fps(sim) == pytest.approx(float(rate))
            with caplog.at_level("WARNING"):
                _roll(sim, float(rate))

            assert not _warnings(caplog), _warnings(caplog)
        finally:
            sim.stop_recording()
            sim.destroy()

    def test_no_recorder_attached_does_not_warn(self, caplog):
        """A rollout with recording off must be completely unaffected."""
        sim = MuJoCoSimEngine()
        try:
            sim.create_world()
            assert sim.add_robot("so101")["status"] == "success"

            with caplog.at_level("WARNING"):
                _roll(sim, 50.0)

            assert not _warnings(caplog)
            assert not sim._world._backend_state.get("recording_fps_mismatch_warned")
        finally:
            sim.destroy()


class TestTheWarningIsOneShot:
    def test_a_second_rollout_does_not_re_warn(self, tmp_path, caplog):
        """A long session must not spam the log for one configuration mistake."""
        sim = _recording_sim(tmp_path, 30, "oneshot")
        try:
            with caplog.at_level("WARNING"):
                _roll(sim, 50.0)
                first = len(_warnings(caplog))
                _roll(sim, 50.0)
                _roll(sim, 50.0)

            assert first == 1, f"{first} warnings from the first rollout"
            assert len(_warnings(caplog)) == 1, f"{len(_warnings(caplog))} warnings across three rollouts"
        finally:
            sim.stop_recording()
            sim.destroy()

    def test_the_latch_lives_on_the_world_state(self, tmp_path):
        sim = _recording_sim(tmp_path, 30, "latch")
        try:
            assert not sim._world._backend_state.get("recording_fps_mismatch_warned")

            _roll(sim, 50.0)

            assert sim._world._backend_state.get("recording_fps_mismatch_warned") is True
        finally:
            sim.stop_recording()
            sim.destroy()


class TestTheHelperIsRobust:
    def test_a_world_without_backend_state_is_a_no_op(self):
        sim = MuJoCoSimEngine()
        try:
            # No create_world: _world is None.
            sim._warn_on_recording_fps_mismatch(50.0, "probe")
        finally:
            sim.destroy()

    def test_a_recorder_with_no_readable_fps_is_a_no_op(self, tmp_path, caplog):
        """An unknown recorder shape must not crash a rollout."""
        sim = _recording_sim(tmp_path, 30, "shape")
        try:
            sim._world._backend_state["dataset_recorder"] = object()

            with caplog.at_level("WARNING"):
                sim._warn_on_recording_fps_mismatch(50.0, "probe")

            assert not _warnings(caplog)
        finally:
            sim._world._backend_state["recording"] = False
            sim.destroy()
