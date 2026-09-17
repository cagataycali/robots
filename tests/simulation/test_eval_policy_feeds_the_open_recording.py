"""``eval_policy`` / ``evaluate_benchmark`` under ``start_recording`` write frames.

``run_policy`` feeds an open recording on its own (its rollout hook calls
``add_frame``), and ``PolicyRunner.evaluate`` already closes one dataset episode
per evaluation episode - but it fed the recorder only through a caller-supplied
``on_frame``, which an agent cannot pass. So ``start_recording`` →
``eval_policy(n_episodes=2)`` advanced 30 steps, wrote 0 frames, answered with a
success rate, and ``stop_recording`` then refused on "captured no frames".

Pinned: with no ``on_frame`` and a recording open, the evaluation facades install
the backend's recording hook (the recording half of the rollout hook, factored
out); the answer says what was recorded; a caller's own ``on_frame`` is kept and
runs alone - chaining would double-write a hook that already calls ``add_frame``
- so that branch keeps its 0 frames and the answer names them; no recording open
→ no hook, no note; ``run_policy`` recording is unchanged.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

from strands_robots.simulation.benchmark import BenchmarkProtocol, StepInfo, register_benchmark, unregister_benchmark
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _json(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


@pytest.fixture
def sim():
    s = MuJoCoSimEngine(tool_name="eval_records", mesh=False)
    s.create_world()
    assert s.add_robot(name="so101", data_config="so101")["status"] == "success"
    yield s
    s.cleanup()


def test_eval_policy_under_an_open_recording_writes_one_dataset_episode_per_episode(sim, tmp_path):
    assert sim.start_recording(repo_id="lab/eval", root=str(tmp_path / "ds"), fps=30)["status"] == "success"
    r = sim.eval_policy("so101", policy_provider="mock", n_episodes=2, max_steps=15, control_frequency=30.0)
    assert r["status"] == "success", _text(r)
    assert "Recorded 2 episode(s), 30 frames to lab/eval" in _text(r)
    assert _json(r)["recording"] == {"repo_id": "lab/eval", "episodes": 2, "frames": 30}
    assert "[recording] 30 steps captured" in _text(sim.get_recording_status())

    stop = sim.stop_recording()
    assert stop["status"] == "success", _text(stop)
    assert "30 frames, 2 episode(s)" in _text(stop)

    replay = sim.replay_episode("lab/eval", robot_name="so101", root=str(tmp_path / "ds"), episode=1, speed=20.0)
    assert replay["status"] == "success", _text(replay)
    assert "Frames: 15/15" in _text(replay)


class _TenStepBenchmark(BenchmarkProtocol):
    """Never succeeds; ten steps per episode; accepts any robot."""

    max_steps = 10

    @property
    def supported_robots(self) -> list[str]:
        return []

    @property
    def default_robot(self) -> str:
        return "so101"

    def is_success(self, sim) -> bool:
        return False

    def on_step(self, sim, obs, action) -> StepInfo:
        return StepInfo(reward=0.0, done=False)


def test_evaluate_benchmark_under_an_open_recording_writes_frames_too(sim, tmp_path):
    register_benchmark("eval_records_probe", _TenStepBenchmark())
    try:
        assert sim.start_recording(repo_id="lab/bench", root=str(tmp_path / "ds"), fps=30)["status"] == "success"
        r = sim.evaluate_benchmark(
            "eval_records_probe", robot_name="so101", policy_provider="mock", n_episodes=1, control_frequency=30.0
        )
        assert r["status"] == "success", _text(r)
        assert "Recorded 1 episode(s), 10 frames to lab/bench" in _text(r)
        assert "10 frames, 1 episode(s)" in _text(sim.stop_recording())
    finally:
        unregister_benchmark("eval_records_probe")


def test_no_recording_open_means_no_hook_and_no_note(sim):
    r = sim.eval_policy("so101", policy_provider="mock", n_episodes=1, max_steps=5)
    assert r["status"] == "success"
    assert "Recorded" not in _text(r)
    assert "recording" not in _json(r)
    # And no hook at all, rather than one that no-ops per frame: an evaluation
    # that is not recording takes none of the recorder's per-frame lock traffic.
    assert sim._evaluation_recording("so101", "", None, "eval_policy") == (None, None)


def test_a_caller_supplied_on_frame_is_kept_and_an_unfed_recorder_is_named(sim, tmp_path):
    """A caller's hook still runs alone - and a recorder it never fed is reported.

    The facade must not chain its own hook onto the caller's: a caller hook that
    already calls ``add_frame`` (the documented pattern before this change)
    would then write every frame twice. So this branch keeps the old outcome -
    0 frames - and the evaluation now says so, at the end of the run that wrote
    them, instead of leaving it to ``stop_recording`` to refuse.
    """
    seen: list[int] = []
    assert sim.start_recording(repo_id="lab/own", root=str(tmp_path / "ds"), fps=30)["status"] == "success"
    r = sim.eval_policy(
        "so101",
        policy_provider="mock",
        n_episodes=1,
        max_steps=5,
        control_frequency=30.0,
        on_frame=lambda step, obs, act: seen.append(step),
    )
    assert r["status"] == "success"
    assert seen == [0, 1, 2, 3, 4]
    assert "Recorded" not in _text(r)
    assert (
        "Recording lab/own is open and this evaluation wrote 0 frames: the on_frame you passed "
        "does not call add_frame. Omit on_frame and the evaluation feeds the recorder itself"
    ) in _text(r)
    assert _json(r)["recording"] == {"repo_id": "lab/own", "episodes": 0, "frames": 0}
    assert "[recording] 0 steps captured" in _text(sim.get_recording_status())


def test_a_caller_on_frame_that_does_feed_is_reported_as_recorded(sim, tmp_path):
    """The other branch: frames landed, so the answer credits them, not a diagnosis."""
    assert sim.start_recording(repo_id="lab/both", root=str(tmp_path / "ds"), fps=30)["status"] == "success"
    hook = sim._make_recording_on_frame("so101", "")
    r = sim.eval_policy(
        "so101",
        policy_provider="mock",
        n_episodes=1,
        max_steps=5,
        control_frequency=30.0,
        on_frame=hook,
    )
    assert r["status"] == "success"
    assert "Recorded 1 episode(s), 5 frames to lab/both" in _text(r)
    assert "wrote 0 frames" not in _text(r)


def test_run_policy_still_records_on_its_own(sim, tmp_path):
    assert sim.start_recording(repo_id="lab/rp", root=str(tmp_path / "ds"), fps=30)["status"] == "success"
    r = sim.run_policy("so101", policy_provider="mock", duration=0.5, control_frequency=30.0)
    assert r["status"] == "success", _text(r)
    assert "[recording] 15 steps captured" in _text(sim.get_recording_status())
    assert "15 frames, 1 episode(s)" in _text(sim.stop_recording())


def test_the_recording_hook_factory_is_none_for_an_unknown_robot(sim):
    assert sim._make_recording_on_frame("ghost", "") is None


def _recorded_tasks(repo_id: str, root) -> set[str]:
    """Every task string the finalized dataset's own frames carry.

    Read through ``LeRobotDataset`` rather than the meta table: this is the
    ``task`` column a training run conditions on.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=str(root))
    return {str(dataset[i]["task"]) for i in range(len(dataset))}


class _LabelledBenchmark(_TenStepBenchmark):
    """Ships its own task language, the way LIBERO and Meta-World do."""

    max_steps = 4

    @property
    def instruction(self) -> str:
        return "pick up the red cube and lift it"


@pytest.mark.parametrize(
    ("caller_instruction", "session_task", "expected"),
    [
        # The benchmark's own language, which is what the policy is conditioned
        # on (#187) and so what the frames are of. Labelled with the caller's
        # empty argument they read "untitled" - add_frame's last resort.
        (None, None, "pick up the red cube and lift it"),
        # An explicit instruction still wins, for the policy and for the label.
        ("wave at the camera", None, "wave at the camera"),
        # Same precedence run_policy(instruction=...) has over the session's
        # task: the frames name the task the rollout was actually given.
        (None, "some session label", "pick up the red cube and lift it"),
    ],
)
def test_a_recorded_benchmark_evaluation_labels_its_frames_with_the_task_the_policy_got(
    sim, tmp_path, caller_instruction, session_task, expected
):
    register_benchmark("eval_records_label", _LabelledBenchmark())
    root = tmp_path / "ds"
    try:
        kwargs = {"task": session_task} if session_task is not None else {}
        assert sim.start_recording(repo_id="lab/label", root=str(root), fps=30, **kwargs)["status"] == "success"
        r = sim.evaluate_benchmark(
            "eval_records_label",
            robot_name="so101",
            policy_provider="mock",
            n_episodes=1,
            control_frequency=30.0,
            **({"instruction": caller_instruction} if caller_instruction is not None else {}),
        )
        assert r["status"] == "success", _text(r)
        assert "Recorded 1 episode(s), 4 frames" in _text(r)
        assert sim.stop_recording()["status"] == "success"
        assert _recorded_tasks("lab/label", root) == {expected}
    finally:
        unregister_benchmark("eval_records_label")


def test_the_benchmarks_task_language_is_read_from_one_place(sim):
    """``spec_instruction`` is that place - the eval loop and the recording label share it."""
    from strands_robots.simulation.benchmark import spec_instruction

    class _Raises(_TenStepBenchmark):
        @property
        def instruction(self) -> str:
            raise RuntimeError("a spec that cannot answer does not fail the evaluation")

    assert spec_instruction(_LabelledBenchmark()) == "pick up the red cube and lift it"
    assert spec_instruction(_TenStepBenchmark()) == ""  # the protocol default
    assert spec_instruction(_Raises()) == ""
    assert spec_instruction(object()) == ""  # no such property at all
