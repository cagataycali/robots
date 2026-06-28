"""Multi-episode contract for ``run_policy(n_episodes=N)``.

``run_policy`` historically ran exactly one rollout. Collecting an N-episode
dataset therefore forced callers into a brittle manual
``for _ in range(N): run_policy(); save_episode(); reset()`` loop - and
forgetting the per-iteration ``save_episode`` silently merged every rollout
into a single ``episode_index=0`` (the dataset reported ``total_episodes=1``
no matter how many rollouts ran).

These tests pin the first-class multi-episode API:

* ``n_episodes`` (default ``1``) runs that many sequential rollouts in one call.
* ``n_episodes == 1`` is byte-for-byte the historical single-rollout result
  shape (no aggregate wrapper), so existing callers are unaffected.
* A malformed ``n_episodes`` is rejected with a structured ASCII error.
* When a recording is active, each rollout is flushed as its OWN dataset
  episode (the data-correctness fix: ``total_episodes == n_episodes``).
* ``reset_between`` controls the inter-episode reset; the final episode never
  triggers a reset.
* A reused ``video`` config is templated per episode (``_ep{i}``) so episodes
  do not overwrite the same MP4.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("mujoco")

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots.simulation import create_simulation  # noqa: E402
from strands_robots.simulation.base import SimEngine  # noqa: E402


@pytest.fixture
def sim():
    s = create_simulation()
    s.create_world()
    s.add_robot("arm1", data_config="so100")
    yield s
    s.cleanup()


class _RecordinglessEngine(SimEngine):
    """Minimal concrete SimEngine that does not support recording.

    Implements the abstract surface as no-ops so the base ``_is_recording`` /
    ``save_episode`` hooks (which backends like MuJoCo override) can be
    exercised directly.
    """

    def create_world(self, *a, **k):  # type: ignore[no-untyped-def]
        return {"status": "success", "content": []}

    def destroy(self):  # type: ignore[no-untyped-def]
        return {"status": "success", "content": []}

    def reset(self):  # type: ignore[no-untyped-def]
        return {"status": "success", "content": []}

    def step(self, n_steps: int = 1):  # type: ignore[no-untyped-def]
        return {"status": "success", "content": []}

    def get_state(self):  # type: ignore[no-untyped-def]
        return {"status": "success", "content": []}

    def add_robot(self, *a, **k):  # type: ignore[no-untyped-def]
        return {"status": "success", "content": []}

    def remove_robot(self, *a, **k):  # type: ignore[no-untyped-def]
        return {"status": "success", "content": []}

    def list_robots(self):  # type: ignore[no-untyped-def]
        return []

    def robot_joint_names(self, robot_name):  # type: ignore[no-untyped-def]
        return []

    def add_object(self, *a, **k):  # type: ignore[no-untyped-def]
        return {"status": "success", "content": []}

    def remove_object(self, *a, **k):  # type: ignore[no-untyped-def]
        return {"status": "success", "content": []}

    def get_observation(self, robot_name=None, skip_images=False):  # type: ignore[no-untyped-def]
        return {"status": "success", "content": []}

    def send_action(self, action, robot_name=None, n_substeps=1):  # type: ignore[no-untyped-def]
        return {"status": "success", "content": []}

    def physics_timestep(self):  # type: ignore[no-untyped-def]
        return 0.002

    def render(self, camera_name="default", width=None, height=None):  # type: ignore[no-untyped-def]
        return {"status": "success", "content": []}


def _json(result: dict) -> dict:
    for blk in result["content"]:
        if isinstance(blk, dict) and isinstance(blk.get("json"), dict):
            return blk["json"]
    raise AssertionError(f"no json block in result: {result}")


def _patch_runner(monkeypatch, run_fn):
    """Patch ``PolicyRunner.run`` on the class ``run_policy`` actually instantiates.

    ``SimEngine.run_policy`` builds ``runner = PolicyRunner(self)`` from the
    ``PolicyRunner`` symbol bound in its OWN module namespace. Patching the name
    imported from elsewhere (e.g. ``policy_runner.PolicyRunner``) silently misses
    that instance whenever another test has reloaded the ``policy_runner`` module:
    the reload rebinds ``sys.modules[...policy_runner].PolicyRunner`` to a fresh
    class object while ``base`` keeps its original import, so the two diverge and
    the real rollout runs unpatched. Resolving the class from the live module that
    owns ``run_policy`` keeps the seam effective regardless of such reloads.
    """
    import sys

    runner_module = sys.modules[SimEngine.run_policy.__module__]
    monkeypatch.setattr(runner_module.PolicyRunner, "run", run_fn)


class TestMultiEpisodeRollout:
    def test_n_episodes_runs_multiple_rollouts(self, sim):
        result = sim.run_policy("arm1", n_steps=5, n_episodes=3, control_frequency=50.0)
        assert result["status"] == "success", result
        payload = _json(result)
        assert payload["n_episodes_requested"] == 3
        assert payload["n_episodes_completed"] == 3
        assert len(payload["episodes"]) == 3
        # Each episode ran the configured per-episode horizon.
        assert [e["n_steps"] for e in payload["episodes"]] == [5, 5, 5]
        assert payload["total_steps"] == 15
        # No recorder attached -> nothing flushed.
        assert payload["episodes_saved"] == 0

    def test_single_episode_keeps_historical_result_shape(self, sim):
        # Default n_episodes=1 must NOT wrap the result in the multi-episode
        # aggregate (the per-episode ``episodes`` list); it stays the
        # single-rollout payload existing callers parse. It DOES, however, carry
        # the episode-count contract fields so an agent can read the truth
        # without parsing text (both run_policy paths expose them).
        result = sim.run_policy("arm1", n_steps=5)
        assert result["status"] == "success"
        payload = _json(result)
        assert payload["n_steps"] == 5
        assert "episodes" not in payload  # no multi-episode aggregate wrapper
        # Episode-count contract fields present on the fast path too.
        assert payload["n_episodes_requested"] == 1
        assert payload["n_episodes_completed"] == 1
        assert payload["episodes_saved"] == 0
        assert payload["dataset_episode_indices"] == []

    def test_explicit_n_episodes_one_matches_single(self, sim):
        result = sim.run_policy("arm1", n_steps=4, n_episodes=1)
        payload = _json(result)
        assert payload["n_steps"] == 4
        assert "episodes" not in payload


class TestMultiEpisodeValidation:
    @pytest.mark.parametrize("bad", [0, -1, -10])
    def test_non_positive_n_episodes_rejected(self, sim, bad):
        result = sim.run_policy("arm1", n_steps=5, n_episodes=bad)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "n_episodes must be a positive integer" in text
        assert str(bad) in text
        text.encode("ascii")  # ASCII-only contract

    def test_non_int_n_episodes_rejected(self, sim):
        result = sim.run_policy("arm1", n_steps=5, n_episodes=2.5)  # type: ignore[arg-type]
        assert result["status"] == "error"
        assert "n_episodes must be a positive integer" in result["content"][0]["text"]


class TestEpisodeBoundaryFlush:
    """The data-correctness fix: each rollout becomes its own dataset episode."""

    def test_recording_flushes_one_episode_per_rollout(self, tmp_path, sim):
        pytest.importorskip("lerobot")
        repo_id = "local/multi_ep_test"
        root = str(tmp_path / "ds")
        assert sim.start_recording(repo_id=repo_id, task="pick", fps=30, root=root)["status"] == "success"
        assert sim._is_recording() is True

        result = sim.run_policy("arm1", n_steps=6, n_episodes=4, control_frequency=30.0)
        assert result["status"] == "success", result
        payload = _json(result)
        # The whole point of #98: N rollouts -> N flushed episodes, not 1 merged.
        assert payload["episodes_saved"] == 4
        assert all(e.get("saved") for e in payload["episodes"])

        assert sim.stop_recording()["status"] == "success"

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id, root=root)
        assert ds.meta.total_episodes == 4
        assert ds.meta.total_frames == 24

    def test_no_recording_skips_flush(self, sim):
        # Without start_recording, the loop must not attempt save_episode.
        assert sim._is_recording() is False
        payload = _json(sim.run_policy("arm1", n_steps=3, n_episodes=2))
        assert payload["episodes_saved"] == 0


class TestResetBetween:
    def test_reset_between_false_chains_episodes(self, sim):
        # reset_between=False must still run all episodes successfully (the
        # sim simply continues from the previous end state).
        result = sim.run_policy("arm1", n_steps=4, n_episodes=3, reset_between=False)
        assert result["status"] == "success"
        assert _json(result)["n_episodes_completed"] == 3


class TestEpisodeVideoTemplating:
    """A reused video config must not let episodes overwrite one MP4."""

    def test_video_path_templated_per_episode(self):
        cfg0 = SimEngine._episode_video_config({"path": "/tmp/out.mp4", "fps": 30}, 0)
        cfg2 = SimEngine._episode_video_config({"path": "/tmp/out.mp4", "fps": 30}, 2)
        assert cfg0 is not None and cfg2 is not None
        assert cfg0.path == "/tmp/out_ep0.mp4"
        assert cfg2.path == "/tmp/out_ep2.mp4"
        # Non-path keys pass through.
        assert cfg0.fps == 30

    def test_no_video_passes_through(self):
        assert SimEngine._episode_video_config(None, 0) is None
        assert SimEngine._episode_video_config({}, 0) is None


class TestBaseHooks:
    """A backend with no recording support inherits safe base hooks."""

    def test_base_hooks_on_recordingless_backend(self):
        # A minimal SimEngine subclass that does NOT mix in RecordingMixin must
        # report no recording and refuse save_episode with a structured error,
        # so the multi-episode loop never tries to flush on such a backend.
        engine = _RecordinglessEngine()
        assert engine._is_recording() is False
        result = engine.save_episode()
        assert result["status"] == "error"
        assert "does not support dataset recording" in result["content"][0]["text"]


class TestMultiEpisodeErrorAbort:
    """A failure mid-loop aborts early and reports episodes completed so far.

    ``_run_episodes`` aborts on any of three failures - a rollout error, a
    dataset episode-flush error, or an inter-episode reset error - returning a
    structured error that names the failing episode, carries the underlying
    message, and never starts the remaining rollouts.
    """

    @staticmethod
    def _ok_rollout(_self, *_a, **_k):
        return {"status": "success", "content": [{"json": {"n_steps": 5}}]}

    def test_rollout_failure_aborts_remaining_episodes(self, sim, monkeypatch):
        calls = {"n": 0}

        def fake_run(_self, *_a, **_k):
            calls["n"] += 1
            if calls["n"] == 2:
                return {"status": "error", "content": [{"text": "boom: rollout exploded"}]}
            return {"status": "success", "content": [{"json": {"n_steps": 5}}]}

        _patch_runner(monkeypatch, fake_run)

        result = sim.run_policy("arm1", n_steps=5, n_episodes=3)

        assert result["status"] == "error"
        payload = _json(result)
        assert payload["n_episodes_requested"] == 3
        # Episode 0 succeeded; episode 1 failed and is recorded; episode 2 never ran.
        assert payload["n_episodes_completed"] == 2
        assert calls["n"] == 2
        assert payload["episodes"][0].get("status") != "error"
        assert payload["episodes"][1]["status"] == "error"
        # Only episode 0's steps count toward the aggregate.
        assert payload["total_steps"] == 5
        text = result["content"][0]["text"]
        assert "Episode 1 rollout failed" in text
        assert "aborting remaining 1 episode(s)" in text
        # The underlying rollout message is surfaced (exercises _first_text).
        assert "boom: rollout exploded" in text
        text.encode("ascii")  # ASCII-only contract

    def test_save_episode_failure_aborts(self, sim, monkeypatch):
        _patch_runner(monkeypatch, self._ok_rollout)
        # Pretend a recording is active so the loop attempts an episode flush,
        # then make that flush fail.
        monkeypatch.setattr(type(sim), "_is_recording", lambda _self: True)
        monkeypatch.setattr(
            type(sim),
            "save_episode",
            lambda _self: {"status": "error", "content": [{"text": "disk full"}]},
        )

        result = sim.run_policy("arm1", n_steps=5, n_episodes=3)

        assert result["status"] == "error"
        payload = _json(result)
        # Aborts right after episode 0's failed flush; nothing was saved.
        assert payload["n_episodes_completed"] == 1
        assert payload["episodes_saved"] == 0
        assert "disk full" in payload["episodes"][0]["save_episode_error"]
        text = result["content"][0]["text"]
        assert "save_episode failed after episode 0" in text
        assert "disk full" in text
        text.encode("ascii")

    def test_reset_failure_between_episodes_aborts(self, sim, monkeypatch):
        _patch_runner(monkeypatch, self._ok_rollout)
        monkeypatch.setattr(
            type(sim),
            "reset",
            lambda _self: {"status": "error", "content": [{"text": "reset boom"}]},
        )

        # reset_between defaults to True, so the post-episode-0 reset runs and fails.
        result = sim.run_policy("arm1", n_steps=5, n_episodes=2)

        assert result["status"] == "error"
        payload = _json(result)
        assert payload["n_episodes_completed"] == 1
        text = result["content"][0]["text"]
        assert "reset() failed after episode 0" in text
        assert "reset boom" in text
        text.encode("ascii")

    def test_final_episode_reset_failure_is_not_reached(self, sim, monkeypatch):
        # The last episode must NOT trigger an inter-episode reset, so a failing
        # reset on a single-iteration-past-last is never invoked: a clean
        # 2-episode run with reset_between=False completes despite a broken reset.
        _patch_runner(monkeypatch, self._ok_rollout)
        monkeypatch.setattr(
            type(sim),
            "reset",
            lambda _self: {"status": "error", "content": [{"text": "should not be called"}]},
        )

        result = sim.run_policy("arm1", n_steps=5, n_episodes=2, reset_between=False)

        assert result["status"] == "success"
        assert _json(result)["n_episodes_completed"] == 2

    def test_first_text_returns_empty_when_no_text_block(self):
        # The abort messages call _first_text on the failing result; when that
        # result carries no human-readable text (only json, or empty), the
        # helper degrades to "" rather than raising, so the aggregate message
        # still renders.
        assert SimEngine._first_text({"content": []}) == ""
        assert SimEngine._first_text({"content": [{"json": {"k": 1}}]}) == ""
        assert SimEngine._first_text({}) == ""
        assert SimEngine._first_text({"content": [{"text": "hello"}]}) == "hello"
