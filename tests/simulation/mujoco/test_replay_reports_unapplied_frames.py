"""Regression: replay reports failure when recorded frames never reach the robot.

``PolicyRunner.replay`` maps each recorded action-vector index onto an action key
and writes it through ``send_action``. ``send_action`` is explicit about failure:
keys it cannot resolve to an actuator or joint come back as ``status="error"``
with an ``unresolved_keys`` json block, precisely so callers can self-correct
instead of silently losing commands.

Replay discarded that result. The consequences were all silent:

* a typo'd / wrong-namespace ``action_key_map`` dropped EVERY recorded value at
  the actuator boundary, yet replay returned ``status="success"`` with
  ``Frames: N/N`` - the robot never moved;
* ``action_key_map="gripper"`` (a bare string) was consumed one key per
  character, so six single-letter keys resolved to nothing - again "success";
* a map shorter than the recorded vector positionally truncated the surplus
  DOFs (a 2-key map swallowing a 6-DOF recording's last four joints) and still
  reported a full-fidelity replay.

``run_policy`` already inspects ``send_action``'s status (it counts action
errors and fail-fasts on a rollout where nothing resolves). These tests pin the
same honesty for ``replay``: a success status means every frame was applied, and
an unapplied frame aborts with the frame index, the frames applied so far and
the backend's unresolved-key detail.
"""

from __future__ import annotations

import tempfile

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from strands_robots.simulation.policy_runner import PolicyRunner  # noqa: E402


@pytest.fixture(scope="module")
def recorded_so101():
    """A real 5-frame so101 recording plus its sim, shared by the tests below.

    Recording through the real recorder (no cameras -> action-only, fast) keeps
    the fixture honest: the ``action`` column really is written in actuator
    order, which is what replay maps back onto.
    """
    pytest.importorskip("lerobot")
    sim = Simulation()
    sim.create_world(ground_plane=True)
    sim.add_robot("so101")
    root = tempfile.mkdtemp(prefix="replay_unapplied_")
    repo = "local/replay_unapplied"
    assert sim.start_recording(repo_id=repo, task="rt", fps=30, root=root, cameras=[])["status"] == "success"
    assert (
        sim.run_policy(robot_name="so101", policy_provider="mock", n_steps=5, control_frequency=30, fast_mode=True)[
            "status"
        ]
        == "success"
    )
    assert sim.stop_recording()["status"] == "success"
    yield sim, repo, root, 5
    sim.cleanup()


def _replay(sim, repo, root, **kw):
    return PolicyRunner(sim).replay(repo, robot_name="so101", root=root, speed=1000.0, **kw)


def test_default_map_replays_every_frame(recorded_so101):
    """Control: the default (``robot_action_keys``) map still succeeds N/N.

    Without this the error assertions below could pass for a broken replay.
    """
    sim, repo, root, n_frames = recorded_so101
    result = _replay(sim, repo, root)
    assert result["status"] == "success", result
    payload = result["content"][1]["json"]
    assert payload["frames_applied"] == n_frames == payload["total_frames"]


def test_unresolvable_action_keys_abort_instead_of_reporting_success(recorded_so101):
    """Keys no actuator can absorb fail the replay and surface the valid keys.

    Pre-fix: ``status="success"``, ``Frames: 5/5``, robot motionless.
    """
    sim, repo, root, n_frames = recorded_so101
    valid = sim.robot_action_keys("so101")
    result = _replay(sim, repo, root, action_key_map=[f"not_{k}" for k in valid])

    assert result["status"] == "error", result
    text = result["content"][0]["text"]
    assert "frame 0" in text
    assert "Applied 0/5 frames" in text
    payload = result["content"][1]["json"]
    assert payload["frames_applied"] == 0
    assert payload["total_frames"] == n_frames
    # The backend's per-key breakdown is forwarded so the caller can self-correct.
    assert payload["unresolved_keys"] == [f"not_{k}" for k in valid]
    assert payload["applied"] == []


def test_map_shorter_than_recorded_vector_is_rejected(recorded_so101):
    """A too-short map would drop recorded DOFs; reject rather than truncate."""
    sim, repo, root, _ = recorded_so101
    valid = sim.robot_action_keys("so101")
    result = _replay(sim, repo, root, action_key_map=valid[:2])

    assert result["status"] == "error", result
    text = result["content"][0]["text"]
    assert f"{len(valid)} values" in text
    assert "2 action keys" in text
    assert result["content"][1]["json"]["recorded_action_width"] == len(valid)


def test_map_longer_than_recorded_vector_is_rejected(recorded_so101):
    """A too-long map leaves trailing keys unfed; reject it symmetrically."""
    sim, repo, root, _ = recorded_so101
    valid = sim.robot_action_keys("so101")
    result = _replay(sim, repo, root, action_key_map=[*valid, "surplus"])

    assert result["status"] == "error", result
    assert f"{len(valid) + 1} action keys" in result["content"][0]["text"]


@pytest.mark.parametrize(
    ("bad_map", "expected"),
    [
        ("gripper", "not a bare string"),
        (b"gripper", "not a bare string"),
        ({"1": "shoulder"}, "must be a list or tuple"),
        ([], "is empty"),
        (["1", 2, None], "non-string entries"),
        (["1", "2", "2", "3", "3", "4"], "duplicate keys"),
    ],
)
def test_malformed_action_key_map_rejected_before_dataset_load(bad_map, expected, monkeypatch):
    """Unusable map shapes are rejected up front, before any dataset download.

    A bare string is the sharpest case: ``list("gripper")`` yields one key per
    character, so every recorded value landed on a nonexistent single-letter
    actuator. The loader is monkeypatched to fail loudly if it is ever reached,
    pinning that a malformed map costs no multi-minute dataset fetch.
    """
    import strands_robots.dataset_recorder as dr

    def _must_not_load(*args, **kwargs):
        raise AssertionError("dataset loader reached despite a malformed action_key_map")

    monkeypatch.setattr(dr, "load_lerobot_episode", _must_not_load, raising=False)

    sim = Simulation()
    sim.create_world(ground_plane=True)
    sim.add_robot("so101")
    try:
        result = PolicyRunner(sim).replay("local/never_loaded", robot_name="so101", action_key_map=bad_map)
    finally:
        sim.cleanup()

    assert result["status"] == "error", result
    assert expected in result["content"][0]["text"]
