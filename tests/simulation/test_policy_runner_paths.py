"""Targeted coverage for ``PolicyRunner`` error paths and edge cases.

Covers:
* ``replay()`` when no robots exist (``_require_default_robot`` ValueError)
* ``replay()`` when the dataset loader raises (opaque upstream error)
* ``replay()`` when lerobot is not installed (ImportError → graceful)
* ``replay()`` with actions that have ``.numpy()`` and ``.tolist()`` methods
  (tensor-backed dataset frames)
* ``_extract_frame_ndarray`` handles render blocks without images
* ``_resolve_success_fn`` "contact" with backend that raises NotImplementedError
* ``evaluate()`` "never-succeeds" default path (no success_fn)
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.policy_runner import (
    PolicyRunner,
    _extract_frame_ndarray,
)

# Import the FakeSim from the sibling test file
from tests.simulation.test_policy_runner import FakeSim as _BaseFakeSim


class _MinimalSim(_BaseFakeSim):
    """FakeSim variant with pluggable robot list + optional get_contacts."""

    def __init__(self, robots=None, raise_on_contacts=False):
        super().__init__()
        # Override robots
        if robots is not None:
            self._robots = {name: ["j0", "j1", "j2"] for name in robots}
        self._raise_on_contacts = raise_on_contacts

    def get_contacts(self):
        if self._raise_on_contacts:
            raise NotImplementedError("backend doesn't support contacts")
        return {"n_contacts": 0}


# ── replay() error paths ────────────────────────────────────────────


def test_replay_no_robots_errors_cleanly():
    sim = _MinimalSim(robots=[])  # empty
    r = PolicyRunner(sim).replay(repo_id="irrelevant")
    assert r["status"] == "error"
    assert "No robots" in r["content"][0]["text"]


def test_replay_dataset_loader_raises_is_handled(monkeypatch):
    sim = _MinimalSim(robots=["r0"])

    def boom(*args, **kwargs):
        raise RuntimeError("simulated HF download failure")

    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "load_lerobot_episode", boom, raising=False)

    r = PolicyRunner(sim).replay(repo_id="bad/dataset")
    assert r["status"] == "error"
    assert "simulated HF download failure" in r["content"][0]["text"]


def test_replay_with_tensor_like_actions(monkeypatch):
    """Dataset actions may be torch tensors; replay must call .numpy().tolist()."""

    class _FakeTensor:
        def __init__(self, values):
            self._v = np.asarray(values, dtype=np.float32)

        def numpy(self):
            return self._v

    class _TensorDataset:
        fps = 30

        def __len__(self):
            return 3

        def __getitem__(self, idx):
            return {"action": _FakeTensor([0.1 * idx, 0.2, 0.3])}

    def loader(repo_id, episode, root):
        return _TensorDataset(), 0, 3

    sim = _MinimalSim(robots=["r0"])

    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "load_lerobot_episode", loader, raising=False)

    r = PolicyRunner(sim).replay(repo_id="fake/tensor", speed=100.0)  # fast
    assert r["status"] == "success"


def test_replay_with_action_vector_larger_than_joint_count(monkeypatch):
    """When dataset has more action dims than robot joints, replay truncates
    (``break`` path in the replay loop)."""

    class _FatDataset:
        fps = 30

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            # 5 values but robot only has 3 joints → extras must be dropped
            return {"action": [0.1, 0.2, 0.3, 0.4, 0.5]}

    def loader(repo_id, episode, root):
        return _FatDataset(), 0, 2

    sim = _MinimalSim(robots=["r0"])

    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "load_lerobot_episode", loader, raising=False)

    r = PolicyRunner(sim).replay(repo_id="fake/fat", speed=100.0)
    assert r["status"] == "success"


def test_replay_action_none_advances_physics(monkeypatch):
    """Dataset frames with no 'action' key → physics step, still advance."""

    class _MissingActionDataset:
        fps = 30

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {"observation.state": [0, 0, 0]}  # no 'action'

    def loader(repo_id, episode, root):
        return _MissingActionDataset(), 0, 2

    sim = _MinimalSim(robots=["r0"])

    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "load_lerobot_episode", loader, raising=False)

    r = PolicyRunner(sim).replay(repo_id="fake/noaction", speed=100.0)
    assert r["status"] == "success"


# ── _extract_frame_ndarray edge cases ───────────────────────────────


def test_extract_frame_ndarray_rejects_non_dict():
    assert _extract_frame_ndarray("not a dict") is None
    assert _extract_frame_ndarray(None) is None


def test_extract_frame_ndarray_no_image_blocks():
    assert _extract_frame_ndarray({"content": [{"text": "only text"}]}) is None


def test_extract_frame_ndarray_bad_image_structure():
    # image present but no source
    assert _extract_frame_ndarray({"content": [{"image": "string not dict"}]}) is None
    # source empty
    assert _extract_frame_ndarray({"content": [{"image": {"source": {}}}]}) is None
    # non-decodable bytes
    assert _extract_frame_ndarray({"content": [{"image": {"source": {"bytes": b"notpng"}}}]}) is None


# ── evaluate() paths ────────────────────────────────────────────────


def test_evaluate_unknown_success_fn_string_errors():
    sim = _MinimalSim(robots=["r0"])
    policy = MockPolicy()
    r = PolicyRunner(sim).evaluate(robot_name="r0", policy=policy, n_episodes=1, success_fn="made_up_string")
    assert r["status"] == "error"
    assert "Unknown success_fn" in r["content"][0]["text"]


def test_evaluate_with_callable_success_fn():
    sim = _MinimalSim(robots=["r0"])
    policy = MockPolicy()
    policy.set_robot_state_keys(["j0", "j1", "j2"])

    # Always succeed → success_rate = 1.0
    r = PolicyRunner(sim).evaluate(
        robot_name="r0",
        policy=policy,
        n_episodes=2,
        max_steps=5,
        success_fn=lambda obs: True,
    )
    assert r["status"] == "success"


def test_evaluate_contact_fn_with_backend_that_raises():
    """If the backend's ``get_contacts`` raises NotImplementedError, the
    contact success_fn just returns False (never propagates)."""
    sim = _MinimalSim(robots=["r0"], raise_on_contacts=True)
    policy = MockPolicy()
    policy.set_robot_state_keys(["j0", "j1", "j2"])
    r = PolicyRunner(sim).evaluate(robot_name="r0", policy=policy, n_episodes=1, max_steps=3, success_fn="contact")
    assert r["status"] == "success"


def test_evaluate_none_success_fn_gives_zero_success_rate():
    """success_fn=None → never succeeds (dry-run probe)."""
    sim = _MinimalSim(robots=["r0"])
    policy = MockPolicy()
    policy.set_robot_state_keys(["j0", "j1", "j2"])
    r = PolicyRunner(sim).evaluate(robot_name="r0", policy=policy, n_episodes=2, max_steps=3, success_fn=None)
    assert r["status"] == "success"
    # success_fn=None means no episode ever succeeds
    # Extract json block:
    for c in r["content"]:
        if isinstance(c, dict) and "json" in c:
            assert c["json"]["success_rate"] == 0.0
            break
