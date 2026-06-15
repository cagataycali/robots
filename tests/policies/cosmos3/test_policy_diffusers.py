"""Unit tests for the Cosmos3Policy in-process diffusers backend.

No GPU, no model weights, no policy server: ``use_diffusers`` is mocked via the
``use_diffusers_fn`` dependency-injection seam on
:class:`~strands_robots.policies.cosmos3.policy_diffusers.Cosmos3DiffusersBackend`
(mirroring the ``client=`` injection the service-backend tests use).
"""

import asyncio

import numpy as np
import pytest

from strands_robots.policies.base import Policy
from strands_robots.policies.cosmos3 import Cosmos3DiffusersBackend, Cosmos3Policy
from strands_robots.policies.cosmos3.embodiments import get_embodiment


def _droid_run_result(t=32, d=8, num_chunks=1, video="/tmp/world.mp4", sound=None):
    """Mock the dict shape strands-diffusers' use_diffusers(action='run') returns.

    The action field mirrors core.io._serialize_action: a nested-list chunk
    set of shape ``[num_chunks, T, D]`` normalized to [-1, 1].
    """
    rng = np.random.default_rng(0)
    chunks = [rng.uniform(-1.0, 1.0, (t, d)).astype(np.float32).tolist() for _ in range(num_chunks)]
    artifacts = []
    if video:
        artifacts.append(video)
    if sound:
        artifacts.append(sound)
    return {
        "status": "success",
        "content": [{"text": "ran Cosmos3OmniPipeline"}],
        "data": {
            "action": {"type": "action", "data": chunks, "chunk_shape": [t, d], "num_chunks": num_chunks},
            "video": video,
            "sound": sound,
        },
        "artifacts": artifacts,
    }


class FakeUseDiffusers:
    """Records calls; returns a canned condition + run result."""

    def __init__(self, run_result=None):
        self.run_result = run_result if run_result is not None else _droid_run_result()
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        action = kwargs.get("action")
        if action == "call":
            # CosmosActionCondition build -> cached object handle.
            return {
                "status": "success",
                "content": [{"text": "cond cached"}],
                "data": {"cached": kwargs.get("cache_key")},
            }
        if action == "run":
            return self.run_result
        if action == "clear_cache":
            return {"status": "success", "content": [{"text": "cleared"}]}
        return {"status": "success", "content": [{"text": "noop"}]}


def _obs_with_state_and_images():
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    obs = {
        "observation/wrist_image_left": img,
        "observation/exterior_image_1_left": img,
        "observation/exterior_image_2_left": img,
    }
    for i in range(7):
        obs[f"joint_{i}"] = float(i) * 0.1
    obs["gripper"] = 0.5
    return obs


def _make_diffusers_policy(fake=None, robot="panda", mode="policy", **kw):
    """Build a Cosmos3Policy(backend='diffusers') with an injected fake backend."""
    fake = fake or FakeUseDiffusers()
    backend = Cosmos3DiffusersBackend(
        embodiment=get_embodiment("droid"),
        mode=mode,
        use_diffusers_fn=fake,
    )
    p = Cosmos3Policy(embodiment="droid", backend="diffusers", diffusers_backend=backend, robot=robot, mode=mode, **kw)
    return p, fake


def test_diffusers_backend_is_a_policy():
    p, _ = _make_diffusers_policy()
    assert isinstance(p, Policy)
    assert p.provider_name == "cosmos3"
    assert p.backend == "diffusers"


def test_diffusers_returns_chunk_of_dicts_with_panda_actuators():
    """backend='diffusers' returns the same list[dict] shape with correct
    actuator names for robot='panda' (reusing _unpack_actions)."""
    p, fake = _make_diffusers_policy(robot="panda")
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    out = asyncio.run(p.get_actions(_obs_with_state_and_images(), "pick up the cube"))
    assert isinstance(out, list)
    assert len(out) == 32
    step = out[0]
    assert set(step.keys()) == {f"joint{i}" for i in range(1, 8)} | {"finger_joint1"}
    assert all(isinstance(v, float) for v in step.values())
    # use_diffusers was driven: a CosmosActionCondition 'call' then a 'run'.
    actions = [c["action"] for c in fake.calls]
    assert "call" in actions and "run" in actions


def test_last_rollout_carries_video_and_action():
    """The predicted world video/sound surface on last_rollout (non-breaking
    channel) - the get_actions return type stays list[dict]."""
    p, _ = _make_diffusers_policy()
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    assert p.last_rollout is None
    out = asyncio.run(p.get_actions(_obs_with_state_and_images(), "go"))
    assert isinstance(out, list) and len(out) == 32
    assert p.last_rollout is not None
    assert p.last_rollout["video"] == "/tmp/world.mp4"
    act = np.asarray(p.last_rollout["action"])
    assert act.shape == (32, 8)


def test_condition_params_use_embodiment_metadata():
    """The CosmosActionCondition call threads domain_name + chunk_size from the
    embodiment, and the run threads the cached condition + fps."""
    p, fake = _make_diffusers_policy()
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    asyncio.run(p.get_actions(_obs_with_state_and_images(), "go"))
    call = next(c for c in fake.calls if c["action"] == "call")
    params = call["parameters"]
    assert params["mode"] == "policy"
    assert params["domain_name"] == "droid_lerobot"
    assert params["chunk_size"] == 32
    assert "image" in params  # policy mode conditions on the first frame
    run = next(c for c in fake.calls if c["action"] == "run")
    assert run["pipeline"] == "Cosmos3OmniPipeline"
    assert run["parameters"]["action"].startswith("cached:")
    assert run["parameters"]["fps"] == 15


def test_service_backend_byte_identical_regression():
    """backend='service' (default) path is unchanged: it never touches the
    diffusers backend and returns the service action chunk verbatim."""

    class FakeClient:
        def __init__(self, action):
            self._action = action
            self.last_obs = None

        def infer(self, observation):
            self.last_obs = observation
            return {"action": self._action}

        def reset(self):
            pass

    action = np.arange(32 * 8, dtype=np.float32).reshape(32, 8)
    p = Cosmos3Policy(embodiment="droid", client=FakeClient(action.copy()), robot="panda")
    assert p.backend == "service"
    assert p.last_rollout is None  # service never populates the world channel
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    out = p.get_actions_sync(_obs_with_state_and_images(), "go")
    # Reconstruct the chunk from the per-step dicts and compare to the input.
    cols = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1"]
    recon = np.asarray([[step[c] for c in cols] for step in out], dtype=np.float32)
    np.testing.assert_array_equal(recon, action)
    assert p.last_rollout is None


def test_missing_strands_diffusers_raises_actionable_error(monkeypatch):
    """When strands-diffusers is not importable, constructing the diffusers
    backend raises an actionable install error (no silent default)."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "strands_diffusers" or name.startswith("strands_diffusers."):
            raise ImportError("No module named 'strands_diffusers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="strands-diffusers"):
        Cosmos3DiffusersBackend(embodiment=get_embodiment("droid"))


def test_forward_dynamics_under_service_raises():
    """mode='forward_dynamics' under backend='service' raises a clear
    unsupported error (the RoboLab server serves only the policy surface)."""
    with pytest.raises(ValueError, match="only available with backend='diffusers'"):
        Cosmos3Policy(embodiment="droid", backend="service", mode="forward_dynamics")


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown Cosmos 3 backend"):
        Cosmos3Policy(embodiment="droid", backend="grpc")


def test_unknown_mode_raises_in_backend():
    with pytest.raises(ValueError, match="Unknown Cosmos 3 action mode"):
        Cosmos3DiffusersBackend(
            embodiment=get_embodiment("droid"), mode="teleport", use_diffusers_fn=FakeUseDiffusers()
        )


def test_forward_dynamics_world_only_returns_empty_but_keeps_video():
    """forward_dynamics predicts world video only (no action chunk). get_actions
    returns [] but the world video is still captured on last_rollout."""
    fake = FakeUseDiffusers(run_result=_droid_run_result(video="/tmp/fd.mp4"))
    # Strip the action field so the run looks world-only.
    fake.run_result["data"]["action"] = None
    p, _ = _make_diffusers_policy(fake=fake, mode="forward_dynamics")
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    raw = np.zeros((32, 8), dtype=np.float32)
    out = asyncio.run(p.get_actions(_obs_with_state_and_images(), "roll forward", raw_actions=raw))
    assert out == []
    assert p.last_rollout["video"] == "/tmp/fd.mp4"
    assert p.last_rollout["action"] is None


def test_forward_dynamics_requires_raw_actions():
    p, _ = _make_diffusers_policy(mode="forward_dynamics")
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    with pytest.raises(ValueError, match="raw_actions"):
        asyncio.run(p.get_actions(_obs_with_state_and_images(), "go"))


def test_inverse_dynamics_requires_video():
    p, _ = _make_diffusers_policy(mode="inverse_dynamics")
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    with pytest.raises(ValueError, match="observed video"):
        asyncio.run(p.get_actions(_obs_with_state_and_images(), "go"))


def test_run_failure_raises_with_tool_text():
    """A use_diffusers error result is surfaced, not silently swallowed."""
    fake = FakeUseDiffusers(run_result={"status": "error", "content": [{"text": "CUDA OOM boom"}]})
    p, _ = _make_diffusers_policy(fake=fake)
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    with pytest.raises(RuntimeError, match="CUDA OOM boom"):
        asyncio.run(p.get_actions(_obs_with_state_and_images(), "go"))


def test_reset_clears_cached_condition():
    fake = FakeUseDiffusers()
    p, _ = _make_diffusers_policy(fake=fake)
    p.reset(seed=3)
    assert any(c["action"] == "clear_cache" for c in fake.calls)
