"""A randomly initialised S1V decider drives the ``Policy`` seam without any network.

:class:`~strands_robots.policies.s1v.policy.S1VPolicy` composes three things a
unit test must be able to hold apart: the frozen DINOv2-small features (a
download), the trained decider (a checkpoint) and the primitive-to-action step
(pure arithmetic). Here the decider is a tiny random :class:`S1VDecider` saved
with ``save_pretrained`` and the backbone is a stand-in that returns zeros of
the right shape, so the test exercises the whole tick - joint keys read from the
observation, radians to degrees/percent, one primitive chosen, the setpoint
integrator, anti-windup, a complete action dict - on CPU in well under a second.

What the tick must guarantee, whatever the random weights decide:

* every joint key of the observation appears in the one returned action;
* at most ONE joint target differs from the previous setpoint (one primitive
  per tick is the whole point of the typed-decision vocabulary);
* an arm target never leads the measured joint by more than the windup limit;
* the choice heads are candidate-invariant: the decision does not depend on
  which primitive the noul heads were asked about.
"""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from strands_robots.policies.s1v import dataset as s1v_dataset  # noqa: E402
from strands_robots.policies.s1v.model import CANDIDATE_NONE, S1VConfig, S1VDecider  # noqa: E402
from strands_robots.policies.s1v.policy import WINDUP_MAX_RAD, S1VPolicy  # noqa: E402

_KEYS = ("1", "2", "3", "4", "5", "6")


class _ZeroBackbone:
    """Stands in for DINOv2-small: ``featurize_images`` only needs a callable with ``last_hidden_state``."""

    def __call__(self, pixel_values):
        b = pixel_values.shape[0]
        return type("Out", (), {"last_hidden_state": torch.zeros(b, 1 + 256, 384)})()


def _fake_backbone(device: str = "cpu"):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return _ZeroBackbone(), mean, std


@pytest.fixture
def tiny_checkpoint(tmp_path, monkeypatch):
    torch.manual_seed(0)
    cfg = S1VConfig(d_model=32, n_heads=2, ffn=64, n_layers=1, grid=2)
    S1VDecider(cfg).save_pretrained(tmp_path / "ckpt")
    monkeypatch.setattr(s1v_dataset, "load_backbone", _fake_backbone)
    return tmp_path / "ckpt"


def _observation(rng: np.random.Generator) -> dict:
    obs = {k: float(v) for k, v in zip(_KEYS, [0.0, -1.5, 1.4, 0.9, 0.0, 0.3], strict=True)}
    obs["scene"] = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    obs["wrist"] = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    return obs


def test_one_tick_changes_at_most_one_joint_and_returns_every_key(tiny_checkpoint):
    policy = S1VPolicy(str(tiny_checkpoint), task="pick", device="cpu")
    policy.set_robot_state_keys(list(_KEYS))
    rng = np.random.default_rng(1)
    obs = _observation(rng)
    first = asyncio.run(policy.get_actions(obs, ""))
    assert len(first) == 1 and set(first[0]) == set(_KEYS)
    assert all(isinstance(v, float) for v in first[0].values())
    changed = [k for k in _KEYS if not math.isclose(first[0][k], obs[k], abs_tol=1e-9)]
    assert len(changed) <= 1, changed
    second = asyncio.run(policy.get_actions(obs, ""))[0]
    moved = [k for k in _KEYS if not math.isclose(second[k], first[0][k], abs_tol=1e-9)]
    assert len(moved) <= 1, moved
    for k in _KEYS[:5]:
        assert abs(second[k] - obs[k]) <= WINDUP_MAX_RAD + 1e-9


def test_reset_forgets_the_setpoint_and_task_is_validated(tiny_checkpoint):
    with pytest.raises(ValueError, match="task must be one of"):
        S1VPolicy(str(tiny_checkpoint), task="juggle", device="cpu")
    policy = S1VPolicy(str(tiny_checkpoint), task="reach", device="cpu")
    rng = np.random.default_rng(2)
    obs = _observation(rng)
    asyncio.run(policy.get_actions(obs, ""))
    policy.reset()
    assert policy._setpoint is None
    assert len(policy.tick_ms) == 1


def test_choice_heads_are_candidate_invariant(tiny_checkpoint):
    model = S1VDecider.from_pretrained(tiny_checkpoint)
    cams = torch.randn(2, model.n_cam_tokens, 384)
    state = torch.randn(2, 6)
    task = torch.tensor([0, 1])
    none = torch.full((2,), CANDIDATE_NONE)
    some = torch.tensor([3, 17])
    with torch.inference_mode():
        a = model(cams, state, task, none)
        b = model(cams, state, task, some)
    for head in ("joint", "direction", "size"):
        assert torch.allclose(a[head], b[head], atol=1e-5), head
    assert not torch.allclose(a["safe"], b["safe"]), "the noul heads must see the candidate"


def test_preflight_names_the_missing_camera():
    with pytest.raises(ValueError, match="wrist='gripper_cam'"):
        S1VPolicy.preflight({"1", "2", "scene"}, camera_map={"wrist": "gripper_cam"})
