"""Regression: a GPU-pinned ``device_processor`` must not silently disable normalization.

A LeRobot checkpoint trained on GPU saves its preprocessor's ``device_processor``
step pinned to ``cuda``. On a CPU/MPS host (a Jetson whose driver predates the
checkpoint's CUDA build, a Mac, x86 CI) reconstructing that step raises because
``get_safe_torch_device`` asserts the device is available. Before the fix the
whole preprocessor was dropped and the bridge ran with NO input normalization --
observations stayed in raw space, so an ACT policy emitted near-constant actions
and the arm barely moved, while the cpu-pinned postprocessor loaded fine and made
the failure look like a model problem rather than a dropped device pin.

``ProcessorBridge`` now remaps the stale pin to the resolved inference device and
retries, so normalization is preserved without a manual ``processor_overrides``.

``torch.cuda.is_available`` is forced False so the cuda pin fails deterministically
on every host (including a real GPU runner), exercising the exact recovery path.
"""

import json

import pytest

pytest.importorskip("lerobot")

import torch

from strands_robots.policies.lerobot_local.processor import ProcessorBridge


def _build_cuda_pinned_single_cam_checkpoint(dest, state_mean=0.0, state_std=1.0):
    """Write a single-camera ACT preprocessor whose device_processor pins ``cuda``."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    cfg = ACTConfig(
        n_action_steps=100,
        chunk_size=100,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    )
    stats = {
        "observation.state": {"mean": torch.full((6,), state_mean), "std": torch.full((6,), state_std)},
        "observation.images.front": {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
        "action": {"mean": torch.zeros(6), "std": torch.ones(6)},
    }
    pre, post = make_act_pre_post_processors(cfg, dataset_stats=stats)
    pre.save_pretrained(str(dest))
    post.save_pretrained(str(dest))

    # Simulate a GPU-trained checkpoint: pin the device_processor step to cuda.
    cfg_path = dest / "policy_preprocessor.json"
    data = json.loads(cfg_path.read_text())
    pinned = False
    for step in data["steps"]:
        if step.get("registry_name") == "device_processor":
            step["config"]["device"] = "cuda"
            pinned = True
    assert pinned, "expected a device_processor step in the ACT preprocessor"
    cfg_path.write_text(json.dumps(data))


@pytest.fixture
def cuda_pinned_checkpoint(tmp_path, monkeypatch):
    """A single-camera ACT checkpoint with a cuda-pinned device_processor + no GPU."""
    _build_cuda_pinned_single_cam_checkpoint(tmp_path)
    # Force the cuda pin to be unavailable so the failure is reproduced on any host.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return tmp_path


def test_cuda_pinned_device_processor_is_remapped_so_normalization_survives(cuda_pinned_checkpoint):
    """The preprocessor must load on CPU even though it is pinned to cuda.

    Fails before the fix: the cuda pin raises, the bridge swallows it, and
    ``has_preprocessor`` is False (normalization silently dropped).
    """
    bridge = ProcessorBridge.from_pretrained(str(cuda_pinned_checkpoint), device="cpu", policy_type="act")
    assert bridge.has_preprocessor, "device pin should be remapped, not silently dropped"
    assert bridge.has_postprocessor


def test_remapped_preprocessor_actually_normalizes_observation_state(tmp_path, monkeypatch):
    """A remapped preprocessor must apply MEAN_STD normalization, not pass through.

    Built with state mean=2, std=4: a raw state of 10 must come back z-scored to
    (10 - 2) / 4 = 2.0. Before the fix the preprocessor is dropped, so the state
    would be returned in raw space (10.0) -- this asserts the numeric difference.
    """
    _build_cuda_pinned_single_cam_checkpoint(tmp_path, state_mean=2.0, state_std=4.0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    bridge = ProcessorBridge.from_pretrained(str(tmp_path), device="cpu", policy_type="act")
    assert bridge.has_preprocessor
    obs = {
        "observation.state": torch.full((6,), 10.0),
        "observation.images.front": torch.zeros(3, 480, 640),
    }
    out = bridge.preprocess(obs)
    state = out["observation.state"].squeeze()
    assert torch.allclose(state, torch.full((6,), 2.0), atol=1e-5), (
        f"state must be normalized (expected 2.0), got {state.tolist()}"
    )


def test_no_device_means_no_remap_and_preprocessor_stays_absent(cuda_pinned_checkpoint):
    """Without a target device there is nothing to remap to, so no false recovery."""
    bridge = ProcessorBridge.from_pretrained(str(cuda_pinned_checkpoint), device=None, policy_type="act")
    assert not bridge.has_preprocessor


def test_explicit_device_processor_override_is_respected(cuda_pinned_checkpoint):
    """A caller-supplied device_processor override is honored without auto-remap."""
    bridge = ProcessorBridge.from_pretrained(
        str(cuda_pinned_checkpoint),
        device="cpu",
        policy_type="act",
        overrides={"device_processor": {"device": "cpu"}},
    )
    assert bridge.has_preprocessor
