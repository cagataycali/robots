# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Declarative path must drive a single-camera SO-101 ACT checkpoint.

Setting joint names via ``set_robot_state_keys`` (no explicit ``embodiment``)
synthesizes a state-only embodiment so the declarative pipeline composes
``observation.state``. That path previously could not process a single-camera
observation:

* The synthesized embodiment had an empty ``obs_rename``, so the pipeline never
  renamed the robot's bare ``front`` key onto the model's declared
  ``observation.images.front`` feature -> ``KeyError`` in the model.
* ``_canonicalize_obs_images`` keyed off the ``"image"`` substring, so a bare
  ``front`` frame was left as HWC ``uint8`` -> the normalizer overflowed.
* ``PackStateProcessorStep`` emitted ``observation.state`` as a numpy array,
  which the pipeline's ``AddBatchDimensionObservationStep`` does not batch (it
  only batches 1-D tensors), so state stayed ``(D,)`` while images were
  ``(1, C, H, W)`` -> a ``torch.stack`` rank mismatch inside the model.

A multi-camera embodiment (e.g. ``so_real`` declares ``front`` + ``wrist``)
also could not be adapted to a single-camera checkpoint: ``obs_rename_override``
merged OVER the declared map but could not DROP the stale ``wrist`` rename whose
target the model never declares, so ``validate`` rejected it.

These tests pin both behaviors. The unit tests are deterministic (no network);
the end-to-end test runs a real (network-free) LeRobot pipeline.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from strands_robots.policies.lerobot_local.embodiment import load_embodiment
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy, _merge_obs_rename


def _visual(shape=(3, 480, 640)):
    return SimpleNamespace(type=SimpleNamespace(name="VISUAL"), shape=shape)


def _state(dim=6):
    return SimpleNamespace(type=SimpleNamespace(name="STATE"), shape=(dim,))


def _action(dim=6):
    return SimpleNamespace(type=SimpleNamespace(name="ACTION"), shape=(dim,))


def _single_camera_policy(image_feature="observation.images.front", camera_key_map=None):
    """A loaded policy declaring one camera + 6-dof state, no model download."""
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path=None, policy_type="act", camera_key_map=camera_key_map)
    policy._input_features = {"observation.state": _state(6), image_feature: _visual()}
    policy._output_features = {"action": _action(6)}
    policy._loaded = True
    return policy


class TestSynthesizedEmbodimentRoutesCamera:
    """F3: state-only synthesized embodiment must route the single camera."""

    def test_synthesized_obs_rename_maps_declared_image_feature(self):
        policy = _single_camera_policy("observation.images.front")
        policy.set_robot_state_keys([f"j{i}" for i in range(6)])
        renames = policy._synthesized_camera_renames()
        # Bare 'front' must route onto the model's declared image feature.
        assert renames == {"front": "observation.images.front"}

    def test_camera_key_map_wins_in_synthesis(self):
        policy = _single_camera_policy(
            "observation.images.front", camera_key_map={"realsense": "observation.images.front"}
        )
        policy.set_robot_state_keys([f"j{i}" for i in range(6)])
        renames = policy._synthesized_camera_renames()
        assert renames == {"realsense": "observation.images.front"}

    def test_state_only_model_has_no_camera_renames(self):
        with patch.object(LerobotLocalPolicy, "_load_model"):
            policy = LerobotLocalPolicy(pretrained_name_or_path=None, policy_type="act")
        policy._input_features = {"observation.state": _state(6)}
        policy._output_features = {"action": _action(6)}
        policy._loaded = True
        policy.set_robot_state_keys([f"j{i}" for i in range(6)])
        assert policy._synthesized_camera_renames() == {}


class TestCanonicalizeBareCameraKey:
    """F3: a bare (pre-rename) camera key must be canonicalized to CHW float."""

    def test_bare_key_converted_when_listed_in_image_keys(self):
        policy = _single_camera_policy()
        obs = {"front": (np.ones((480, 640, 3), dtype=np.uint8) * 255)}
        out = policy._canonicalize_obs_images(obs, image_source_keys={"front"})
        front = out["front"]
        assert front.dtype.is_floating_point  # was uint8
        assert tuple(front.shape) == (3, 480, 640)  # was HWC
        assert float(front.max()) <= 1.0  # normalized from [0,255]

    def test_bare_key_untouched_without_image_keys(self):
        # Default behavior preserved: a bare key not flagged stays raw.
        policy = _single_camera_policy()
        obs = {"front": np.ones((480, 640, 3), dtype=np.uint8)}
        out = policy._canonicalize_obs_images(obs)
        assert out["front"].dtype == np.uint8


class TestDropStaleObsRename:
    """F1: obs_rename_override must be able to DROP a stale source key."""

    def test_falsy_override_drops_key(self):
        base = {"front": "observation.images.image", "wrist": "observation.images.wrist_image"}
        assert _merge_obs_rename(base, {"wrist": None}) == {"front": "observation.images.image"}
        assert _merge_obs_rename(base, {"wrist": ""}) == {"front": "observation.images.image"}

    def test_truthy_override_still_sets(self):
        base = {"front": "observation.images.image"}
        merged = _merge_obs_rename(base, {"front": "observation.images.front", "wrist": None})
        assert merged == {"front": "observation.images.front"}

    def test_multi_camera_embodiment_adapts_to_single_camera_model(self):
        # so_real declares front + wrist; a single-camera checkpoint declares
        # only observation.images.image. Dropping wrist lets validate pass.
        embodiment = load_embodiment("so_real")
        assert "wrist" in embodiment.obs_rename  # precondition
        merged = _merge_obs_rename(embodiment.obs_rename, {"wrist": None})
        single_cam = replace(embodiment, obs_rename=merged)
        input_features = {"observation.state": _state(6), "observation.images.image": _visual()}
        output_features = {"action": _action(6)}
        # Pre-fix (wrist retained) this raised: target observation.images.wrist_image
        # is not a declared model feature.
        single_cam.validate(input_features, output_features)

    def test_stale_key_retained_fails_validation(self):
        # Documents the failure the drop fixes: keeping wrist rejects the model.
        embodiment = load_embodiment("so_real")
        input_features = {"observation.state": _state(6), "observation.images.image": _visual()}
        output_features = {"action": _action(6)}
        with pytest.raises(ValueError, match="wrist"):
            embodiment.validate(input_features, output_features)


class TestPreflightHonorsDrop:
    """F1: the pre-flight check must also honor a dropped key."""

    def test_preflight_passes_when_stale_camera_dropped(self):
        # front present, wrist absent at runtime; dropping wrist must satisfy
        # preflight for a model whose only image feature is fed by front.
        LerobotLocalPolicy.preflight(
            {"front", "shoulder_pan.pos"},
            embodiment="so_real",
            obs_rename_override={"wrist": None, "front": "observation.images.image"},
        )


class TestDeclarativeSingleCameraEndToEnd:
    """F3 end-to-end against a real (network-free) LeRobot pipeline."""

    def test_real_pipeline_renames_normalizes_and_batches(self):
        pytest.importorskip("lerobot.processor.pipeline")
        from lerobot.processor import AddBatchDimensionProcessorStep, RenameObservationsProcessorStep
        from lerobot.processor.pipeline import DataProcessorPipeline

        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        policy = _single_camera_policy("observation.images.front")
        pipe = DataProcessorPipeline(
            steps=[RenameObservationsProcessorStep(rename_map={}), AddBatchDimensionProcessorStep()]
        )
        policy._processor_bridge = ProcessorBridge(preprocessor=pipe, device="cpu")
        policy.set_robot_state_keys([f"j{i}" for i in range(6)])
        policy._configure_embodiment()

        obs = {f"j{i}": float(i) for i in range(6)}
        obs["front"] = np.ones((48, 64, 3), dtype=np.uint8) * 255
        obs = policy._canonicalize_obs_images(obs, image_source_keys=policy._embodiment_image_source_keys())
        batch = policy._processor_bridge.preprocess(obs, instruction="pick up the cube")

        # Camera renamed onto the model feature, normalized float, batched.
        assert "observation.images.front" in batch
        assert "front" not in batch
        img = batch["observation.images.front"]
        assert tuple(img.shape) == (1, 3, 48, 64)
        assert img.dtype.is_floating_point
        assert float(img.max()) <= 1.0

        # State composed AND batched to (1, D) like the image (no rank mismatch).
        state = batch["observation.state"]
        assert tuple(state.shape) == (1, 6)
        assert state.dtype.is_floating_point
