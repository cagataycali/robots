# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A camera at the wrong resolution must not silently change the policy's output.

Nothing in this stack compared an incoming frame's H/W against the shape the
checkpoint declares (``observation.images.laptop: {shape: [3, 480, 640]}``), and
lerobot 0.6.0's only registered resize step
(``hil_processor.image_crop_resize_processor``) is NOT included by
``make_pre_post_processors`` - so the CALLER owns the resize and this caller did
not do it. ACT's ResNet backbone plus its spatial-softmax head accepts any input
size, so a 1080p or 320x240 camera ran to completion and returned different
actions with no exception and no warning.

Measured on the real cached ACT SO-101 checkpoint, ONE deterministic scene
antialias-rescaled to three resolutions, ``reset()`` between calls::

    BEFORE  480x640 107.584 | 1080x1920 128.526 | 240x320 106.539   spread 21.99 deg
    AFTER   480x640 107.584 | 1080x1920 109.319 | 240x320 117.406   spread  9.82 deg

Frames are now resized to the declared shape (bilinear, antialias), matching what
the training dataloader saw, with a once-per-camera INFO log - or a raise under
``strict_keys`` for callers who would rather fix the camera config.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

_FEATURE_KEY = "observation.images.laptop"


def _visual(shape: tuple[int, ...]) -> SimpleNamespace:
    return SimpleNamespace(type=SimpleNamespace(name="VISUAL"), shape=shape)


def _policy(features: dict | None = None, **kwargs) -> LerobotLocalPolicy:
    policy = LerobotLocalPolicy(**kwargs)
    policy._input_features = features if features is not None else {_FEATURE_KEY: _visual((3, 480, 640))}
    return policy


def _frame(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestMismatchedResolutionIsResized:
    @pytest.mark.parametrize("height,width", [(1080, 1920), (240, 320), (224, 224)])
    def test_the_frame_reaches_the_declared_shape(self, height, width):
        """Regression: a differently-sized frame used to pass through untouched."""
        out = _policy()._canonicalize_obs_images({_FEATURE_KEY: _frame(height, width)})

        assert tuple(out[_FEATURE_KEY].shape) == (3, 480, 640)

    def test_a_matching_frame_is_untouched(self):
        policy = _policy()
        frame = _frame(480, 640)

        out = policy._canonicalize_obs_images({_FEATURE_KEY: frame})

        assert tuple(out[_FEATURE_KEY].shape) == (3, 480, 640)
        assert policy._image_resize_warned == set(), "a matching frame must not log"

    def test_the_resize_is_logged_once_per_camera(self, caplog):
        policy = _policy()

        with caplog.at_level(logging.INFO):
            for _ in range(5):
                policy._canonicalize_obs_images({_FEATURE_KEY: _frame(240, 320)})

        notices = [r.getMessage() for r in caplog.records if "resizing every frame" in r.getMessage()]
        assert len(notices) == 1, notices
        assert "240x320" in notices[0]
        assert "480x640" in notices[0]

    def test_the_log_message_is_plain_ascii(self, caplog):
        """AGENTS.md: user-facing strings are plain ASCII only."""
        with caplog.at_level(logging.INFO):
            _policy()._canonicalize_obs_images({_FEATURE_KEY: _frame(240, 320)})

        for record in caplog.records:
            assert record.getMessage().isascii()

    def test_the_output_is_still_chw_float_in_unit_range(self):
        """The resize must not undo the canonicalization it follows."""
        frame = np.full((240, 320, 3), 255, dtype=np.uint8)

        out = _policy()._canonicalize_obs_images({_FEATURE_KEY: frame})[_FEATURE_KEY]

        assert out.dtype == torch.float32
        assert out.shape[0] == 3
        assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


class TestStrictKeysRaises:
    def test_a_mismatch_raises_naming_both_shapes(self):
        policy = _policy(strict_keys=True)

        with pytest.raises(ValueError) as excinfo:
            policy._canonicalize_obs_images({_FEATURE_KEY: _frame(240, 320)})

        message = str(excinfo.value)
        assert "240x320" in message
        assert "480x640" in message
        assert "strict_keys" in message

    def test_a_matching_frame_does_not_raise_under_strict(self):
        policy = _policy(strict_keys=True)

        assert policy._canonicalize_obs_images({_FEATURE_KEY: _frame(480, 640)}) is not None


class TestNoDeclaredShapeMeansNoCheck:
    def test_empty_input_features_is_passthrough(self):
        policy = _policy(features={})

        out = policy._canonicalize_obs_images({"observation.images.x": _frame(240, 320)})

        assert tuple(out["observation.images.x"].shape) == (3, 240, 320)

    def test_a_feature_without_a_spatial_shape_is_ignored(self):
        policy = _policy(features={_FEATURE_KEY: _visual((3,))})

        out = policy._canonicalize_obs_images({_FEATURE_KEY: _frame(240, 320)})

        assert tuple(out[_FEATURE_KEY].shape) == (3, 240, 320)

    def test_an_unknown_key_is_ignored(self):
        policy = _policy()

        out = policy._canonicalize_obs_images({"observation.images.other": _frame(240, 320)})

        assert tuple(out["observation.images.other"].shape) == (3, 240, 320)


class TestBareSourceKeysAreResolved:
    def test_a_key_renamed_by_the_embodiment_is_checked(self):
        """In the declarative path the rename runs INSIDE the pipeline.

        So the frame arrives under the robot-native key (``front``), and the
        declared shape has to be resolved through the embodiment's obs_rename.
        """
        from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap

        policy = _policy()
        policy._embodiment = EmbodimentMap(
            name="test",
            obs_rename={"front": _FEATURE_KEY},
            state_keys=[],
            action_keys=[],
            dim_policy="pad",
        )

        out = policy._canonicalize_obs_images({"front": _frame(240, 320)}, image_source_keys={"front"})

        assert tuple(out["front"].shape) == (3, 480, 640)

    def test_declared_shape_lookup_returns_none_without_an_embodiment(self):
        assert _policy()._declared_image_shape("front") is None
