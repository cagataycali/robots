# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Image-feature detection must use the declared FeatureType, not a name substring.

A camera observation is routed to a model image slot only if the slot is
recognized as an image feature. Detecting that by the ``"image" in key``
substring silently misclassifies a declared ``FeatureType.VISUAL`` feature
whose key does not contain the literal ``"image"`` - e.g. MolmoAct2 declares
image feature keys such as ``base``/``wrist``. The frames were then dropped
from the remapped observation and surfaced later as a confusing
``image_keys missing from observation`` failure inside the preprocessor.

These tests pin that image features are detected by their authoritative
``PolicyFeature.type`` (``VISUAL``), with the name convention only as a
no-type-metadata fallback.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from strands_robots.policies.lerobot_local.policy import (
    LerobotLocalPolicy,
    _declared_feature_is_image,
)


def _visual(shape=(3, 224, 224)):
    return SimpleNamespace(type=SimpleNamespace(name="VISUAL"), shape=shape)


def _state(dim=6):
    return SimpleNamespace(type=SimpleNamespace(name="STATE"), shape=(dim,))


def _policy(input_features):
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path=None, policy_type="molmoact2")
    policy._input_features = input_features
    policy.robot_state_keys = [f"j{i}" for i in range(6)]
    return policy


def _obs():
    obs = {"base": np.zeros((224, 224, 3), np.uint8), "wrist": np.zeros((224, 224, 3), np.uint8)}
    obs.update({f"j{i}": 0.0 for i in range(6)})
    return obs


def _routed_images(out):
    return sorted(k for k, v in out.items() if isinstance(v, np.ndarray) and v.ndim >= 2)


class TestDeclaredFeatureIsImage:
    def test_prefers_visual_type_over_name(self):
        # Bare key with VISUAL type IS an image; STATE never is.
        assert _declared_feature_is_image("base", _visual()) is True
        assert _declared_feature_is_image("observation.state", _state()) is False
        # A non-image key whose name contains "image" is still not visual.
        assert _declared_feature_is_image("observation.image_token_mask", _state()) is False

    def test_name_fallback_without_type_metadata(self):
        assert _declared_feature_is_image("observation.images.top", None) is True
        assert _declared_feature_is_image("observation.state", None) is False


class TestVisualFeatureRouting:
    def test_bare_visual_keys_route_camera_frames(self):
        # Regression: bare-named VISUAL slots (MolmoAct2 base/wrist) must receive
        # the camera frames, not be dropped because the key lacks "image".
        policy = _policy({"base": _visual(), "wrist": _visual(), "observation.state": _state()})
        out = policy._to_lerobot_observation(_obs())
        assert _routed_images(out) == ["base", "wrist"]

    def test_qualified_image_keys_still_route(self):
        # No regression for the conventional observation.images.* naming.
        policy = _policy(
            {
                "observation.images.base": _visual(),
                "observation.images.wrist": _visual(),
                "observation.state": _state(),
            }
        )
        out = policy._to_lerobot_observation(_obs())
        assert _routed_images(out) == ["observation.images.base", "observation.images.wrist"]

    def test_policy_image_keys_includes_bare_visual_slots(self):
        policy = _policy({"base": _visual(), "wrist": _visual(), "observation.state": _state()})
        assert sorted(policy._policy_image_keys()) == ["base", "wrist"]
