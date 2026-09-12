# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""An explicit ``camera_key_map`` routes cameras on the declarative path too.

``camera_key_map`` is documented as the first rung of camera routing, and the
other three routers honor it there: ``_resolve_camera_targets`` (torch-batch
path), ``_to_lerobot_observation`` (preprocessor path with no embodiment) and
``_synthesized_camera_renames`` (embodiment synthesized from
``set_robot_state_keys``) all resolve the map before any name match. The path
taken when an ``embodiment`` IS declared did not: it merged only
``obs_rename_override`` over the embodiment's declared ``obs_rename``, so a
scene whose cameras are named for the scene (``cam_top``, ``cam_wrist``) rather
than for the embodiment (``front``, ``wrist``) was

* refused by :meth:`LerobotLocalPolicy.preflight` before any download, naming
  ``obs_rename_override`` as the remedy for a caller who had already bound the
  cameras with ``camera_key_map``; and
* if that check was bypassed, left unrouted at inference: the rename map still
  named the embodiment's absent sources, so the frames were neither renamed onto
  the model's image features nor recognized as camera frames by
  ``_canonicalize_obs_images`` (they stayed HWC ``uint8``), and the model
  received no image input at all.

These tests pin the routing, the target-scoped replacement it performs, and the
bindings it must not change (no map at all, and an ``obs_rename_override`` that
still has the last word because it is the only spelling that can DROP a rename).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from strands_robots.policies.lerobot_local.embodiment import load_embodiment
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy, _route_camera_key_map

SO101_JOINTS = {"1", "2", "3", "4", "5", "6"}
SCENE_CAMS = {"cam_top", "cam_wrist"}
SCENE_MAP = {
    "cam_top": "observation.images.image",
    "cam_wrist": "observation.images.wrist_image",
}


def _visual(shape=(3, 48, 64)):
    return SimpleNamespace(type=SimpleNamespace(name="VISUAL"), shape=shape)


def _state(dim=6):
    return SimpleNamespace(type=SimpleNamespace(name="STATE"), shape=(dim,))


def _action(dim=6):
    return SimpleNamespace(type=SimpleNamespace(name="ACTION"), shape=(dim,))


def _so101_policy(camera_key_map=None, obs_rename_override=None):
    """A loaded two-camera SO-101 policy declaring ``embodiment="so101"``.

    ``so101`` renames ``front`` -> ``observation.images.image`` and ``wrist`` ->
    ``observation.images.wrist_image``; the model declares both features plus a
    6-dof state, so only the camera SOURCE keys are in question.
    """
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(
            pretrained_name_or_path=None,
            policy_type="act",
            embodiment="so101",
            camera_key_map=camera_key_map,
            obs_rename_override=obs_rename_override,
        )
    policy._input_features = {
        "observation.state": _state(6),
        "observation.images.image": _visual(),
        "observation.images.wrist_image": _visual(),
    }
    policy._output_features = {"action": _action(6)}
    policy._loaded = True
    return policy


def _real_pipeline_bridge(policy):
    """Attach a real (network-free) LeRobot rename+batch pipeline to ``policy``."""
    pytest.importorskip("lerobot.processor.pipeline")
    from lerobot.processor import AddBatchDimensionProcessorStep, RenameObservationsProcessorStep
    from lerobot.processor.pipeline import DataProcessorPipeline

    from strands_robots.policies.lerobot_local.processor import ProcessorBridge

    pipe = DataProcessorPipeline(
        steps=[RenameObservationsProcessorStep(rename_map={}), AddBatchDimensionProcessorStep()]
    )
    policy._processor_bridge = ProcessorBridge(preprocessor=pipe, device="cpu")
    return policy._processor_bridge


class TestRouteCameraKeyMapReplacesByTarget:
    """The merge is scoped by rename TARGET, so one source feeds one feature."""

    def test_an_entry_replaces_the_declared_source_for_its_feature(self):
        base = {"front": "observation.images.image", "wrist": "observation.images.wrist_image"}
        # 'front' and 'wrist' are dropped: two sources feeding one target would
        # let whichever the pipeline renamed last win.
        assert _route_camera_key_map(base, SCENE_MAP) == SCENE_MAP

    def test_a_feature_no_entry_claims_keeps_its_declared_source(self):
        base = {"front": "observation.images.image", "wrist": "observation.images.wrist_image"}
        assert _route_camera_key_map(base, {"cam_top": "observation.images.image"}) == {
            "wrist": "observation.images.wrist_image",
            "cam_top": "observation.images.image",
        }

    def test_no_map_leaves_the_declared_renames_alone(self):
        base = {"front": "observation.images.image"}
        for empty in (None, {}):
            assert _route_camera_key_map(base, empty) == base


class TestPreflightHonorsCameraKeyMap:
    """The pre-download check must not refuse a binding the caller supplied."""

    def test_a_camera_key_map_satisfies_the_embodiment_image_features(self):
        LerobotLocalPolicy.preflight(SCENE_CAMS | SO101_JOINTS, embodiment="so101", camera_key_map=SCENE_MAP)

    def test_an_unbound_camera_is_still_refused(self):
        # Over-reach control: routing rung 1 is honored, not assumed.
        with pytest.raises(ValueError) as ei:
            LerobotLocalPolicy.preflight(SCENE_CAMS | SO101_JOINTS, embodiment="so101")
        assert "front" in str(ei.value) and "wrist" in str(ei.value)

    def test_the_refusal_names_camera_key_map_as_a_remedy(self):
        with pytest.raises(ValueError) as ei:
            LerobotLocalPolicy.preflight(SCENE_CAMS | SO101_JOINTS, embodiment="so101")
        msg = str(ei.value)
        assert "camera_key_map" in msg
        assert "obs_rename_override" in msg


class TestConfiguredEmbodimentRoutesTheSceneCameras:
    """The configured map, its canonicalization set, and the model's batch."""

    def test_obs_rename_carries_the_caller_cameras(self):
        policy = _so101_policy(camera_key_map=SCENE_MAP)
        _real_pipeline_bridge(policy)
        policy._configure_embodiment()
        assert policy._embodiment.obs_rename == SCENE_MAP

    def test_the_caller_cameras_are_canonicalized_as_frames(self):
        # The rename runs INSIDE the pipeline, so canonicalization has to
        # recognize the caller's bare source keys or the frames stay HWC uint8.
        policy = _so101_policy(camera_key_map=SCENE_MAP)
        _real_pipeline_bridge(policy)
        policy._configure_embodiment()
        assert policy._embodiment_image_source_keys() == SCENE_CAMS

        obs = {"cam_top": np.full((48, 64, 3), 255, dtype=np.uint8)}
        out = policy._canonicalize_obs_images(obs, image_source_keys=policy._embodiment_image_source_keys())
        frame = out["cam_top"]
        assert frame.dtype.is_floating_point  # was uint8
        assert tuple(frame.shape) == (3, 48, 64)  # was HWC
        assert float(frame.max()) <= 1.0

    def test_a_real_pipeline_feeds_both_declared_image_features(self):
        policy = _so101_policy(camera_key_map=SCENE_MAP)
        bridge = _real_pipeline_bridge(policy)
        policy._configure_embodiment()

        obs = {name: float(i) for i, name in enumerate(load_embodiment("so101").state_keys)}
        obs["cam_top"] = np.full((48, 64, 3), 255, dtype=np.uint8)
        obs["cam_wrist"] = np.full((48, 64, 3), 128, dtype=np.uint8)
        obs = policy._canonicalize_obs_images(obs, image_source_keys=policy._embodiment_image_source_keys())
        batch = bridge.preprocess(obs, instruction="pick up the cube")

        assert "observation.images.image" in batch
        assert "observation.images.wrist_image" in batch
        assert "cam_top" not in batch and "cam_wrist" not in batch
        assert tuple(batch["observation.images.image"].shape) == (1, 3, 48, 64)
        assert tuple(batch["observation.state"].shape) == (1, 6)

    def test_an_undeclared_image_feature_is_refused_by_name(self):
        # House convention for a bad map target: the other routers raise naming
        # the feature, so validate must too rather than route it silently.
        policy = _so101_policy(camera_key_map={"cam_top": "observation.images.nope"})
        _real_pipeline_bridge(policy)
        with pytest.raises((ValueError, RuntimeError), match="observation.images.nope"):
            policy._configure_embodiment()


class TestBindingsThatMustNotChange:
    """Over-reach controls: what a caller had before still holds."""

    def test_no_map_keeps_the_embodiment_declared_sources(self):
        policy = _so101_policy()
        _real_pipeline_bridge(policy)
        policy._configure_embodiment()
        assert policy._embodiment.obs_rename == {
            "front": "observation.images.image",
            "wrist": "observation.images.wrist_image",
        }

    def test_obs_rename_override_still_has_the_last_word(self):
        # The override is applied after the map: it is the only spelling that
        # can DROP a rename, so a caller relying on it keeps their exact map.
        policy = _so101_policy(
            camera_key_map=SCENE_MAP,
            obs_rename_override={"cam_top": None, "front": "observation.images.image"},
        )
        _real_pipeline_bridge(policy)
        policy._configure_embodiment()
        assert policy._embodiment.obs_rename == {
            "front": "observation.images.image",
            "cam_wrist": "observation.images.wrist_image",
        }
