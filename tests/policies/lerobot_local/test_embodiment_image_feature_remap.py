"""Unambiguous image-feature remap keeps normalization alive.

Every SO-arm embodiment hardcodes ``obs_rename`` onto
``observation.images.image`` (plus ``wrist_image`` for the dual-cam ones), but a
real checkpoint may declare a differently named single image feature - an ACT
SO-101 checkpoint trained with ``observation.images.laptop``, for instance. That
name mismatch used to discard the entire ACTIVE processor pipeline, which drops
input normalization AND action unnormalization: the policy then emitted raw
model-space values (order 0.5) that ``SOFollower.send_action`` accepted as
degrees, driving every joint to ~0 while the call reported success.

When the model declares exactly ONE image feature the intended routing is not in
doubt, so the embodiment's primary camera is rerouted onto it and the extra
renames are dropped, preserving normalization. Anything ambiguous (two or more
declared image features) still fails loudly and asks the caller to be explicit.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


def _image_feature() -> MagicMock:
    feat = MagicMock()
    feat.shape = (3, 480, 640)
    # lerobot's PolicyFeature.type is a FeatureType ENUM, so the authoritative
    # classifier reads ``feature.type.name``. A plain string here would leave
    # ``.name`` a MagicMock and silently fall back to the name convention.
    feat.type = SimpleNamespace(name="VISUAL")
    return feat


def _vector_feature(dim: int) -> MagicMock:
    feat = MagicMock()
    feat.shape = (dim,)
    return feat


def _dual_cam_embodiment() -> EmbodimentMap:
    """A so_real-shaped embodiment: two cameras, both onto features named 'image'."""
    return EmbodimentMap(
        name="so_real_like",
        obs_rename={
            "front": "observation.images.image",
            "wrist": "observation.images.wrist_image",
        },
        state_keys=[f"j{i}.pos" for i in range(6)],
        action_keys=[f"j{i}.pos" for i in range(6)],
        dim_policy="pad",
    )


def _make_policy(declared_images: list[str], **kwargs) -> LerobotLocalPolicy:
    with patch.object(LerobotLocalPolicy, "_load_model"):
        pol = LerobotLocalPolicy(pretrained_name_or_path="fake/ckpt", **kwargs)
    pol._device = None
    pol._input_features = {name: _image_feature() for name in declared_images}
    pol._input_features["observation.state"] = _vector_feature(6)
    pol._output_features = {"action": _vector_feature(6)}
    return pol


def _fake_bridge() -> MagicMock:
    bridge = MagicMock(name="ProcessorBridge")
    bridge.is_active = True
    bridge.has_postprocessor = True
    bridge.inert_normalization_features.return_value = []
    return bridge


def _patch_from_pretrained(monkeypatch, bridge) -> None:
    monkeypatch.setattr(
        "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
        classmethod(lambda cls, *a, **k: bridge),
    )


class TestSingleDeclaredImageIsRepaired:
    def test_pipeline_survives_a_name_only_mismatch(self, monkeypatch):
        """The regression: one declared image feature under a different name."""
        bridge = _fake_bridge()
        _patch_from_pretrained(monkeypatch, bridge)
        pol = _make_policy(["observation.images.laptop"], embodiment=_dual_cam_embodiment())

        pol._load_processor_bridge()

        # Normalization is preserved: the bridge is still live, not discarded.
        assert pol._processor_bridge is bridge
        assert pol._embodiment_config_failed is False
        # The primary camera was routed onto the model's declared feature and the
        # camera the model has no feature for was dropped.
        assert pol._embodiment.obs_rename == {"front": "observation.images.laptop"}

    def test_single_declared_image_matching_first_rename_target_is_repaired(self, monkeypatch):
        """The DOMINANT real SO-101 shape, and the one the repair used to decline.

        Every SO embodiment renames BOTH ``front -> observation.images.image``
        AND ``wrist -> observation.images.wrist_image``. Against a single-camera
        checkpoint declaring exactly ``observation.images.image`` the FRONT rename
        is already correct and the WRIST rename is the mismatch, so the trigger
        has to be "is any rename target undeclared?" rather than "is the declared
        target already routed?" - both are true here at once. Asking the wrong
        question declined the repair, and with allow_unnormalized defaulting to
        False the load then RAISED, so the feature could not work for its most
        common intended input (embodiment='so_real' on a real arm).
        """
        bridge = _fake_bridge()
        _patch_from_pretrained(monkeypatch, bridge)
        pol = _make_policy(["observation.images.image"], embodiment=_dual_cam_embodiment())

        pol._load_processor_bridge()

        # The policy CONSTRUCTS (pre-fix this raised RuntimeError) ...
        assert pol._processor_bridge is bridge
        assert pol._embodiment_config_failed is False
        # ... the already-correct front rename is kept, not rewritten, and only
        # the rename whose target the model does not declare is dropped.
        assert pol._embodiment.obs_rename == {"front": "observation.images.image"}

    def test_repair_keeps_the_camera_that_already_targets_the_declared_feature(self, monkeypatch):
        """Rename order must not decide which camera survives; the routing does.

        With the sources reversed, the primary must still be the one already
        pointing at the declared feature, not merely the first one listed.
        """
        from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap

        reversed_map = EmbodimentMap(
            name="wrist_first",
            obs_rename={
                "wrist": "observation.images.wrist_image",
                "front": "observation.images.image",
            },
            state_keys=[f"j{i}.pos" for i in range(6)],
            action_keys=[f"j{i}.pos" for i in range(6)],
            dim_policy="pad",
        )
        bridge = _fake_bridge()
        _patch_from_pretrained(monkeypatch, bridge)
        pol = _make_policy(["observation.images.image"], embodiment=reversed_map)

        pol._load_processor_bridge()

        assert pol._processor_bridge is bridge
        assert pol._embodiment.obs_rename == {"front": "observation.images.image"}

    def test_bare_visual_target_is_classified_as_image(self, monkeypatch):
        """A rename target that is a BARE declared image key must count as one.

        MolmoAct2 declares image features named ``base`` / ``wrist`` - no
        ``"image"`` substring - which is the very case
        ``_declared_feature_is_image`` exists for. The repair classified rename
        TARGETS with a bare ``"image" in dst`` substring instead, so such an
        embodiment yielded no image sources, the repair declined, and the load
        raised even though the routing was unambiguous.
        """
        from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap

        bare_target_map = EmbodimentMap(
            name="molmo_style",
            obs_rename={"front": "base", "wrist": "wrist"},
            state_keys=[f"j{i}.pos" for i in range(6)],
            action_keys=[f"j{i}.pos" for i in range(6)],
            dim_policy="pad",
        )
        bridge = _fake_bridge()
        _patch_from_pretrained(monkeypatch, bridge)
        pol = _make_policy(["primary"], embodiment=bare_target_map)

        pol._load_processor_bridge()

        assert pol._processor_bridge is bridge
        assert pol._embodiment.obs_rename == {"front": "primary"}

    def test_the_target_classifier_prefers_declared_feature_type(self):
        """A bare key the model DECLARES as VISUAL is an image; an unknown is not."""
        from strands_robots.policies.lerobot_local.policy import _rename_target_is_image

        features = {"primary": _image_feature(), "observation.state": _vector_feature(6)}

        assert _rename_target_is_image("primary", features) is True
        assert _rename_target_is_image("observation.state", features) is False
        # No features available (the classmethod preflight path) -> name convention.
        assert _rename_target_is_image("observation.images.top") is True
        assert _rename_target_is_image("primary") is False

    def test_explicit_override_is_not_overridden_by_the_repair(self, monkeypatch):
        """A caller who routes wrist -> the declared feature keeps that routing."""
        bridge = _fake_bridge()
        _patch_from_pretrained(monkeypatch, bridge)
        pol = _make_policy(
            ["observation.images.laptop"],
            embodiment=_dual_cam_embodiment(),
            obs_rename_override={"wrist": "observation.images.laptop", "front": None},
        )

        pol._load_processor_bridge()

        assert pol._processor_bridge is bridge
        assert pol._embodiment.obs_rename == {"wrist": "observation.images.laptop"}

    def test_matching_names_need_no_repair(self, monkeypatch):
        """When the embodiment already names the declared features, nothing changes."""
        bridge = _fake_bridge()
        _patch_from_pretrained(monkeypatch, bridge)
        pol = _make_policy(
            ["observation.images.image", "observation.images.wrist_image"],
            embodiment=_dual_cam_embodiment(),
        )

        pol._load_processor_bridge()

        assert pol._processor_bridge is bridge
        assert pol._embodiment.obs_rename == {
            "front": "observation.images.image",
            "wrist": "observation.images.wrist_image",
        }


class TestAmbiguousCasesStillFailLoudly:
    def test_two_declared_images_with_no_matching_name_raises(self, monkeypatch):
        """Ambiguous routing must not be guessed: the caller has to say which is which."""
        bridge = _fake_bridge()
        _patch_from_pretrained(monkeypatch, bridge)
        pol = _make_policy(
            ["observation.images.cam_high", "observation.images.cam_low"],
            embodiment=_dual_cam_embodiment(),
        )

        with pytest.raises(RuntimeError, match="embodiment could not be configured"):
            pol._load_processor_bridge()

    def test_a_structural_mismatch_is_not_repaired_by_renaming(self, monkeypatch):
        """A wrong action dim is not an image-naming problem; it must still raise."""
        bridge = _fake_bridge()
        _patch_from_pretrained(monkeypatch, bridge)
        wrong_dims = EmbodimentMap(
            name="wrong_dims",
            obs_rename={"front": "observation.images.image"},
            state_keys=[],
            action_keys=["a", "b", "c"],
            dim_policy="pad",
        )
        pol = _make_policy(["observation.images.laptop"], embodiment=wrong_dims)

        with pytest.raises(RuntimeError, match="embodiment could not be configured"):
            pol._load_processor_bridge()
