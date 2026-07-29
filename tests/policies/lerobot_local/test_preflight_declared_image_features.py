"""An explicit ``image_keys`` must declare the features the embodiment feeds.

``Policy.preflight`` exists for one purpose, in its own words: catch a camera
routing mismatch before "the multi-minute weight download". It checked one
direction only - that each image rename TARGET has a source key present in the
observation - and never checked that those targets are features the model will
actually declare.

An explicit ``image_keys=`` is priority 1 in
:func:`~strands_robots.policies.lerobot_local.molmoact2.derive_image_keys`, so it
replaces the feature list otherwise derived from the embodiment's ``obs_rename``
targets. A list disjoint from those targets therefore builds a model without the
inputs the embodiment routes, and ``EmbodimentMap.validate`` refuses it - but
only after the download, on the MolmoAct2 path unguarded by any ``try`` (see
``_load_molmoact2_model``), so the load aborts. Both sides of that verdict are in
``policy_config`` before anything is fetched.

These tests pin:

* the withholding configuration is refused up front, naming both sides;
* every remedy the message names is verified by configuring it and re-running
  BOTH ``preflight`` and ``EmbodimentMap.validate`` (a recommendation that would
  not work cannot pass);
* two-way parity - ``preflight`` refuses a list exactly when ``validate`` would;
* no false positive where ``image_keys`` is inert (non-MolmoAct2 checkpoints) or
  absent (the targets are then derived from the embodiment and cannot diverge);
* the declared-feature contradiction is reported before the camera-source check,
  since no camera rename can resolve it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from lerobot.configs.types import FeatureType, PolicyFeature

from strands_robots.policies.lerobot_local import molmoact2 as molmoact2_mod
from strands_robots.policies.lerobot_local.embodiment import load_embodiment
from strands_robots.policies.lerobot_local.molmoact2 import derive_image_keys
from strands_robots.policies.lerobot_local.policy import (
    LerobotLocalPolicy,
    _merge_obs_rename,
)

EMBODIMENT = "so_real"
# Declaring the type short-circuits is_molmoact2 with no I/O, so these tests
# never touch the network.
MOLMOACT2 = {
    "policy_type": "molmoact2",
    "pretrained_name_or_path": "allenai/MolmoAct2-SO100_101",
}


def _embodiment_targets() -> list[str]:
    """The image rename targets the embodiment feeds, in sorted order."""
    emb = load_embodiment(EMBODIMENT)
    return sorted({dst for dst in emb.obs_rename.values() if "image" in dst})


def _observation_keys() -> set[str]:
    """A runtime observation carrying the embodiment's joints and source cameras."""
    emb = load_embodiment(EMBODIMENT)
    sources = {src for src, dst in emb.obs_rename.items() if "image" in dst}
    return set(emb.state_keys) | sources


def _validate_after_download(
    image_keys: list[str] | None,
    obs_rename_override: dict[str, str | None] | None = None,
) -> str | None:
    """Return ``validate``'s post-download refusal for a config, or ``None``.

    Reproduces the declared-feature construction ``molmoact2.build_policy``
    performs after the weight download, so the comparison is against what the
    model really declares rather than an invented feature set.
    """
    emb = load_embodiment(EMBODIMENT)
    if obs_rename_override:
        emb = replace(emb, obs_rename=_merge_obs_rename(emb.obs_rename, obs_rename_override))
    declared = derive_image_keys(image_keys, EMBODIMENT)
    input_features = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for k in declared}
    input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    try:
        emb.validate(input_features, output_features)
    except ValueError as exc:
        return str(exc)
    return None


def _preflight(image_keys: list[str] | None = None, **extra: object) -> str | None:
    """Return ``preflight``'s refusal for a config, or ``None`` when accepted."""
    config: dict[str, object] = {"embodiment": EMBODIMENT, **MOLMOACT2, **extra}
    if image_keys is not None:
        config["image_keys"] = image_keys
    try:
        LerobotLocalPolicy.preflight(_observation_keys(), **config)
    except ValueError as exc:
        return str(exc)
    return None


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------
def test_image_keys_withholding_an_embodiment_feature_is_refused():
    """A list disjoint from the rename targets is refused before the download."""
    message = _preflight(["base", "wrist"])
    assert message is not None, "preflight accepted a list that validate() refuses"
    for target in _embodiment_targets():
        assert target in message, message


def test_refusal_names_both_the_fed_features_and_the_declared_list():
    """The message names what the embodiment feeds AND what the caller declared."""
    message = _preflight(["base", "wrist"])
    assert message is not None
    assert "'base'" in message and "'wrist'" in message, message
    assert "image_keys" in message, message


def test_a_prefixed_but_unrelated_key_is_also_refused():
    """Following the naming convention does not make a wrong feature satisfiable."""
    assert _preflight(["observation.images.top"]) is not None


# ---------------------------------------------------------------------------
# Every remedy the message names must work
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("label", "image_keys", "override"),
    [
        ("(a) drop image_keys", None, None),
        ("(b) declare what the embodiment feeds", "TARGETS", None),
        ("(c) retarget every declared key", ["base", "wrist"], {"front": "base", "wrist": "wrist"}),
    ],
)
def test_every_remedy_the_message_names_is_accepted_end_to_end(label, image_keys, override):
    """Configuring any offered remedy passes preflight AND the post-download check."""
    keys = _embodiment_targets() if image_keys == "TARGETS" else image_keys
    extra = {"obs_rename_override": override} if override else {}
    assert _preflight(keys, **extra) is None, f"{label} was refused by preflight"
    assert _validate_after_download(keys, override) is None, f"{label} fails after the download"


def test_a_partial_retarget_is_still_refused():
    """Remedy (c) requires every declared key; one rename does not suffice."""
    assert _preflight(["base", "wrist"], obs_rename_override={"front": "base"}) is not None


# ---------------------------------------------------------------------------
# Two-way parity with the post-download check
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "image_keys",
    [
        None,
        ["base", "wrist"],
        ["observation.images.top"],
        ["observation.images.image"],
        ["observation.images.image", "observation.images.wrist_image"],
        ["observation.images.image", "observation.images.wrist_image", "observation.images.spare"],
    ],
)
def test_preflight_refuses_exactly_what_validate_refuses(image_keys):
    """The pre-download verdict matches the post-download one for every list."""
    assert (_preflight(image_keys) is None) == (_validate_after_download(image_keys) is None), (
        f"verdicts differ for image_keys={image_keys!r}"
    )


# ---------------------------------------------------------------------------
# No false positives
# ---------------------------------------------------------------------------
def test_omitted_image_keys_is_unaffected():
    """Without an explicit list the features are derived from the embodiment."""
    assert _preflight(None) is None


def test_an_empty_list_falls_back_to_the_embodiment():
    """An empty list is not an override; derive_image_keys ignores it."""
    assert _preflight([]) is None


def test_a_non_molmoact2_checkpoint_is_unaffected():
    """``image_keys`` is inert off the MolmoAct2 path, so it is not refused there."""
    config = {
        "embodiment": EMBODIMENT,
        "image_keys": ["base", "wrist"],
        "policy_type": "act",
        "pretrained_name_or_path": "lerobot/act_so101",
    }
    LerobotLocalPolicy.preflight(_observation_keys(), **config)


def test_autodetected_molmoact2_checkpoint_is_checked(monkeypatch):
    """Auto-detect is the documented MolmoAct2 path, so it is covered too."""
    monkeypatch.setattr(
        molmoact2_mod,
        "_read_config_json",
        lambda _path: {"model_type": "molmoact2"},
    )
    config = {
        "embodiment": EMBODIMENT,
        "image_keys": ["base", "wrist"],
        "pretrained_name_or_path": "some-org/molmoact2-so101",
    }
    with pytest.raises(ValueError):
        LerobotLocalPolicy.preflight(_observation_keys(), **config)


def test_no_embodiment_is_still_a_noop():
    """The hook cannot reason about heuristic routing, so it stays a no-op."""
    LerobotLocalPolicy.preflight({"front", "wrist"}, image_keys=["base"], **MOLMOACT2)


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------
def test_declared_feature_contradiction_is_reported_before_a_camera_mismatch():
    """No camera rename can fix a withheld feature, so it is reported first."""
    config = {"embodiment": EMBODIMENT, "image_keys": ["base", "wrist"], **MOLMOACT2}
    with pytest.raises(ValueError) as ei:
        # Camera names are wrong too; the config contradiction still wins.
        LerobotLocalPolicy.preflight({"realsense_top", "realsense_side"}, **config)
    assert "does not declare them" in str(ei.value), str(ei.value)
