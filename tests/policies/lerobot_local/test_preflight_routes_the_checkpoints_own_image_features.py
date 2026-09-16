"""The camera pre-flight check compares against the features the model declares.

``Policy.preflight`` requires a source key for every image rename TARGET an
embodiment feeds, and its refusal calls those targets "the model's image
feature(s)". That naming holds only where the feature set is BUILT from the
embodiment - the MolmoAct2 path, whose ``build_policy`` derives it from
``derive_image_keys``. A pretrained LeRobot checkpoint records its own
``input_features`` instead, and an embodiment has no say in them: ``so101``
feeds ``observation.images.image`` + ``observation.images.wrist_image`` while
``lerobot/smolvla_base`` declares ``observation.images.camera1..3``.

For such a pair the old refusal named features the model does not have, both of
its remedies were unreachable (a ``camera_key_map`` onto the named key is
refused by ``_resolve_camera_targets``; renaming cameras to the embodiment's
source keys passes pre-flight and then loses the embodiment entirely, because
``EmbodimentMap.validate`` rejects the same rename after the download and the
processor pipeline - state/action unit conversion included - is discarded).

These tests pin:

* the contradiction is refused up front, naming both sides;
* LOOP CLOSURE - the ``obs_rename_override`` the message prints is parsed back
  out of the text and re-measured through BOTH ``preflight`` and
  ``EmbodimentMap.validate``, so a remedy that would not work cannot pass. The
  drop half ALONE does not work (it leaves the declarative path with no camera
  routing), which is why the printed override carries both halves;
* two-way parity - ``preflight`` refuses exactly when ``validate`` would;
* no false positive: a checkpoint that declares the targets keeps today's
  source-availability refusal, an unreadable declared set falls back to today's
  behaviour, and MolmoAct2 is untouched.
"""

from __future__ import annotations

import ast
import re
from dataclasses import replace

import pytest
from lerobot.configs.types import FeatureType, PolicyFeature

from strands_robots.policies.lerobot_local import policy as policy_mod
from strands_robots.policies.lerobot_local.embodiment import load_embodiment
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy, _merge_obs_rename
from strands_robots.policies.lerobot_local.resolution import declared_image_features

EMBODIMENT = "so101"
CHECKPOINT = "lerobot/smolvla_base"
# The three views smolvla_base really declares, and the cameras a scene named
# for the model provides.
DECLARED = {f"observation.images.camera{i}" for i in (1, 2, 3)}
CAMERAS = ["camera1", "camera2", "camera3"]
JOINTS = [str(i) for i in range(1, 7)]
# Declaring the type short-circuits is_molmoact2 with no I/O.
BASE_CONFIG = {"pretrained_name_or_path": CHECKPOINT, "policy_type": "smolvla", "embodiment": EMBODIMENT}


@pytest.fixture
def declares(monkeypatch):
    """Pin what the checkpoint declares, so no test touches the network."""

    def _set(features):
        monkeypatch.setattr(policy_mod, "declared_image_features", lambda *_a, **_k: features)

    return _set


def _targets() -> list[str]:
    return sorted({dst for dst in load_embodiment(EMBODIMENT).obs_rename.values() if "image" in dst})


def _validate(override):
    """Run the embodiment's own post-download validation on the merged map."""
    merged = _merge_obs_rename(load_embodiment(EMBODIMENT).obs_rename, override)
    replace(load_embodiment(EMBODIMENT), obs_rename=merged).validate(
        {name: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)) for name in DECLARED},
        {},
    )


def _preflight(observation_keys=None, **extra):
    LerobotLocalPolicy.preflight(observation_keys or [*JOINTS, *CAMERAS], **{**BASE_CONFIG, **extra})


def test_a_target_the_checkpoint_does_not_declare_is_refused_naming_both_sides(declares):
    declares(DECLARED)
    with pytest.raises(ValueError) as excinfo:
        _preflight()
    message = str(excinfo.value)
    for target in _targets():
        assert target in message, f"the refusal does not name the inapplicable target {target}"
    for feature in sorted(DECLARED):
        assert feature in message, f"the refusal does not name the declared feature {feature}"
    assert CHECKPOINT in message


def test_the_override_the_message_prints_is_accepted_by_preflight_and_by_validate(declares):
    """Loop closure: follow the printed remedy verbatim, both sides of the download."""
    declares(DECLARED)
    with pytest.raises(ValueError) as excinfo:
        _preflight()
    printed = re.search(r"'obs_rename_override': (\{.*?\})", str(excinfo.value))
    assert printed, "the refusal prints no obs_rename_override literal to follow"
    override = ast.literal_eval(printed.group(1))

    _preflight(obs_rename_override=override)  # pre-download verdict
    _validate(override)  # the post-download verdict that discards the pipeline

    # ...and the condition neither of those two sees: SmolVLAPolicy.prepare_images
    # raises "All image features are missing from the batch" unless every feature
    # the checkpoint declares is routed from a key the observation provides. The
    # drop half alone satisfies preflight and validate and still fails here.
    merged = _merge_obs_rename(load_embodiment(EMBODIMENT).obs_rename, override)
    routed = {dst: src for src, dst in merged.items() if src in {*JOINTS, *CAMERAS}}
    assert set(routed) == DECLARED, f"the printed override routes {sorted(routed)}, not {sorted(DECLARED)}"


def test_dropping_the_inapplicable_renames_alone_leaves_no_camera_routing(declares):
    """Why the printed override carries both halves, not just the drops."""
    declares(DECLARED)
    drops = dict.fromkeys(load_embodiment(EMBODIMENT).obs_rename)
    _validate(drops)  # validate is happy: nothing targets an undeclared feature
    merged = _merge_obs_rename(load_embodiment(EMBODIMENT).obs_rename, drops)
    assert merged == {}, "the drops leave a rename behind"
    # ...but no declared feature has a source, which is the failure the model
    # reports as "All image features are missing from the batch".
    assert not (DECLARED & set(merged.values()))


@pytest.mark.parametrize(
    ("declared", "refused"),
    [
        (DECLARED, True),  # the measured smolvla_base pair
        (set(), True),  # a state-only checkpoint declares no image at all
        ({"observation.images.image"}, True),  # one target declared, one not
        (set(_targets()), False),  # both declared: nothing to report
        (set(_targets()) | DECLARED, False),  # a superset is still consistent
    ],
)
def test_preflight_refuses_exactly_when_the_embodiment_validation_would(declares, declared, refused):
    declares(declared)
    features = {name: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)) for name in declared}
    embodiment = load_embodiment(EMBODIMENT)
    validate_refuses = False
    try:
        embodiment.validate(features, {})
    except ValueError:
        validate_refuses = True

    preflight_refuses = False
    try:
        _preflight(observation_keys=[*JOINTS, *CAMERAS, *load_embodiment(EMBODIMENT).obs_rename])
    except ValueError:
        preflight_refuses = True

    assert preflight_refuses is refused
    assert preflight_refuses is validate_refuses, "the pre- and post-download verdicts disagree"


def test_a_checkpoint_that_declares_the_targets_still_reports_a_missing_camera(declares):
    """No false positive, and the old source-availability refusal is unchanged."""
    declares(set(_targets()))
    with pytest.raises(ValueError, match="cannot route cameras"):
        _preflight()


def test_an_unreadable_declared_set_keeps_the_source_availability_behaviour(declares):
    """``None`` means unknown, and must not be read as "declares no images"."""
    declares(None)
    with pytest.raises(ValueError, match="cannot route cameras"):
        _preflight()


def test_a_molmoact2_checkpoint_builds_its_features_and_is_untouched(declares):
    """Its feature set is derived FROM the embodiment, so the targets fit by construction."""
    declares(set())  # would refuse if the check applied
    with pytest.raises(ValueError, match="cannot route cameras"):
        LerobotLocalPolicy.preflight(
            [*JOINTS, *CAMERAS],
            pretrained_name_or_path="allenai/MolmoAct2-SO100_101",
            policy_type="molmoact2",
            embodiment=EMBODIMENT,
        )


def test_the_message_names_a_declared_feature_no_camera_supplies(declares):
    declares({"observation.images.top"})
    with pytest.raises(ValueError) as excinfo:
        _preflight(observation_keys=[*JOINTS, "cam_a"])
    assert "observation.images.top" in str(excinfo.value)
    assert "<your_camera_name>" in str(excinfo.value)


def test_declared_image_features_reads_only_the_visual_inputs(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"input_features": {"observation.images.camera1": {"type": "VISUAL"}, "observation.state": {"type": "STATE"}}}'
    )
    assert declared_image_features(str(tmp_path)) == {"observation.images.camera1"}


def test_declared_image_features_is_unknown_when_the_config_declares_none(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "molmoact2"}')
    assert declared_image_features(str(tmp_path)) is None


def test_revision_is_forwarded_to_declared_image_features(monkeypatch):
    """The revision the policy honours must reach the pre-flight feature read.

    Without it, a pinned load reads the default branch's config.json instead of
    the pinned revision's, so the pre-flight verdict can diverge from the
    post-download validate in both directions.
    """
    captured = {}

    def _spy(reference, revision=None):
        captured["reference"] = reference
        captured["revision"] = revision
        return DECLARED

    monkeypatch.setattr(policy_mod, "declared_image_features", _spy)

    pinned_config = {**BASE_CONFIG, "revision": "v1.0.0"}
    policy_mod._inapplicable_image_target_error(
        {t: [t.rsplit(".", 1)[-1]] for t in _targets()},
        EMBODIMENT,
        pinned_config,
        [*JOINTS, *CAMERAS],
    )
    assert captured["reference"] == CHECKPOINT
    assert captured["revision"] == "v1.0.0", (
        "declared_image_features must receive the revision the policy honours; got {!r}".format(captured["revision"])
    )
