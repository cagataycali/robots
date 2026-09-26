# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A checkpoint that runs on fewer views than it declares gets the views there are.

Both camera routers refused whenever ANY declared image feature went unfilled.
That is right for the checkpoints that index ``batch[key]`` for each declared
feature - the alternative is a ``KeyError`` deep inside lerobot - and wrong for
the flow-matching VLAs, which build their view list from the features PRESENT in
the batch and refuse only when none is ("All image features are missing from the
batch. At least one expected."). So a one-camera arm against
``lerobot/smolvla_base``, which declares ``observation.images.camera1..3``, was a
dead end whose stated remedy - cameras the robot does not have - could not be
followed, measured on an SO-101 scene as ``Robot supplies 1 camera(s) ['front']
but the policy requires image input(s) [...camera1, ...camera2, ...camera3]``.

Zero cameras still refuse by name on both routers (the state-only case
:mod:`tests.policies.lerobot_local.test_camera_less_observation_is_refused_by_name`
pins), and so does a type whose family was never resolved.
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.lerobot_local.resolution import (
    PARTIAL_IMAGE_POLICY_TYPES,
    accepts_partial_images,
)

from .test_camera_target_resolution import _make_policy, _prefixed_two_cam

_STATE = {"shoulder": 0.1, "elbow": 0.2}
_TOP = "observation.images.top"


def _policy(policy_type: str | None) -> Any:
    policy = _make_policy(_prefixed_two_cam())
    policy.policy_type = policy_type
    policy.set_robot_state_keys(list(_STATE))
    return policy


def _batch_router(policy: Any, cameras: tuple[str, ...]) -> set[str]:
    """Declared features that got a frame, via the no-preprocessor router."""
    return set(policy._resolve_camera_targets(list(cameras)).values())


def _preprocessor_router(policy: Any, cameras: tuple[str, ...]) -> set[str]:
    """The same, via the router used when the checkpoint ships a preprocessor."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    out = policy._to_lerobot_observation(dict(_STATE) | {cam: frame for cam in cameras})
    return {key for key in out if key.startswith("observation.images.")}


#: ``(policy type, cameras supplied, the declared features that get a frame)``.
#: ``None`` for the third element means the router refuses. Two declared slots
#: (``top``, ``wrist``) throughout, so every row differs from its neighbour in
#: one of the two things that decide the verdict: the family, or how many
#: cameras there are.
_CASES = (
    ("smolvla", ("top",), {_TOP}),
    ("pi0", ("top",), {_TOP}),
    ("smolvla", (), None),
    ("act", ("top",), None),
    (None, ("top",), None),
)


@pytest.mark.parametrize("router", [_batch_router, _preprocessor_router], ids=["batch", "preprocessor"])
@pytest.mark.parametrize(("policy_type", "cameras", "filled"), _CASES)
def test_a_partial_camera_set_is_routed_only_where_the_family_runs_on_it(
    router: Any,
    policy_type: str | None,
    cameras: tuple[str, ...],
    filled: set[str] | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy = _policy(policy_type)
    if filled is None:
        with pytest.raises(ValueError, match="requires image input"):
            router(policy, cameras)
        return
    with caplog.at_level(logging.WARNING):
        assert router(policy, cameras) == filled
    assert "observation.images.wrist" in caplog.text
    assert "prepares the views it was given" in caplog.text


def test_the_roster_names_only_types_lerobot_runs_on_a_partial_batch() -> None:
    """The roster is graded against the installed lerobot, by direction.

    Over-reporting is the direction that hurts at every version: a type named
    here that indexes every declared feature gets a ``KeyError`` from inside
    lerobot in place of the refusal that names both sides. Lagging is drift a
    written roster cannot avoid - the installed lerobot moves ahead of the floor
    the lockfile resolves - so it is reported, not failed.
    """
    lerobot = pytest.importorskip("lerobot")
    present = re.compile(r"\[\s*key\s+for\s+key\s+in\s+self\.config\.image_features\s+if\s+key\s+in\s+batch\s*\]")
    at_least_one = re.compile(r"len\(present_img_keys\)\s*==\s*0")
    over_budget = re.compile(r"len\(missing\w*\)\s*>\s*self\.config\.empty_cameras")

    derived: set[str] = set()
    families = 0
    for family in sorted(p for p in (Path(lerobot.__file__).parent / "policies").iterdir() if p.is_dir()):
        configs = sorted(family.glob("configuration_*.py"))
        models = sorted(family.glob("modeling_*.py"))
        if not configs or not models:
            continue
        registered = re.search(r'PreTrainedConfig\.register_subclass\("([^"]+)"\)', configs[0].read_text())
        if not registered:
            continue
        families += 1
        source = "\n".join(model.read_text() for model in models)
        if present.search(source) and at_least_one.search(source) and not over_budget.search(source):
            derived.add(registered.group(1))

    assert families >= 10, f"read only {families} lerobot policy families - the derivation below grades nothing"
    assert derived, "no lerobot family prepares a partial batch - the roster cannot be graded against that"
    assert not (PARTIAL_IMAGE_POLICY_TYPES - derived), (
        f"PARTIAL_IMAGE_POLICY_TYPES names {sorted(PARTIAL_IMAGE_POLICY_TYPES - derived)}, which the installed "
        f"lerobot does not prepare from the present features only; those types index every declared image "
        f"feature, so routing a partial observation to them raises a KeyError inside lerobot instead of the "
        f"refusal that names both sides. Installed lerobot accepts: {sorted(derived)}."
    )
    if lag := derived - PARTIAL_IMAGE_POLICY_TYPES:
        warnings.warn(
            f"the installed lerobot also prepares a partial batch for {sorted(lag)}; add them to "
            "PARTIAL_IMAGE_POLICY_TYPES when the dependency floor reaches this release",
            stacklevel=1,
        )


def test_an_unresolved_policy_type_is_not_assumed_to_accept_a_partial_batch() -> None:
    """``None`` is "family unknown", and the conservative half is the refusal."""
    assert accepts_partial_images(None) is False
