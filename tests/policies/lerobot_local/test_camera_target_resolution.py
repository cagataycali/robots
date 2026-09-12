"""Direct-unit coverage of ``LerobotLocalPolicy._resolve_camera_targets``.

The camera-name -> policy-image-key router encodes a documented precedence:

  1. an explicit ``camera_key_map`` ctor param wins for any name it lists,
  2. an exact name match (``top`` -> ``observation.images.top`` OR a bare
     ``top`` that the policy declares directly),
  3. positional fallback into the remaining declared slots (loud WARN), and
  4. a hard ``ValueError`` when the robot under-supplies cameras.

The method is exercised indirectly through the torch-batch build path, but its
bare-declared-key branch (a policy that declares ``base`` / ``wrist`` instead of
``observation.images.<cam>``, e.g. MolmoAct2) had no direct test. These pin the
full contract so a routing regression is caught at the method boundary rather
than only through a higher-level rollout.

That precedence is not one method's, it is the router's, and there are two of
them: ``_resolve_camera_targets`` builds the batch when the checkpoint ships no
preprocessor, and ``_to_lerobot_observation`` remaps the observation when it
does. Whether a checkpoint ships a preprocessor is not something a camera
binding may depend on, so the closing cells drive both on the same input and
require the same answer.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pytest

from strands_robots.policies.lerobot_local.policy import (
    LerobotLocalPolicy,
    _declared_feature_is_image,
)


class _VisualFeature:
    """Minimal stand-in for a declared VISUAL ``PolicyFeature``."""

    class _T:
        name = "VISUAL"

    type = _T()


def _make_policy(
    features: dict[str, object],
    *,
    camera_key_map: dict[str, str] | None = None,
    strict_keys: bool = False,
) -> LerobotLocalPolicy:
    """Build a policy with ``_load_model`` patched and declared input features."""
    with patch.object(LerobotLocalPolicy, "_load_model"):
        p = LerobotLocalPolicy(
            pretrained_name_or_path="test/model",
            camera_key_map=camera_key_map,
            strict_keys=strict_keys,
        )
    p._input_features = features
    return p


def _prefixed_two_cam() -> dict[str, object]:
    return {
        "observation.images.top": _VisualFeature(),
        "observation.images.wrist": _VisualFeature(),
        "observation.state": object(),
    }


def _bare_two_cam() -> dict[str, object]:
    return {
        "base": _VisualFeature(),
        "wrist": _VisualFeature(),
        "observation.state": object(),
    }


def test_bare_declared_key_binds_by_exact_name():
    """A camera named exactly like a BARE declared key binds by name, not positionally."""
    p = _make_policy(_bare_two_cam())
    result = p._resolve_camera_targets(["base", "wrist"])
    assert result == {"base": "base", "wrist": "wrist"}
    assert p.positional_fallback_used is False


def test_prefixed_declared_key_binds_by_short_name():
    """``top`` binds to the declared ``observation.images.top`` slot."""
    p = _make_policy(_prefixed_two_cam())
    result = p._resolve_camera_targets(["top", "wrist"])
    assert result == {
        "top": "observation.images.top",
        "wrist": "observation.images.wrist",
    }
    assert p.positional_fallback_used is False


def test_explicit_camera_key_map_wins():
    """An explicit mapping routes a mismatched camera name onto a declared key."""
    p = _make_policy(
        _prefixed_two_cam(),
        camera_key_map={"cam_left": "observation.images.top"},
    )
    result = p._resolve_camera_targets(["cam_left", "wrist"])
    assert result["cam_left"] == "observation.images.top"
    assert result["wrist"] == "observation.images.wrist"
    assert p.positional_fallback_used is False


def test_explicit_map_to_undeclared_key_raises():
    """A camera_key_map entry targeting an undeclared image key is rejected."""
    p = _make_policy(
        _prefixed_two_cam(),
        camera_key_map={"top": "observation.images.nonexistent"},
    )
    with pytest.raises(ValueError, match="does not declare"):
        p._resolve_camera_targets(["top"])


def test_positional_fallback_when_names_do_not_match(caplog):
    """Unmatched names fill remaining declared slots positionally with a WARN."""
    p = _make_policy(_prefixed_two_cam())
    with caplog.at_level(logging.WARNING):
        result = p._resolve_camera_targets(["cam_a", "cam_b"])
    assert set(result.values()) == {
        "observation.images.top",
        "observation.images.wrist",
    }
    assert p.positional_fallback_used is True
    assert "does not match any declared policy image key" in caplog.text


def test_strict_keys_raises_instead_of_positional_fallback():
    """strict_keys=True turns an unmatched-name fallback into an actionable error."""
    p = _make_policy(_prefixed_two_cam(), strict_keys=True)
    with pytest.raises(ValueError, match="strict_keys=True"):
        p._resolve_camera_targets(["cam_a", "cam_b"])
    assert p.positional_fallback_used is False


def test_extra_cameras_beyond_policy_slots_are_dropped():
    """Cameras past what the policy consumes are omitted, not force-fed."""
    p = _make_policy(_bare_two_cam())
    result = p._resolve_camera_targets(["base", "wrist", "extra"])
    assert result == {"base": "base", "wrist": "wrist"}
    assert "extra" not in result


def test_under_supplied_cameras_raises():
    """Fewer cameras than the policy's declared image slots is a hard error."""
    p = _make_policy(_prefixed_two_cam())
    with pytest.raises(ValueError, match="requires image input"):
        p._resolve_camera_targets(["top"])


# ---------------------------------------------------------------------------
# The precedence is the router's, not one method's: both must agree
# ---------------------------------------------------------------------------
def _slots_from_remap(
    features: dict[str, object],
    cam_names: list[str],
    camera_key_map: dict[str, str] | None,
) -> dict[str, str]:
    """Which source camera reaches which declared slot, via the remap router.

    Each camera is given a frame filled with its own index, so the source is
    recoverable from the routed array without depending on object identity.
    """
    p = _make_policy(features, camera_key_map=camera_key_map)
    p.set_robot_state_keys(["j1"])
    obs: dict[str, object] = {"j1": 0.0}
    for i, name in enumerate(cam_names):
        obs[name] = np.full((4, 4, 3), i + 1, dtype=np.uint8)
    routed = p._to_lerobot_observation(obs)
    declared = {f for f, feat in features.items() if _declared_feature_is_image(f, feat)}
    return {
        key: cam_names[int(value.flat[0]) - 1]
        for key, value in routed.items()
        if key in declared and isinstance(value, np.ndarray)
    }


def _slots_from_targets(
    features: dict[str, object],
    cam_names: list[str],
    camera_key_map: dict[str, str] | None,
) -> dict[str, str]:
    """The same slot -> source answer, via the batch-path router."""
    p = _make_policy(features, camera_key_map=camera_key_map)
    return {feat: cam for cam, feat in p._resolve_camera_targets(list(cam_names)).items()}


# An extra unconsumed camera is included because the sim always supplies one
# (MuJoCo's ``default`` free camera), and it is first in observation order.
_ROUTING_CASES = [
    ({"cam_left": "observation.images.top"}, "a name that matches nothing"),
    ({"wrist": "observation.images.top"}, "a name that also matches a declared slot"),
    ({"top": "observation.images.wrist", "wrist": "observation.images.top"}, "both, crossed"),
    ({"top": "observation.images.top"}, "a redundant map naming its own slot"),
    (None, "no map at all"),
]


@pytest.mark.parametrize(("camera_key_map", "shape"), _ROUTING_CASES, ids=[c[1] for c in _ROUTING_CASES])
def test_both_routers_bind_the_same_cameras(camera_key_map, shape: str) -> None:
    """Whether the checkpoint ships a preprocessor cannot change a binding."""
    features, cams = _prefixed_two_cam(), ["default", "top", "wrist"]
    remap = _slots_from_remap(features, cams, camera_key_map)
    targets = _slots_from_targets(features, cams, camera_key_map)
    assert remap == targets, f"the two routers disagree for {shape}: {remap} vs {targets}"


def test_an_explicit_map_outranks_another_cameras_name_match() -> None:
    """The mapped camera reaches its slot even when another name matches it.

    Precedence rung 1 is "an explicit ``camera_key_map`` wins for any name it
    lists". Resolved per-camera in observation order instead, the earlier
    ``top`` claims ``observation.images.top`` by name first and the caller's
    ``wrist`` binding is then dropped for having no free slot left - a silent
    substitution, since nothing reports a map entry that was discarded.

    What fills the slot ``top`` vacates is a separate question and is not
    graded here: with its own slot taken by the map, ``top`` has none, so the
    remaining cameras reach the last slot through the documented positional
    fallback, which warns.
    """
    slots = _slots_from_remap(
        _prefixed_two_cam(),
        ["default", "top", "wrist"],
        {"wrist": "observation.images.top"},
    )
    assert slots["observation.images.top"] == "wrist"


def test_a_mapped_target_the_policy_does_not_declare_still_raises() -> None:
    """The first pass keeps the map's own validation, it does not skip it."""
    p = _make_policy(_prefixed_two_cam(), camera_key_map={"top": "observation.images.nope"})
    p.set_robot_state_keys(["j1"])
    with pytest.raises(ValueError, match="does not declare"):
        p._to_lerobot_observation({"j1": 0.0, "top": np.zeros((4, 4, 3), dtype=np.uint8)})
