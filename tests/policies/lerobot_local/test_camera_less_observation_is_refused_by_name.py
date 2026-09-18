"""A state-only observation is refused by name, not by lerobot's bare KeyError.

Both camera routers on :class:`LerobotLocalPolicy` end in the under-supplied
refusal ("Robot supplies N camera(s) ... but the policy requires image
input(s)"). The batch router only entered ``_resolve_camera_targets`` when the
observation carried at least one frame, and the preprocessor router had no
unfilled check at all - so an image policy fed a camera-less observation (a
``Robot("so100")`` with no camera, ``MUJOCO_GL=disable``) built a batch with no
image key and died inside lerobot's ``modeling_act.py`` with
``KeyError: 'observation.images.top'``. These cells pin that zero cameras reach
the same refusal one camera too few already reached, on both routers.
"""

from __future__ import annotations

import pytest

from .test_camera_target_resolution import _make_policy, _prefixed_two_cam

_STATE_ONLY = {"shoulder": 0.1, "elbow": 0.2}


def _state_only_policy():
    p = _make_policy(_prefixed_two_cam())
    p.set_robot_state_keys(list(_STATE_ONLY))
    return p


def test_the_batch_router_refuses_a_camera_less_observation():
    p = _state_only_policy()
    with pytest.raises(ValueError, match="supplies 0 camera.*requires image input"):
        p._build_batch_from_strands_format(dict(_STATE_ONLY), {})


def test_the_preprocessor_router_refuses_a_camera_less_observation():
    p = _state_only_policy()
    with pytest.raises(ValueError, match="supplies 0 camera.*requires image input"):
        p._to_lerobot_observation(dict(_STATE_ONLY))


def test_a_policy_with_no_image_inputs_still_accepts_a_state_only_observation():
    """The refusal is about declared image slots; a state-only policy has none."""
    p = _make_policy({"observation.state": object()})
    p.set_robot_state_keys(list(_STATE_ONLY))
    assert "observation.state" in p._to_lerobot_observation(dict(_STATE_ONLY))
    assert "observation.state" in p._build_batch_from_strands_format(dict(_STATE_ONLY), {})
