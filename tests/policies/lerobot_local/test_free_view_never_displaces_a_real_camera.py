"""The backend's own free view does not outrank a camera the caller added.

Both camera routers on :class:`LerobotLocalPolicy` end in a positional
fallback: cameras that matched no declared image key by name fill the declared
slots that are still free. The candidate order was the observation's order, and
``create_world`` registers the built-in free view (``"default"``) BEFORE any
``add_camera`` call - so it is first in ``get_observation()`` and it took slot 0,
a fixed three-quarter debug view of the whole scene reaching the input a
checkpoint trained on a task view. With more cameras than slots it also cost the
caller a real view: the last real camera had no free slot left and was dropped.

That is the outcome the exact-name rung of ``_to_lerobot_observation``
already names in its own comment as the reason it binds by name ("drops a real
one when an extra free camera such as the sim ``default`` is present"), reached
through the fallback rung instead whenever NO camera matches by name.

:func:`~strands_robots.utils.free_camera_routing_rank` ranks the token names last
in both routers. These cells pin the ordering, that it covers the whole
:data:`~strands_robots.utils.FREE_CAMERA_TOKENS` set rather than ``"default"``
alone, and the two things it must NOT change: the free view still fills a slot
when it is the only camera, and a name a policy declares still binds by name.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.utils import FREE_CAMERA_TOKENS, free_camera_routing_rank

from .test_camera_target_resolution import _make_policy, _prefixed_two_cam

# The token names a backend may register its free view under. ``None`` and ``""``
# cannot be observation keys, so only the spellings a camera map can carry.
FREE_NAMES = [t for t in FREE_CAMERA_TOKENS if isinstance(t, str) and t]


@pytest.mark.parametrize("free_name", FREE_NAMES)
def test_the_free_view_does_not_take_a_slot_a_real_camera_needs(free_name):
    """Two real cameras and the free view fill two slots: the real ones."""
    p = _make_policy(_prefixed_two_cam())
    result = p._resolve_camera_targets([free_name, "cam_a", "cam_b"])

    assert free_name not in result, f"the free view {free_name!r} took a declared slot"
    assert result == {
        "cam_a": "observation.images.top",
        "cam_b": "observation.images.wrist",
    }


@pytest.mark.parametrize("free_name", FREE_NAMES)
def test_the_preprocessor_router_orders_the_free_view_the_same_way(free_name):
    """The VLA/preprocessor path binds the same cameras as the batch path."""
    p = _make_policy(_prefixed_two_cam())
    frames = {
        free_name: np.full((8, 8, 3), 1, dtype=np.uint8),
        "cam_a": np.full((8, 8, 3), 2, dtype=np.uint8),
        "cam_b": np.full((8, 8, 3), 3, dtype=np.uint8),
    }
    out = p._to_lerobot_observation({**frames, "shoulder": 0.1})

    assert np.array_equal(out["observation.images.top"], frames["cam_a"])
    assert np.array_equal(out["observation.images.wrist"], frames["cam_b"])
    assert not any(np.array_equal(v, frames[free_name]) for k, v in out.items() if k.startswith("observation.images"))


def test_the_rank_orders_every_token_behind_every_real_camera():
    """The rule covers the whole token set, not the sim's spelling of it."""
    assert [free_camera_routing_rank(n) for n in FREE_CAMERA_TOKENS] == [1] * len(FREE_CAMERA_TOKENS)
    assert free_camera_routing_rank("cam_a") == 0
    assert free_camera_routing_rank(object()) == 0


def test_the_free_view_still_fills_a_slot_when_it_is_the_only_camera():
    """Ranking last is not dropping: a lone free view binds as it always did."""
    p = _make_policy({"observation.images.top": _prefixed_two_cam()["observation.images.top"]})
    assert p._resolve_camera_targets(["default"]) == {"default": "observation.images.top"}


def test_real_cameras_keep_their_observation_order_among_themselves():
    """Only the free view moves; the guess over real cameras is unchanged."""
    p = _make_policy(_prefixed_two_cam())
    assert p._resolve_camera_targets(["z_cam", "a_cam"]) == {
        "z_cam": "observation.images.top",
        "a_cam": "observation.images.wrist",
    }


def test_a_declared_free_name_still_binds_by_name():
    """The rank decides a guess only - it never overrides the exact-name rung."""
    feats = _prefixed_two_cam()
    feats["observation.images.default"] = feats.pop("observation.images.wrist")
    p = _make_policy(feats)
    assert p._resolve_camera_targets(["default", "cam_a"]) == {
        "default": "observation.images.default",
        "cam_a": "observation.images.top",
    }
