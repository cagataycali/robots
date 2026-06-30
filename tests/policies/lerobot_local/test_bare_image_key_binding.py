"""Exact-name camera binding for policies that declare BARE image keys.

Some VLAs (e.g. MolmoAct2) declare their visual ``input_features`` with bare
camera names (``base`` / ``wrist``) rather than the ``observation.images.<cam>``
form used by ACT/diffusion checkpoints. The strands-native observation path
``_to_lerobot_observation`` used to test only the prefixed form
(``observation.images.<cam>``) against ``_input_features``, so a camera whose
name matched a bare declared key NEVER bound by name -- it fell through to
positional fill. That:

  * emitted a misleading "does not match any declared policy image key" warning
    on an exact-name hit,
  * shuffled views when the observation key order differed from the declared
    order, and
  * dropped a real view when an extra free camera (such as the sim ``default``
    free camera) occupied a positional slot.

These pin the fix: a bare-key camera binds by name (mirroring the precedence in
``_resolve_camera_targets``), with no positional fallback and no extra view
displacing a named one.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


class _VisualFeature:
    """Minimal stand-in for a declared VISUAL ``PolicyFeature``."""

    class _T:
        name = "VISUAL"

    type = _T()


def _molmoact2_style_policy() -> LerobotLocalPolicy:
    """A policy declaring BARE image keys (``base`` / ``wrist``), MolmoAct2-style."""
    with patch.object(LerobotLocalPolicy, "_load_model"):
        p = LerobotLocalPolicy(pretrained_name_or_path="test/model")
    p._input_features = {
        "base": _VisualFeature(),
        "wrist": _VisualFeature(),
        "observation.state": object(),
    }
    p.robot_state_keys = ["1", "2", "3", "4", "5", "6"]
    return p


def _state_scalars() -> dict[str, float]:
    return {"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4, "5": 0.5, "6": 0.6}


def test_bare_image_keys_bind_by_name_no_positional_fallback():
    """Cameras whose names equal bare declared keys bind by name, not positionally."""
    p = _molmoact2_style_policy()
    base = np.ones((8, 8, 3), dtype=np.uint8) * 1
    wrist = np.ones((8, 8, 3), dtype=np.uint8) * 2
    out = p._to_lerobot_observation({"base": base, "wrist": wrist, **_state_scalars()})

    assert p.positional_fallback_used is False
    # Each frame reached its OWN declared slot (not swapped).
    assert float(np.asarray(out["base"]).mean()) == 1.0
    assert float(np.asarray(out["wrist"]).mean()) == 2.0


def test_extra_free_camera_does_not_displace_a_named_view():
    """A leaked free camera (sim ``default``) is dropped, not fed as a named view.

    With the observation iterating ``default`` first, the pre-fix positional
    fill bound ``default`` -> ``base`` and ``base`` -> ``wrist`` while DROPPING
    the real ``wrist`` frame. After the fix, ``base``/``wrist`` bind by name and
    the unmatched ``default`` is discarded.
    """
    p = _molmoact2_style_policy()
    default = np.ones((8, 8, 3), dtype=np.uint8) * 9
    base = np.ones((8, 8, 3), dtype=np.uint8) * 1
    wrist = np.ones((8, 8, 3), dtype=np.uint8) * 2
    out = p._to_lerobot_observation({"default": default, "base": base, "wrist": wrist, **_state_scalars()})

    assert p.positional_fallback_used is False
    assert float(np.asarray(out["base"]).mean()) == 1.0  # not 9.0 (the default cam)
    assert float(np.asarray(out["wrist"]).mean()) == 2.0  # not dropped
    assert "default" not in out  # the extra free camera is discarded


def test_prefixed_image_keys_still_bind_by_name():
    """ACT/diffusion-style ``observation.images.<cam>`` features keep binding by name."""
    with patch.object(LerobotLocalPolicy, "_load_model"):
        p = LerobotLocalPolicy(pretrained_name_or_path="test/model")
    p._input_features = {
        "observation.images.top": _VisualFeature(),
        "observation.images.wrist": _VisualFeature(),
        "observation.state": object(),
    }
    p.robot_state_keys = ["1", "2", "3", "4", "5", "6"]
    top = np.ones((8, 8, 3), dtype=np.uint8) * 1
    wrist = np.ones((8, 8, 3), dtype=np.uint8) * 2
    out = p._to_lerobot_observation({"top": top, "wrist": wrist, **_state_scalars()})

    assert p.positional_fallback_used is False
    assert float(np.asarray(out["observation.images.top"]).mean()) == 1.0
    assert float(np.asarray(out["observation.images.wrist"]).mean()) == 2.0
