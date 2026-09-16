"""The state-key remedy stops naming an embodiment once one has been rejected.

``state_key_remedy`` names an embodiment only when the registry confirms its
declared ``state_keys`` bind the observation, so the advice cannot send a caller
back to the guard that printed it. Matching ``state_keys`` is necessary but not
sufficient: a declared embodiment is applied as a whole, and
``_configure_embodiment`` also validates its ``obs_rename`` against the model's
declared image features. A map naming a feature the checkpoint does not declare
is rejected, which discards the state binding along with the camera routing - so
the auto-generated ``joint_0..joint_N`` ordering survives and the caller lands
back on the all-missing guard holding the identical sentence, recommending the
embodiment they just passed.

That is the same loop the remedy exists to prevent, reached through
``obs_rename`` rather than ``state_keys``, and the policy already records it as
``_embodiment_config_failed`` - the flag the sibling missing-postprocessor
warning consults so it cannot assert a cause the discard invalidated. These
tests pin that both state-key guards consult it too.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap, state_key_remedy
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

SIM_OBS = {str(i): 0.0 for i in range(1, 7)}
GENERIC_KEYS = [f"joint_{i}" for i in range(6)]


def _feature(dim: int) -> MagicMock:
    feat = MagicMock()
    feat.shape = (dim,)
    return feat


def _rejected_policy(monkeypatch) -> LerobotLocalPolicy:
    """A policy whose declared embodiment is rejected by the real load path.

    Mirrors ``lerobot/smolvla_base`` + ``embodiment='so101'``: the embodiment's
    ``state_keys`` are exactly what the observation carries, while its
    ``obs_rename`` targets an image feature the checkpoint does not declare.
    """
    bridge = MagicMock(name="ProcessorBridge")
    bridge.is_active = True
    bridge.has_postprocessor = True
    bridge.inert_normalization_features.return_value = []
    bridge.mismatched_normalization_widths.return_value = []
    monkeypatch.setattr(
        "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
        classmethod(lambda cls, *a, **k: bridge),
    )
    embodiment = EmbodimentMap(
        name="so101_like",
        obs_rename={"front": "observation.images.image"},
        state_keys=list(SIM_OBS),
        action_keys=list(SIM_OBS),
        dim_policy="pad",
    )
    with patch.object(LerobotLocalPolicy, "_load_model"):
        pol = LerobotLocalPolicy(pretrained_name_or_path="fake/ckpt", embodiment=embodiment)
    pol._device = None
    pol._input_features = {"observation.images.camera1": _feature(3), "observation.state": _feature(6)}
    pol._output_features = {"action": _feature(6)}
    pol._load_processor_bridge()
    assert pol._embodiment_config_failed is True, "fixture must reproduce the rejection"
    # What the loader leaves behind: the declared map was discarded, so the
    # auto-generated ordering derived from the model action dim is still in force.
    pol.robot_state_keys = list(GENERIC_KEYS)
    pol.strict_keys = True
    return pol


def test_all_missing_guard_does_not_re_offer_the_rejected_embodiment(monkeypatch):
    pol = _rejected_policy(monkeypatch)
    with pytest.raises(ValueError) as excinfo:
        pol._resolve_state_order(SIM_OBS, list(SIM_OBS))
    msg = str(excinfo.value)
    assert "embodiment='" not in msg, f"re-offered an embodiment that was already rejected: {msg}"
    assert "rejected" in msg, msg
    # Both mechanisms stay named: only the value that cannot resolve is withheld.
    assert "set_robot_state_keys" in msg, msg
    assert "obs_rename_override" in msg and "camera_key_map" in msg, msg


def test_partial_missing_guard_carries_the_same_rule(monkeypatch):
    """The sibling guard reads the one helper, so its advice cannot drift."""
    pol = _rejected_policy(monkeypatch)
    pol.robot_state_keys = ["1", "gripper_absent"]
    with pytest.raises(ValueError) as excinfo:
        pol._collect_state_values(SIM_OBS, pol.robot_state_keys)
    msg = str(excinfo.value)
    assert "embodiment='" not in msg, msg
    assert "set_robot_state_keys" in msg, msg


def test_a_healthy_policy_still_names_the_matching_embodiment(monkeypatch):
    """No rejection -> the registry-derived recommendation is unchanged."""
    pol = _rejected_policy(monkeypatch)
    pol._embodiment_config_failed = False
    with pytest.raises(ValueError) as excinfo:
        pol._resolve_state_order(SIM_OBS, list(SIM_OBS))
    assert "embodiment='so101'" in str(excinfo.value)


def test_remedy_withholds_only_the_value_when_an_embodiment_was_rejected():
    offered = state_key_remedy(list(SIM_OBS))
    withheld = state_key_remedy(list(SIM_OBS), embodiment_rejected=True)
    assert "embodiment='so101'" in offered
    assert "embodiment='" not in withheld
    assert "set_robot_state_keys" in offered and "set_robot_state_keys" in withheld
    assert withheld.isascii() and withheld.endswith(".")
