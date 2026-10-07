"""plastic_wam provider: registry entry, Policy subclass, and get_actions / learn hooks with STUB weights
(the real model is replaced by a tiny fake runtime so no torch checkpoint or HF access is needed)."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from strands_robots.policies.base import Policy


def test_registry_entry_and_class():
    reg = json.loads((Path(__file__).parents[1] / "strands_robots/registry/policies.json").read_text())
    entry = reg["providers"]["plastic_wam"]
    assert entry["class"] == "PlasticWAMPolicy" and "plastic_wam" in entry["shorthands"]
    from strands_robots.policies.plastic_wam import PlasticWAMPolicy
    assert issubclass(PlasticWAMPolicy, Policy)


class _FakeRuntime:
    """Stands in for plastic_wam.runtime.WAMPolicy: returns a 4-step chunk equal to the input state + step."""
    def __init__(self):
        self.reset_calls = 0
        self.last_obs = None

    def reset(self):
        self.reset_calls += 1

    def __call__(self, obs):
        self.last_obs = obs
        s = np.asarray(obs["state_lerobot"], np.float32)
        return [s + i for i in range(4)]


def _policy_with_stub(joint_units="deg"):
    from strands_robots.policies.plastic_wam import PlasticWAMPolicy
    p = object.__new__(PlasticWAMPolicy)      # skip __init__ (which downloads weights)
    p._w = _FakeRuntime()
    p.camera_map = ["scene", "wrist"]
    p.units = None
    if joint_units == "rad":
        from strands_robots.policies.flux3_action.units import UnitAdapter
        p.units = UnitAdapter()
    p.state_keys = [str(i) for i in range(1, 7)]
    p.learner = None
    return p


def _obs():
    o = {str(i): 0.1 * i for i in range(1, 7)}
    o["scene"] = np.zeros((224, 224, 3), np.uint8)
    o["observation.images.wrist"] = np.zeros((224, 224, 3), np.uint8)
    return o


def test_get_actions_deg_identity_chunk():
    p = _policy_with_stub("deg")
    acts = p.get_actions_sync(_obs(), "reach the red cube")
    assert len(acts) == 4 and sorted(acts[0]) == [str(i) for i in range(1, 7)]
    assert acts[2]["3"] == pytest.approx(0.3 + 2)
    assert p._w.last_obs["instruction"] == "reach the red cube" and len(p._w.last_obs["images"]) == 2


def test_get_actions_rad_roundtrip_units():
    p = _policy_with_stub("rad")
    obs = _obs()
    acts = p.get_actions_sync(obs, "x")
    # first chunk step = state + 0 → after rad→LeRobot→rad conversion it must equal the input joints
    for k in p.state_keys:
        assert acts[0][k] == pytest.approx(obs[k], abs=1e-5)


def test_reset_and_state_keys_and_missing_camera():
    p = _policy_with_stub()
    p.reset(); assert p._w.reset_calls == 1
    p.set_robot_state_keys(["a", "a.vel", "b", "c", "d", "e", "f", "g"])
    assert p.state_keys == ["a", "b", "c", "d", "e", "f"]
    with pytest.raises(KeyError):
        p._image({}, "scene")


def test_learn_requires_plastic_flag():
    p = _policy_with_stub()
    with pytest.raises(RuntimeError):
        p.learn_from_correction([[0.0] * 6] * 4)


def test_missing_package_message(monkeypatch):
    from strands_robots.policies.plastic_wam import PlasticWAMPolicy
    monkeypatch.setitem(sys.modules, "plastic_wam", None)          # simulate not installed
    with pytest.raises(ImportError, match="plastic-model"):
        PlasticWAMPolicy("x")
