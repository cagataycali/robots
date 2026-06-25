"""Tests for the MolmoAct2 load-path orchestration in
``LerobotLocalPolicy._load_molmoact2_model``.

The ``molmoact2`` module's ``build_policy`` helper is covered separately
(``test_molmoact2.py``). What these tests pin is the *policy-class wiring*
around it: how ``LerobotLocalPolicy`` consumes the
``(policy, preprocessor, postprocessor, cfg)`` 4-tuple to populate its inference
state - device resolution, input/output feature capture, inference-kwargs
defaulting, ``actions_per_step`` auto-detection, generic state-key synthesis,
ProcessorBridge construction, and the RTC no-op. This orchestration ran only on
a 21GB hardware checkpoint before; here it is exercised dependency-free by
stubbing ``build_policy`` so a regression in the wiring is caught in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from strands_robots.policies.lerobot_local import molmoact2
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


class _FakeConfig:
    """Minimal stand-in for a loaded MolmoAct2 policy config."""

    def __init__(self, n_action_steps=30, input_features=None, output_features=None):
        self.n_action_steps = n_action_steps
        self.input_features = input_features or {}
        self.output_features = output_features or {}


class _FakeParam:
    """A stand-in parameter exposing only the ``.device`` the loader reads."""

    def __init__(self, device):
        self.device = device


class _FakePolicy:
    """Plain policy stub whose ``parameters()`` yields a CPU-device param.

    Deliberately NOT a ``torch.nn.Module`` so the test runs identically under
    the real torch and the lightweight conftest torch mock used under coverage.
    """

    def __init__(self, config):
        self.config = config

    def parameters(self):
        return iter([_FakeParam(torch.device("cpu"))])


def _feat(dim):
    f = MagicMock()
    f.shape = (dim,)
    return f


def _patch_build_policy(monkeypatch, *, policy, pre, post, cfg):
    """Stub molmoact2.build_policy to return a fixed 4-tuple without I/O."""
    captured: dict = {}

    def fake_build_policy(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return policy, pre, post, cfg

    monkeypatch.setattr(molmoact2, "build_policy", fake_build_policy)
    return captured


def _make_unloaded(**kwargs):
    """Construct the policy with _load_model patched out (no real load)."""
    with patch.object(LerobotLocalPolicy, "_load_model"):
        return LerobotLocalPolicy(pretrained_name_or_path="allenai/MolmoAct2-SO100_101", **kwargs)


def test_load_molmoact2_wires_policy_state(monkeypatch):
    """The 4-tuple from build_policy populates device, features, and loaded flag."""
    cfg = _FakeConfig(
        n_action_steps=30,
        input_features={"observation.state": _feat(6)},
        output_features={"action": _feat(6)},
    )
    policy_obj = _FakePolicy(cfg)
    _patch_build_policy(monkeypatch, policy=policy_obj, pre=None, post=None, cfg=cfg)

    policy = _make_unloaded(use_processor=False)
    policy._load_molmoact2_model()

    assert policy._loaded is True
    assert policy._policy is policy_obj
    assert policy._device == torch.device("cpu")
    assert policy.policy_type == molmoact2.MOLMOACT2_TYPE
    assert "observation.state" in policy._input_features
    assert "action" in policy._output_features
    # inference_action_mode is defaulted into inference_kwargs for select_action.
    assert "inference_action_mode" in policy.inference_kwargs


def test_load_molmoact2_auto_detects_action_horizon(monkeypatch):
    """A default actions_per_step=1 adopts the model's n_action_steps chunk."""
    cfg = _FakeConfig(n_action_steps=30, output_features={"action": _feat(6)})
    _patch_build_policy(monkeypatch, policy=_FakePolicy(cfg), pre=None, post=None, cfg=cfg)

    policy = _make_unloaded(use_processor=False)
    assert policy.actions_per_step == 1  # default before load
    policy._load_molmoact2_model()

    assert policy.actions_per_step == 30


def test_load_molmoact2_respects_explicit_action_horizon(monkeypatch):
    """An explicit actions_per_step is never overridden by auto-detection."""
    cfg = _FakeConfig(n_action_steps=30, output_features={"action": _feat(6)})
    _patch_build_policy(monkeypatch, policy=_FakePolicy(cfg), pre=None, post=None, cfg=cfg)

    policy = _make_unloaded(use_processor=False, actions_per_step=4)
    policy._load_molmoact2_model()

    assert policy.actions_per_step == 4


def test_load_molmoact2_synthesizes_state_keys_from_action_dim(monkeypatch):
    """With no robot_state_keys, generic joint_N keys are derived from action dim."""
    cfg = _FakeConfig(output_features={"action": _feat(6)})
    _patch_build_policy(monkeypatch, policy=_FakePolicy(cfg), pre=None, post=None, cfg=cfg)

    policy = _make_unloaded(use_processor=False)
    assert policy.robot_state_keys == []
    policy._load_molmoact2_model()

    assert policy.robot_state_keys == [f"joint_{i}" for i in range(6)]


def test_load_molmoact2_builds_processor_bridge_when_active(monkeypatch):
    """A non-passthrough pre/post pair yields an active ProcessorBridge."""
    cfg = _FakeConfig(
        input_features={"observation.state": _feat(6)},
        output_features={"action": _feat(6)},
    )
    # Truthy pre/post -> bridge.is_active becomes True.
    _patch_build_policy(monkeypatch, policy=_FakePolicy(cfg), pre=MagicMock(), post=MagicMock(), cfg=cfg)
    # _configure_embodiment is exercised in its own test; isolate the bridge here.
    monkeypatch.setattr(LerobotLocalPolicy, "_configure_embodiment", lambda self: None)

    policy = _make_unloaded(use_processor=True)
    policy._load_molmoact2_model()

    assert policy._processor_bridge is not None
    assert policy._processor_bridge.is_active is True


def test_load_molmoact2_no_bridge_when_passthrough(monkeypatch):
    """Passthrough (no pre/post) leaves the processor bridge unset."""
    cfg = _FakeConfig(output_features={"action": _feat(6)})
    _patch_build_policy(monkeypatch, policy=_FakePolicy(cfg), pre=None, post=None, cfg=cfg)

    policy = _make_unloaded(use_processor=True)
    policy._load_molmoact2_model()

    assert policy._processor_bridge is None


def test_load_molmoact2_rtc_is_noop(monkeypatch):
    """MolmoAct2 has no rtc_config, so RTC stays disabled after load."""
    cfg = _FakeConfig(output_features={"action": _feat(6)})
    _patch_build_policy(monkeypatch, policy=_FakePolicy(cfg), pre=None, post=None, cfg=cfg)

    policy = _make_unloaded(use_processor=False)
    policy._load_molmoact2_model()

    assert policy._rtc_enabled is False


def test_load_molmoact2_state_dim_from_preset_keys(monkeypatch):
    """Pre-set robot_state_keys drive state_dim/action_dim passed to build_policy."""
    cfg = _FakeConfig(output_features={"action": _feat(7)})
    captured = _patch_build_policy(monkeypatch, policy=_FakePolicy(cfg), pre=None, post=None, cfg=cfg)

    policy = _make_unloaded(use_processor=False)
    policy.set_robot_state_keys(["j0", "j1", "j2", "j3", "j4", "j5", "j6"])
    policy._load_molmoact2_model()

    assert captured["kwargs"]["state_dim"] == 7
    assert captured["kwargs"]["action_dim"] == 7
    # Pre-set keys are preserved (not overwritten by joint_N synthesis).
    assert policy.robot_state_keys == ["j0", "j1", "j2", "j3", "j4", "j5", "j6"]
