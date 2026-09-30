"""Two checkpoint config fields no longer decide how a policy runs at inference.

* ``device``: where the checkpoint was trained or saved. ``lerobot/pi05_droid``
  ships ``"device": "cpu"`` and lerobot_local honoured it, so a 4B pi0.5 ran at
  6.6-10.4 s per chunk on an idle L40S (0.76 s on the GPU).
* ``compile_model``: the LIBERO pi0 / pi0.5 / pi0-FAST fine-tunes ship
  ``compile_model: true, compile_mode: max-autotune``; the first inference then
  spent 8+ minutes in inductor autotuning inside the control loop, logging
  nothing, and the rollout looked hung.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from strands_robots.policies.lerobot_local import policy as policy_mod
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


class _Config:
    def __init__(self, device: str = "cpu", compile_model: bool = False) -> None:
        self.device, self.compile_model, self.compile_mode = device, compile_model, "max-autotune"
        self.input_features: dict[str, Any] = {}
        self.output_features: dict[str, Any] = {}


@pytest.fixture
def lerobot_config(monkeypatch: pytest.MonkeyPatch):
    """A stand-in ``lerobot.configs.policies`` whose ``PreTrainedConfig`` returns the config under test."""
    holder: dict[str, Any] = {"config": _Config()}
    module = types.ModuleType("lerobot.configs.policies")

    class PreTrainedConfig:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> _Config:
            return holder["config"]

    module.PreTrainedConfig = PreTrainedConfig  # type: ignore[attr-defined]
    for name in ("lerobot", "lerobot.configs"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "lerobot.configs.policies", module)
    return holder


def _policy(**kwargs: Any) -> LerobotLocalPolicy:
    with patch.object(LerobotLocalPolicy, "_load_model"):
        return LerobotLocalPolicy(pretrained_name_or_path="lerobot/pi05_droid", **kwargs)


def _config(**kwargs: Any) -> Any:
    config = _policy(**kwargs)._inference_config()
    assert config is not None
    return config


def test_a_cpu_trained_checkpoint_runs_on_the_gpu(lerobot_config, monkeypatch, caplog) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with caplog.at_level(logging.WARNING):
        config = _config()
    assert config.device == "cuda"
    assert "names device 'cpu'" in caplog.text and "Pass device= to choose" in caplog.text


def test_a_requested_device_wins_silently(lerobot_config, monkeypatch, caplog) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with caplog.at_level(logging.WARNING):
        config = _config(device="cpu")
    assert config.device == "cpu" and "names device" not in caplog.text


def test_the_checkpoints_compile_is_off_for_inference_unless_asked_for(lerobot_config, caplog) -> None:
    lerobot_config["config"] = _Config(compile_model=True)
    with caplog.at_level(logging.WARNING):
        assert _config().compile_model is False
    assert "compiles for minutes" in caplog.text and "compile_model=True" in caplog.text
    lerobot_config["config"] = _Config(compile_model=True)
    assert _config(compile_model=True).compile_model is True
    lerobot_config["config"] = _Config(compile_model=False)
    assert _config(compile_model=True).compile_model is True


def test_a_compile_posture_that_is_not_a_bool_is_refused() -> None:
    with pytest.raises(ValueError, match="compile_model"):
        _policy(compile_model="false")


def test_the_load_builds_the_policy_from_that_config(lerobot_config, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    lerobot_config["config"] = _Config(device="cpu", compile_model=True)
    built: dict[str, Any] = {}

    class _Policy:
        config = lerobot_config["config"]

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> Any:
            built.update(kwargs)
            inst = MagicMock()
            inst.config = kwargs["config"]
            inst.parameters.return_value = iter(())
            return inst

    monkeypatch.setattr(policy_mod, "resolve_policy_class_from_hub", lambda path, revision=None: (_Policy, "pi05"))
    monkeypatch.setattr(LerobotLocalPolicy, "_load_processor_bridge", lambda self: None, raising=False)
    policy = _policy(cache_model=False)
    with patch.object(LerobotLocalPolicy, "_auto_detect_actions_per_step"):
        try:
            LerobotLocalPolicy._load_model(policy)
        except Exception:  # noqa: BLE001 - later load stages are not under test
            pass
    assert built["config"].device == "cuda" and built["config"].compile_model is False
    assert policy._device == torch.device("cuda")


def test_best_device_prefers_cuda_then_mps_then_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert policy_mod.best_inference_device() == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert policy_mod.best_inference_device() == "cuda"
