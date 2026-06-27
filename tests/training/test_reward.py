"""Tests for the SARM reward-model production helpers (the producing half of RA-BC).

Covers :mod:`strands_robots.training.reward`:

* :func:`compute_rabc_weights` - the in-process wrapper that runs a trained SARM
  over a dataset and emits the ``sarm_progress.parquet`` RA-BC consumes. lerobot
  is mocked at the seam (``_require_sarm_progress``) so the test asserts argument
  forwarding + input validation without GPU, a real model, or a real dataset.
* :func:`reward_progress` - tensor/list/scalar return normalization to floats.
* :func:`load_reward_model` - reward-config build + load delegation to lerobot.
"""

from __future__ import annotations

import pytest

from strands_robots.training import reward as reward_mod
from strands_robots.training.reward import (
    compute_rabc_weights,
    load_reward_model,
    reward_progress,
)


class TestComputeRabcWeights:
    """compute_rabc_weights() argument forwarding + input validation."""

    def _patch_compute(self, monkeypatch):
        calls: dict = {}

        def fake_compute_sarm_progress(**kwargs):
            calls.update(kwargs)
            return "/tmp/ds/sarm_progress.parquet"

        monkeypatch.setattr(reward_mod, "_require_sarm_progress", lambda: fake_compute_sarm_progress)
        return calls

    def test_forwards_local_root_and_returns_path(self, monkeypatch):
        calls = self._patch_compute(monkeypatch)
        out = compute_rabc_weights("/ckpt/sarm", dataset_root="/tmp/ds", head_mode="sparse", device="cpu", stride=5)
        assert out == "/tmp/ds/sarm_progress.parquet"
        # A local root is passed through as the dataset arg (lerobot accepts a path).
        assert calls["dataset_repo_id"] == "/tmp/ds"
        assert calls["reward_model_path"] == "/ckpt/sarm"
        assert calls["head_mode"] == "sparse"
        assert calls["device"] == "cpu"
        assert calls["stride"] == 5
        # Visualizations are always disabled by the wrapper.
        assert calls["num_visualizations"] == 0

    def test_forwards_hub_repo_and_output_path(self, monkeypatch):
        calls = self._patch_compute(monkeypatch)
        compute_rabc_weights("/ckpt", dataset_repo_id="org/ds", output_path="/out/p.parquet", device="cpu")
        assert calls["dataset_repo_id"] == "org/ds"
        assert calls["output_path"] == "/out/p.parquet"

    def test_requires_exactly_one_data_source(self, monkeypatch):
        self._patch_compute(monkeypatch)
        with pytest.raises(ValueError, match="exactly one data source"):
            compute_rabc_weights("/ckpt")
        with pytest.raises(ValueError, match="exactly one data source"):
            compute_rabc_weights("/ckpt", dataset_root="/d", dataset_repo_id="org/ds")

    def test_rejects_bad_head_mode(self, monkeypatch):
        self._patch_compute(monkeypatch)
        with pytest.raises(ValueError, match="head_mode"):
            compute_rabc_weights("/ckpt", dataset_root="/d", head_mode="bogus")

    def test_rejects_bad_stride(self, monkeypatch):
        self._patch_compute(monkeypatch)
        with pytest.raises(ValueError, match="stride"):
            compute_rabc_weights("/ckpt", dataset_root="/d", stride=0)

    def test_rejects_leading_dash_value(self, monkeypatch):
        self._patch_compute(monkeypatch)
        with pytest.raises(ValueError, match="must not start with '-'"):
            compute_rabc_weights("-x", dataset_root="/d")


class TestRewardProgress:
    """reward_progress() normalizes the model's compute_reward() return to floats."""

    def test_normalizes_tensor(self):
        torch = pytest.importorskip("torch")

        class Model:
            def compute_reward(self, batch):
                return torch.tensor([[0.1], [0.9]])

        assert reward_progress(Model(), {}) == pytest.approx([0.1, 0.9], abs=1e-6)

    def test_passes_through_plain_list(self):
        class Model:
            def compute_reward(self, batch):
                return [0.2, 0.8]

        assert reward_progress(Model(), {}) == [0.2, 0.8]

    def test_wraps_scalar(self):
        class Model:
            def compute_reward(self, batch):
                return 0.5

        assert reward_progress(Model(), {}) == [0.5]


class TestLoadRewardModel:
    """load_reward_model() builds the reward config and delegates the load to lerobot."""

    def test_builds_config_and_loads(self, monkeypatch):
        lr = pytest.importorskip("lerobot.rewards")
        captured: dict = {}

        def fake_make_cfg(reward_type, **kw):
            captured["type"] = reward_type
            captured.update(kw)
            return "CFG"

        def fake_make_model(cfg):
            captured["model_cfg"] = cfg
            return "MODEL"

        monkeypatch.setattr(lr, "make_reward_model_config", fake_make_cfg)
        monkeypatch.setattr(lr, "make_reward_model", fake_make_model)

        model = load_reward_model("/ckpt/sarm", device="cpu")
        assert model == "MODEL"
        assert captured["type"] == "sarm"
        assert captured["pretrained_path"] == "/ckpt/sarm"
        assert captured["device"] == "cpu"
        assert captured["model_cfg"] == "CFG"
