# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``create_policy("rl", checkpoint_dir=...)`` loads an rsl_rl run, a ``model_<n>.pt`` file or a Hub repo id.

Before this, the provider took one shape only: a directory a strands trainer
wrote (``policy.pt`` + ``policy_meta.json``). The converter for rsl_rl's
``model_<iteration>.pt`` existed (``strands_robots.training.rl.rsl_rl``) but only
the Isaac Lab trainer's ``export`` action reached it, so a checkpoint trained on
another machine, or published on the HuggingFace Hub, needed a hand-written
adapter. These cells pin the three shapes the provider now detects, the order it
detects them in, and the refusal a directory holding none of them earns. No
rsl_rl, no Isaac Lab and no network: the checkpoint is written here in rsl_rl
5.x's layout and ``snapshot_download`` is replaced by a local copy.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from strands_robots.policies import create_policy  # noqa: E402
from strands_robots.policies import rl as rl_module  # noqa: E402
from strands_robots.policies.rl import RLCheckpointPolicy  # noqa: E402
from tests.training.test_rsl_rl_actor_export import ACT, OBS, _reference, write_rsl_rl_run  # noqa: E402

HUB_ID = "someone/an-isaaclab-run"


def _obs(seed: int = 1, n: int = 16) -> Any:
    return torch.randn(n, OBS, generator=torch.Generator().manual_seed(seed)) * 3


def _load(checkpoint_dir: str) -> RLCheckpointPolicy:
    policy = create_policy("rl", checkpoint_dir=checkpoint_dir)
    assert isinstance(policy, RLCheckpointPolicy)
    return policy


def _acts(policy: RLCheckpointPolicy, x: Any) -> Any:
    """Actions through the Policy surface, one observation per call.

    Compared with the batched reference under ``rtol=1e-5``: a row-at-a-time
    matmul and a batched one take different BLAS paths on x86, and float32
    outputs in the tens differ in the last bits (a CI runner showed 1e-5 where
    Apple silicon agreed exactly); the Tanh-vs-ELU divergence this guards
    against is orders of magnitude larger.
    """
    rows = []
    for row in x:
        out = policy.get_actions_sync({"policy_obs": row.tolist()}, "walk")[0]
        rows.append([out[k] for k in policy.action_keys])
    return torch.tensor(rows)


def _bind(policy: RLCheckpointPolicy) -> None:
    policy.set_robot_state_keys([f"j{i}" for i in range(ACT)])


class TestARunDirectoryLoads:
    def test_the_newest_model_in_a_run_dir_acts_as_rsl_rl_would(self, tmp_path: Path) -> None:
        write_rsl_rl_run(tmp_path / "run", iteration=50, normalize=True)
        _, actor = write_rsl_rl_run(tmp_path / "run", iteration=99, normalize=True)
        policy = _load(str(tmp_path / "run"))
        _bind(policy)
        x = _obs()
        assert torch.allclose(_acts(policy, x), _reference(actor, x), rtol=1e-5, atol=1e-5)
        assert policy.trained_by == "rsl_rl"
        assert (tmp_path / "run" / "strands_policy" / "policy_meta.json").is_file()

    def test_a_path_to_one_model_file_loads_that_file(self, tmp_path: Path) -> None:
        model, actor = write_rsl_rl_run(tmp_path / "run", iteration=50, normalize=True)
        write_rsl_rl_run(tmp_path / "run", iteration=99, normalize=True)
        policy = _load(str(model))
        _bind(policy)
        x = _obs(2)
        assert torch.allclose(_acts(policy, x), _reference(actor, x), rtol=1e-5, atol=1e-5)

    def test_a_strands_directory_still_loads_unchanged(self, tmp_path: Path) -> None:
        from strands_robots.training.rl import rsl_rl

        model, actor = write_rsl_rl_run(tmp_path / "run")
        out = rsl_rl.convert_checkpoint(str(model), str(tmp_path / "strands"))
        policy = _load(out)
        _bind(policy)
        x = _obs(3)
        assert torch.allclose(_acts(policy, x), _reference(actor, x), rtol=1e-5, atol=1e-5)


class TestTheConvertedCopyIsReused:
    def test_a_stale_conversion_is_rebuilt_when_the_model_is_newer(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        model, _ = write_rsl_rl_run(run, iteration=99, normalize=True)
        create_policy("rl", checkpoint_dir=str(run))
        converted = run / "strands_policy" / "policy.pt"
        first = converted.stat().st_mtime
        # The run is trained further: same file name, new weights, newer mtime.
        _, actor = write_rsl_rl_run(run, iteration=99, normalize=False)
        os.utime(model, (time.time() + 5, time.time() + 5))
        policy = _load(str(run))
        _bind(policy)
        x = _obs(4)
        assert torch.allclose(_acts(policy, x), _reference(actor, x), rtol=1e-5, atol=1e-5)
        assert converted.stat().st_mtime > first

    def test_a_fresh_conversion_is_not_rewritten(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        write_rsl_rl_run(run, iteration=99)
        create_policy("rl", checkpoint_dir=str(run))
        converted = run / "strands_policy" / "policy.pt"
        stamp = converted.stat().st_mtime_ns
        create_policy("rl", checkpoint_dir=str(run))
        assert converted.stat().st_mtime_ns == stamp


class TestActionNamesComeFromTheRecord:
    def test_record_json_action_names_bind_without_a_robot(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        write_rsl_rl_run(run, iteration=7)
        names = [f"joint_pos.j{i}" for i in range(ACT)]
        (run / "record.json").write_text(json.dumps({"action_names": names, "action_dim": ACT}), encoding="utf-8")
        policy = _load(str(run))
        assert policy.action_keys == names
        out = policy.get_actions_sync({"policy_obs": [0.0] * OBS}, "walk")
        assert len(out) == 1 and list(out[0]) == names

    def test_action_names_of_the_wrong_width_are_left_to_the_robot(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        write_rsl_rl_run(run, iteration=7)
        (run / "record.json").write_text(json.dumps({"action_names": ["only_one"]}), encoding="utf-8")
        policy = _load(str(run))
        assert policy.action_keys == []


class TestAHubRepoIdLoads:
    @pytest.fixture
    def hub(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """A local stand-in for the Hub: ``snapshot_download`` copies the run into a cache dir."""
        import shutil

        remote = tmp_path / "remote"
        _, actor = write_rsl_rl_run(remote, iteration=1499, normalize=True)
        (remote / "record.json").write_text(
            json.dumps({"action_names": [f"joint_pos.j{i}" for i in range(ACT)]}), encoding="utf-8"
        )
        (remote / "exported").mkdir()
        (remote / "exported" / "policy.onnx").write_bytes(b"not fetched")
        calls: list[dict] = []

        def fake_snapshot_download(repo_id: str, **kwargs):
            calls.append({"repo_id": repo_id, **kwargs})
            if repo_id != HUB_ID:
                raise OSError(f"Repository Not Found for url: https://huggingface.co/api/models/{repo_id}")
            cache = tmp_path / "cache" / repo_id.replace("/", "--") / (kwargs.get("revision") or "main")
            if not cache.exists():
                shutil.copytree(remote, cache)
            return str(cache)

        monkeypatch.setattr(rl_module, "_snapshot_download", fake_snapshot_download)
        return actor, calls

    def test_a_bare_repo_id_acts_as_rsl_rl_would(self, hub) -> None:
        actor, calls = hub
        policy = _load(HUB_ID)
        x = _obs(5)
        assert torch.allclose(_acts(policy, x), _reference(actor, x), rtol=1e-5, atol=1e-5)
        assert policy.action_keys == [f"joint_pos.j{i}" for i in range(ACT)]
        assert calls[0]["repo_id"] == HUB_ID and calls[0]["revision"] is None

    def test_only_the_weights_and_their_metadata_are_fetched(self, hub) -> None:
        _, calls = hub
        create_policy("rl", checkpoint_dir=f"hf://{HUB_ID}")
        patterns = calls[0]["allow_patterns"]
        assert "model_*.pt" in patterns and "params/agent.yaml" in patterns and "record.json" in patterns
        assert "policy.pt" in patterns and "policy_meta.json" in patterns
        assert not any("onnx" in p or "mp4" in p or "gif" in p or "png" in p for p in patterns)

    def test_a_revision_suffix_reaches_the_download(self, hub) -> None:
        _, calls = hub
        create_policy("rl", checkpoint_dir=f"{HUB_ID}@v2")
        assert calls[0]["repo_id"] == HUB_ID and calls[0]["revision"] == "v2"

    def test_a_missing_repo_is_refused_by_the_provider(self, hub) -> None:
        with pytest.raises(RuntimeError, match=r"^rl: .*someone/nothing-here.*HuggingFace") as excinfo:
            create_policy("rl", checkpoint_dir="someone/nothing-here")
        assert "local path" in str(excinfo.value)

    def test_an_existing_local_directory_is_never_sent_to_the_network(self, tmp_path: Path, hub) -> None:
        _, calls = hub
        run = tmp_path / "owner" / "repo"
        write_rsl_rl_run(run)
        create_policy("rl", checkpoint_dir=str(run))
        assert calls == []


class TestNothingLoadableIsRefused:
    def test_a_directory_with_none_of_the_three_shapes(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
        with pytest.raises(FileNotFoundError) as excinfo:
            create_policy("rl", checkpoint_dir=str(tmp_path))
        text = str(excinfo.value)
        assert "policy.pt" in text and "model_<n>.pt" in text and "HuggingFace" in text

    def test_a_path_that_is_neither_a_file_a_dir_nor_a_repo_id(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            create_policy("rl", checkpoint_dir=str(tmp_path / "missing" / "deeper"))
