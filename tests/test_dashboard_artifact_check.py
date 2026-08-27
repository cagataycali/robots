"""An exported "checkpoint" is checked on disk before it is offered to a real arm (Q36).

The defect being pinned: LeRobotTrainer.latest_checkpoint finds the loadable directory from the
resume CONFIG file's parent, so a directory qualifies because a config is in it and nothing ever
looks for weights. A run killed between writing train_config.json and model.safetensors exports
and stages happily, and the failure lands when a policy is asked to drive hardware.

Each test is a physical event that really produces its layout: an unmounted volume, a process
killed mid-save, a full disk, an output dir before the first checkpoint.
"""

from __future__ import annotations

from strands_robots.dashboard.artifact_check import MIN_WEIGHT_BYTES, artifact_verdict


def _weights(p, name="model.safetensors", size=MIN_WEIGHT_BYTES * 4):
    f = p / name
    f.write_bytes(b"\0" * size)
    return f


class TestTheDefect:
    def test_a_config_with_no_weights_is_refused_by_name(self, tmp_path) -> None:
        # A run killed between the two writes. The trainer calls this loadable.
        (tmp_path / "train_config.json").write_text("{}")
        v = artifact_verdict(tmp_path)
        assert v["ok"] is False
        assert v["reason"] == "config_without_weights"
        assert "BY ITS CONFIG" in v["message"], "the operator needs the mechanism, not 'invalid'"
        assert "Re-export" in v["message"]

    def test_a_complete_checkpoint_passes(self, tmp_path) -> None:
        (tmp_path / "train_config.json").write_text("{}")
        _weights(tmp_path)
        v = artifact_verdict(tmp_path)
        assert v["ok"] is True
        assert v["weights"] == ["model.safetensors"]
        assert v["weight_bytes"] > 0

    def test_ok_never_claims_the_policy_loads(self, tmp_path) -> None:
        # Overclaiming here would only move the lie one step later - into the run.
        (tmp_path / "config.json").write_text("{}")
        _weights(tmp_path)
        v = artifact_verdict(tmp_path)
        assert "disk" in v["note"] and "load" in v["note"]


class TestTheOtherWaysItGoesWrong:
    def test_an_unmounted_volume_says_so(self, tmp_path) -> None:
        v = artifact_verdict(tmp_path / "outputs" / "run-7" / "pretrained_model")
        assert v["ok"] is False and v["reason"] == "missing"
        assert "mounted" in v["message"], "a network/removable volume is the common cause"

    def test_an_empty_output_dir_is_a_run_that_died_early(self, tmp_path) -> None:
        v = artifact_verdict(tmp_path)
        assert v["reason"] == "empty"
        assert "first checkpoint" in v["message"]

    def test_a_truncated_weight_file_is_not_weights(self, tmp_path) -> None:
        (tmp_path / "config.json").write_text("{}")
        _weights(tmp_path, size=512)  # one partial block: a killed writer, a full disk
        v = artifact_verdict(tmp_path)
        assert v["ok"] is False and v["reason"] == "truncated"
        assert "free space" in v["message"]

    def test_an_empty_path_is_refused_without_touching_the_filesystem(self, tmp_path) -> None:
        for bad in (None, "", "   "):
            v = artifact_verdict(bad)
            assert v["ok"] is False and v["reason"] == "empty_path"
            assert "named no artifact path" in v["message"]

    def test_a_single_converted_safetensors_file_is_legitimate(self, tmp_path) -> None:
        f = _weights(tmp_path, name="policy.safetensors")
        v = artifact_verdict(f)
        assert v["ok"] is True and v["kind"] == "file"

    def test_a_file_that_is_not_weights_is_refused(self, tmp_path) -> None:
        f = tmp_path / "train_config.json"
        f.write_text("{}")
        v = artifact_verdict(f)
        assert v["ok"] is False and v["reason"] == "not_a_policy"


class TestWhatItMustNotRefuse:
    def test_dcp_shards_one_level_down_count(self, tmp_path) -> None:
        # Cosmos/DCP writes shards into a subdirectory; refusing that would block a whole
        # backend's exports on a layout rule rather than on a problem.
        (tmp_path / "config.json").write_text("{}")
        shard_dir = tmp_path / "checkpoint"
        shard_dir.mkdir()
        _weights(shard_dir, name="__0_0.distcp")
        assert artifact_verdict(tmp_path)["ok"] is True

    def test_weights_without_a_config_pass_with_a_warning(self, tmp_path) -> None:
        # Some loaders infer the architecture. A refusal here would be a guess; a warning
        # is the honest shape, because this is also what half a checkpoint looks like from
        # the other side.
        _weights(tmp_path)
        v = artifact_verdict(tmp_path)
        assert v["ok"] is True
        assert "policy family" in v["warning"]

    def test_torch_pickle_checkpoints_count(self, tmp_path) -> None:
        # The RL algos (PPO, SAC) save .pt, not safetensors.
        _weights(tmp_path, name="actor.pt")
        assert artifact_verdict(tmp_path)["ok"] is True

    def test_an_unreadable_directory_does_not_raise(self, tmp_path) -> None:
        # A request thread must not 500 because of a permission bit.
        d = tmp_path / "locked"
        d.mkdir()
        (d / "config.json").write_text("{}")
        d.chmod(0o000)
        try:
            v = artifact_verdict(d)
            assert isinstance(v, dict) and v["ok"] is False
        finally:
            d.chmod(0o755)
