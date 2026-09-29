"""RslRlTrainer: TrainSpec -> mjlab mapping, preflight, checkpoint discovery, tool gate.

No GPU, no mjlab run: everything here is the contract around ``run_train``.
"""

from __future__ import annotations

import os
import time

import pytest

from strands_robots.training import TrainSpec, create_trainer, list_trainers
from strands_robots.training.base import Trainer
from strands_robots.training.rsl_rl import DEFAULT_TASK_FOR_EMBODIMENT, RslRlTrainer

pytest.importorskip("mjlab", reason="rsl_rl trainer needs mjlab for task-registry preflight")


@pytest.fixture
def trainer() -> RslRlTrainer:
    return RslRlTrainer()


class TestRegistration:
    def test_one_name_owns_train_and_deploy(self):
        assert isinstance(create_trainer("rsl_rl_onnx"), RslRlTrainer)
        assert isinstance(create_trainer("rsl_rl"), RslRlTrainer)
        assert isinstance(create_trainer("mjlab"), RslRlTrainer)
        assert "rsl_rl" in list_trainers()

    def test_provider_name_pairs_with_the_policy(self, trainer):
        from strands_robots.policies.rsl_rl_onnx import RslRlOnnxPolicy

        assert trainer.provider_name == "rsl_rl_onnx"
        assert trainer.provider_name == RslRlOnnxPolicy.provider_name.fget(None)

    def test_rl_trainer_waives_the_dataset_gate(self, trainer):
        assert Trainer.requires_dataset is True  # the default
        assert trainer.requires_dataset is False


class TestTaskResolution:
    def test_extra_task_wins(self, trainer):
        spec = TrainSpec(output_dir="o", embodiment="so101", extra={"task": "Mjlab-Cartpole-Balance"})
        assert trainer.task_for(spec) == "Mjlab-Cartpole-Balance"

    @pytest.mark.parametrize("emb,task", sorted(DEFAULT_TASK_FOR_EMBODIMENT.items()))
    def test_embodiment_table(self, trainer, emb, task):
        assert trainer.task_for(TrainSpec(output_dir="o", embodiment=emb.upper())) == task

    def test_no_task_is_a_problem(self, trainer):
        problems = trainer.validate(TrainSpec(output_dir="o", steps=1, global_batch_size=1))
        assert any("a task is required" in p for p in problems)


class TestValidate:
    def test_clean_spec(self, trainer):
        spec = TrainSpec(output_dir="runs/x", extra={"task": "Strands-Reach-SO101"}, steps=3, global_batch_size=8)
        assert trainer.validate(spec) == []

    def test_strands_task_is_registered(self, trainer):
        spec = TrainSpec(output_dir="o", embodiment="so101", steps=1, global_batch_size=1)
        assert trainer.validate(spec) == []

    def test_unknown_task_names_the_known_ones(self, trainer):
        problems = trainer.validate(TrainSpec(output_dir="o", extra={"task": "Nope"}, steps=1, global_batch_size=1))
        assert len(problems) == 1
        assert "unknown mjlab task 'Nope'" in problems[0]
        assert "Strands-Reach-SO101" in problems[0]

    def test_dataset_and_method_are_refused(self, trainer):
        spec = TrainSpec(
            output_dir="o", extra={"task": "Strands-Reach-SO101"}, dataset_root="/tmp", method="lora", steps=1
        )
        text = " ".join(trainer.validate(spec))
        assert "dataset_root must be empty" in text
        assert "method must be 'full'" in text

    def test_missing_output_dir(self, trainer):
        assert any("output_dir" in p for p in trainer.validate(TrainSpec(extra={"task": "Strands-Reach-SO101"})))

    def test_security_gate_runs_first(self, trainer):
        spec = TrainSpec(output_dir="o", extra={"task": "Strands-Reach-SO101", "-Bad": 1}, steps=1)
        assert any("extra key" in p for p in trainer.validate(spec))

    def test_run_size_gate(self, trainer):
        spec = TrainSpec(output_dir="o", extra={"task": "Strands-Reach-SO101"}, steps=0, global_batch_size=8)
        assert any("steps" in p for p in trainer.validate(spec))


class TestCheckpointDiscovery:
    def _run(self, root, name, its, age=0.0):
        d = root / "so101_reach" / name
        d.mkdir(parents=True)
        for i in its:
            p = d / f"model_{i}.pt"
            p.write_bytes(b"x")
            t = time.time() - age
            os.utime(p, (t, t))
        return d

    def test_latest_run_and_highest_iteration(self, trainer, tmp_path):
        old = self._run(tmp_path, "2026-01-01_00-00-00_a", [0, 50, 99], age=100)
        new = self._run(tmp_path, "2026-01-02_00-00-00_b", [0, 9, 100, 20], age=0)
        assert trainer.latest_checkpoint(str(tmp_path)) == str(new)
        assert trainer.latest_model_file(new).name == "model_100.pt"
        assert trainer.latest_model_file(old).name == "model_99.pt"

    def test_empty(self, trainer, tmp_path):
        assert trainer.latest_checkpoint(str(tmp_path)) is None
        assert trainer.latest_checkpoint(str(tmp_path / "missing")) is None
        assert trainer.latest_model_file(tmp_path) is None

    def test_export_refuses_without_a_model(self, trainer, tmp_path):
        with pytest.raises(FileNotFoundError):
            trainer.export(TrainSpec(output_dir=str(tmp_path), extra={"task": "Strands-Reach-SO101"}), str(tmp_path))

    def test_export_reads_the_task_marker(self, trainer, tmp_path):
        (tmp_path / "model_1.pt").write_bytes(b"x")
        spec = TrainSpec(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="task unknown"):
            trainer.export(spec, str(tmp_path))
        (tmp_path / "strands_task.txt").write_text("Strands-Reach-SO101\n")
        # A fresh ONNX newer than the model short-circuits the (GPU) export.
        (tmp_path / "model_1.onnx").write_bytes(b"o")
        assert trainer.export(spec, str(tmp_path)).endswith("model_1.onnx")


class TestToolGate:
    def test_train_policy_accepts_rsl_rl_without_a_dataset(self, monkeypatch, tmp_path):
        """The tool no longer demands a data source for a trainer that reads none."""
        import importlib

        tp = importlib.import_module("strands_robots.tools.train_policy")
        seen = {}

        class _Stub(RslRlTrainer):
            def train(self, spec):
                seen["spec"] = spec
                from strands_robots.training.base import TrainResult

                return TrainResult(status="success", job_id="j", checkpoint_dir="c", exported_model="c/m.onnx")

        monkeypatch.setattr(tp, "create_trainer", lambda provider: _Stub())
        res = tp.train_policy(
            action="train",
            provider="rsl_rl",
            output_dir=str(tmp_path),
            steps=3,
            batch_size=16,
            extra={"task": "Strands-Reach-SO101"},
        )
        assert res["status"] == "success", res
        assert seen["spec"].dataset_root == ""
        assert seen["spec"].global_batch_size == 16

    def test_supervised_trainers_still_need_data(self, tmp_path):
        import importlib

        tp = importlib.import_module("strands_robots.tools.train_policy")
        res = tp.train_policy(action="validate", provider="mock", output_dir=str(tmp_path))
        assert res["status"] == "error"
        assert "data source" in res["content"][0]["text"]


class TestNextStepHint:
    def test_the_train_result_names_the_rsl_rl_onnx_load(self, monkeypatch, tmp_path):
        """The next-step line loads the ONNX actor through its own provider.

        ``create_policy('<checkpoint_dir>')`` (the generic hint) resolves to
        ``lerobot_local`` and raises its trust gate for an rsl_rl run; the hint a
        user copies must be the call that works.
        """
        import importlib

        tp = importlib.import_module("strands_robots.tools.train_policy")

        class _Stub(RslRlTrainer):
            def train(self, spec):
                from strands_robots.training.base import TrainResult

                return TrainResult(
                    status="success", job_id="j", checkpoint_dir="/r/run", exported_model="/r/run/model_29.onnx"
                )

        monkeypatch.setattr(tp, "create_trainer", lambda provider: _Stub())
        res = tp.train_policy(
            action="train",
            provider="rsl_rl",
            output_dir=str(tmp_path),
            steps=3,
            batch_size=16,
            extra={"task": "Strands-Reach-SO101"},
        )
        text = res["content"][0]["text"]
        assert "Load the result with: create_policy('rsl_rl_onnx', onnx_path='/r/run/model_29.onnx')" in text
        assert "create_policy('/r/run')" not in text

    def test_the_default_load_call_is_the_lerobot_path_form(self):
        from strands_robots.training.mock import MockTrainer

        assert MockTrainer().load_call("/x/pretrained_model") == "create_policy('/x/pretrained_model')"


class TestOneRunPerProcess:
    def test_a_second_concurrent_train_is_refused_with_the_cause(self, monkeypatch, tmp_path):
        """Two trains in one process: the second returns a named refusal, not a Warp graph-capture error."""
        import threading

        from strands_robots.training import rsl_rl as mod

        entered = threading.Event()
        release = threading.Event()

        def fake_locked(self, spec):
            from strands_robots.training.base import TrainResult

            mod._ACTIVE_RUN["run"] = "Strands-Reach-SO101 x 16 envs"
            entered.set()
            release.wait(5)
            return TrainResult(status="success", job_id="first", checkpoint_dir=str(tmp_path))

        monkeypatch.setattr(RslRlTrainer, "_train_locked", fake_locked)
        spec = TrainSpec(
            dataset_root="",
            output_dir=str(tmp_path),
            steps=3,
            global_batch_size=16,
            extra={"task": "Strands-Reach-SO101"},
        )
        trainer = RslRlTrainer()
        first: dict = {}
        t = threading.Thread(target=lambda: first.setdefault("res", trainer.train(spec)))
        t.start()
        assert entered.wait(5)
        try:
            second = RslRlTrainer().train(spec)
        finally:
            release.set()
            t.join(5)
        assert second.status == "error"
        assert "already active in this process" in second.message
        assert "Strands-Reach-SO101 x 16 envs" in second.message
        assert "SequentialToolExecutor" in second.message
        assert first["res"].status == "success"
        # The lock is released once the first run returns: a third run may start.
        assert mod._TRAIN_LOCK.acquire(blocking=False)
        mod._TRAIN_LOCK.release()
