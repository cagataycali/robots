"""Unit tests for the qwen_vla_train @tool (Phase G)."""

from strands_robots.tools.qwen_vla_train import qwen_vla_train


class TestStagesAndConfig:
    def test_stages_lists_four(self):
        res = qwen_vla_train(action="stages")
        assert res["status"] == "success"
        assert {s["name"] for s in res["stages"]} == {"t2a", "cpt", "sft", "rl"}

    def test_config_t2a(self):
        res = qwen_vla_train(action="config", stage="t2a")
        assert res["status"] == "success"
        assert res["config"]["timestep_dist"] == "sigmoid_normal"
        assert res["config"]["freeze_vlm"] is True

    def test_config_override_max_steps(self):
        res = qwen_vla_train(action="config", stage="cpt", max_steps=500, batch_size=8)
        assert res["config"]["max_steps"] == 500
        assert res["config"]["batch_size"] == 8

    def test_config_bad_stage(self):
        res = qwen_vla_train(action="config", stage="bogus")
        assert res["status"] == "error"
        assert "Unknown stage" in res["message"]


class TestCorpus:
    def test_corpus_preview(self):
        res = qwen_vla_train(action="corpus", embodiment="so100", corpus_size=10)
        assert res["status"] == "success"
        assert res["count"] == 10
        assert len(res["preview"]) == 5
        assert "so100" in res["preview"][0]["prompt"]

    def test_corpus_alias_embodiment(self):
        res = qwen_vla_train(action="corpus", embodiment="aloha", corpus_size=3)
        assert res["status"] == "success"
        assert "aloha" in res["preview"][0]["prompt"]

    def test_corpus_unknown_embodiment(self):
        res = qwen_vla_train(action="corpus", embodiment="no_robot")
        assert res["status"] == "error"
        assert "Unknown embodiment" in res["message"]


class TestValidation:
    def test_unknown_action(self):
        res = qwen_vla_train(action="frobnicate")
        assert res["status"] == "error"

    def test_traversal_output_dir_rejected(self):
        res = qwen_vla_train(action="config", stage="t2a", output_dir="../../etc/x")
        assert res["status"] == "error"
        assert ".." in res["message"]

    def test_protected_checkpoint_rejected(self):
        res = qwen_vla_train(action="config", stage="cpt", checkpoint="/etc/passwd")
        assert res["status"] == "error"
        assert "protected" in res["message"]


class TestTrainAndHotswap:
    def test_train_returns_setup_guidance(self):
        res = qwen_vla_train(action="train", stage="sft")
        assert res["status"] == "error"
        assert "run_sft" in res["message"]
        assert "qwen-vla-train" in res["message"]

    def test_hotswap_requires_checkpoint(self):
        res = qwen_vla_train(action="hotswap", server_port=5599)
        assert res["status"] == "error"
        assert "checkpoint" in res["message"]

    def test_hotswap_no_server(self):
        res = qwen_vla_train(action="hotswap", checkpoint="checkpoints/new", server_port=5599)
        assert res["status"] == "error"
        assert "No Qwen-VLA server reachable" in res["message"]
