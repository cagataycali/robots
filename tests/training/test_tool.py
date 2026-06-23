"""Tests for the train_policy agent tool (Strands @tool wrapper)."""

import json

import pytest

from strands_robots.tools import train_policy


def _text(res):
    return res["content"][0]["text"]


@pytest.fixture
def dataset_root(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 5}))
    return str(tmp_path)


class TestActions:
    def test_list(self):
        res = train_policy(action="list")
        assert res["status"] == "success"
        for p in ("mock", "lerobot_local", "groot", "cosmos3"):
            assert p in _text(res)

    def test_validate_clean(self, dataset_root, tmp_path):
        res = train_policy(
            action="validate", provider="mock",
            dataset_root=dataset_root, base_model="m",
            output_dir=str(tmp_path / "o"), steps=10,
        )
        assert res["status"] == "success"
        assert "valid and launchable" in _text(res)

    def test_validate_reports_problems(self, tmp_path):
        res = train_policy(
            action="validate", provider="mock",
            dataset_root=str(tmp_path / "nope"), base_model="m",
            output_dir=str(tmp_path / "o"), steps=0,
        )
        assert res["status"] == "error"
        assert "validation problems" in _text(res)

    def test_missing_required_args(self):
        res = train_policy(action="train", provider="mock")
        assert res["status"] == "error"
        assert "required" in _text(res)

    def test_train_mock_full_loop(self, dataset_root, tmp_path):
        out = str(tmp_path / "out")
        res = train_policy(
            action="train", provider="mock",
            dataset_root=dataset_root, base_model="mock/base",
            output_dir=out, steps=50,
        )
        assert res["status"] == "success", _text(res)
        assert res["job_id"]
        assert res["checkpoint_dir"]
        assert res["metrics"]["learning"] is True
        assert "create_policy(" in _text(res)

    def test_status_requires_job_id(self):
        res = train_policy(action="status", provider="mock")
        assert res["status"] == "error"
        assert "job_id" in _text(res)

    def test_status_verdict(self):
        res = train_policy(action="status", provider="mock", job_id="mock-123")
        assert res["status"] == "success"
        assert res["metrics"]["learning"] is True

    def test_unknown_action(self, dataset_root, tmp_path):
        res = train_policy(
            action="frobnicate", provider="mock",
            dataset_root=dataset_root, base_model="m",
            output_dir=str(tmp_path / "o"),
        )
        assert res["status"] == "error"
        assert "Unknown action" in _text(res)


class TestProviderRouting:
    def test_lerobot_validate_routes_to_lerobot(self, dataset_root, tmp_path):
        # non-native policy_type -> lerobot-specific validation message
        res = train_policy(
            action="validate", provider="lerobot_local",
            dataset_root=dataset_root, base_model="",
            output_dir=str(tmp_path / "o"), steps=10,
            extra={"policy_type": "openvla"},
        )
        assert res["status"] == "error"
        assert "not LeRobot-native" in _text(res)

    def test_groot_requires_embodiment(self, dataset_root, tmp_path):
        res = train_policy(
            action="validate", provider="groot",
            dataset_root=dataset_root, base_model="nvidia/GR00T-N1.5-3B",
            output_dir=str(tmp_path / "o"), steps=10,
            extra={"groot_root": "/tmp"},  # missing launch script -> also errors, but embodiment first
        )
        assert res["status"] == "error"
        assert "embodiment is required" in _text(res)
