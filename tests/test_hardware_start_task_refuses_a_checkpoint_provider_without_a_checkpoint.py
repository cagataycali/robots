"""A checkpoint provider with no checkpoint is refused before the arm is touched.

``LerobotLocalPolicy`` constructs with its default ``pretrained_name_or_path=""``
and loads lazily, so ``start_task(policy_provider="lerobot_local")`` with no
checkpoint used to answer "Task started", connect and energize the arm, and
only then fail on the executor thread with "No model loaded and no
pretrained_name_or_path set". The registry now lists the checkpoint under
``lerobot_local``'s ``requires``, and both task entry points judge every
non-port ``requires`` keyword before the bus is claimed - the same place the
port is judged.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState
from strands_robots.registry.policies import get_policy_provider, list_policy_providers
from tests._daemon_executor import DaemonThreadExecutor

QUICKSTART = Path(__file__).resolve().parents[1] / "docs" / "getting-started" / "quickstart.md"


class _Arm:
    name = "so101"
    robot_type = "so_follower"
    is_connected = False
    config = type("Cfg", (), {"port": "/dev/null", "cameras": {}})()

    def connect(self, *a, **k):  # pragma: no cover - the point is that this is never reached
        raise AssertionError("the arm was connected for a task that could not act")


def _hw() -> HwRobot:
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "so101"
    hw.data_config = None
    hw._task_state = RobotTaskState()
    hw._executor = DaemonThreadExecutor(max_workers=1, thread_name_prefix="t")
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_admission = threading.Lock()
    hw._task_claimed = False
    hw.mesh = None
    hw.peer_id = None
    hw.robot = _Arm()
    return hw


def _text(result: dict) -> str:
    return result["content"][0]["text"]


class TestTheRegistryNamesTheCheckpoint:
    def test_lerobot_local_requires_its_checkpoint(self):
        spec = get_policy_provider("lerobot_local")
        assert "pretrained_name_or_path" in (spec.get("requires") or ())

    def test_lerobot_local_still_requires_no_port(self):
        spec = get_policy_provider("lerobot_local")
        assert "port" not in (spec.get("requires") or ())

    @pytest.mark.parametrize("name", sorted(list_policy_providers()))
    def test_every_required_keyword_is_one_its_provider_reads(self, name):
        """A required keyword no ``config_key`` names would refuse every caller forever."""
        spec = get_policy_provider(name) or {}
        assert set(spec.get("requires") or ()) <= set(spec.get("config_keys") or ())


class TestStartTask:
    @pytest.mark.parametrize("kwargs", [{}, {"pretrained_name_or_path": ""}, {"pretrained_name_or_path": None}])
    def test_lerobot_local_without_a_checkpoint_is_refused_before_the_claim(self, kwargs):
        hw = _hw()
        result = hw.start_task("pick up the cube", policy_provider="lerobot_local", duration=10.0, **kwargs)
        assert result["status"] == "error"
        text = _text(result)
        assert text.startswith(
            "start_task: policy_provider='lerobot_local' builds its policy from pretrained_name_or_path"
        )
        assert "Pass pretrained_name_or_path=..." in text
        assert "lerobot/smolvla_base" in text
        assert "Without it the task would start, energize the arm" in text
        assert hw._task_claimed is False
        assert hw._task_state.status.name != "RUNNING"

    def test_lerobot_async_names_both_missing_keywords(self):
        hw = _hw()
        result = hw.start_task("pick", policy_provider="lerobot_async", policy_port=8080, duration=1.0)
        assert result["status"] == "error"
        text = _text(result)
        assert "builds its policy from policy_type and pretrained_name_or_path" in text
        assert "policy_type=... (the checkpoint's policy type" in text
        assert "pretrained_name_or_path=... (a Hub id" in text
        assert "Without them the task would start" in text

    def test_lerobot_async_with_only_one_names_the_other(self):
        hw = _hw()
        result = hw.start_task(
            "pick", policy_provider="lerobot_async", policy_port=8080, duration=1.0, pretrained_name_or_path="me/ckpt"
        )
        text = _text(result)
        assert "from policy_type, and none" in text
        assert "pretrained_name_or_path=..." not in text

    def test_the_port_refusal_still_comes_first_for_a_dialing_provider(self):
        hw = _hw()
        result = hw.start_task("pick", policy_provider="groot", policy_port=None, duration=1.0)
        assert "policy_port is required" in _text(result)

    def test_an_unknown_provider_is_left_to_create_policy(self):
        assert HwRobot._policy_requires_error("no_such_provider", {}, "start_task") is None

    def test_no_provider_is_left_alone(self):
        assert HwRobot._policy_requires_error(None, {}, "start_task") is None
        assert HwRobot._policy_requires_error("", {}, "start_task") is None

    def test_a_provider_without_requirements_passes(self):
        assert HwRobot._policy_requires_error("mock", {}, "start_task") is None

    def test_a_required_port_belongs_to_the_other_guard(self):
        """``port`` arrives as ``policy_port``, never in ``policy_kwargs``.

        Judging it here would find it absent for every caller and refuse a port
        that WAS supplied; ``_policy_port_error`` is the one that reads it.
        """
        assert HwRobot._policy_requires_error("groot", {}, "start_task") is None


class TestExecuteTask:
    def test_lerobot_local_without_a_checkpoint_is_refused(self):
        hw = _hw()
        result = hw._execute_task_sync("pick", policy_provider="lerobot_local", duration=1.0)
        assert result["status"] == "error"
        assert _text(result).startswith("execute_task: policy_provider='lerobot_local' builds its policy from")
        assert hw._task_claimed is False

    def test_a_pre_built_policy_object_makes_the_keyword_inert(self):
        """With ``policy_object`` nothing is built, so nothing is required of the kwargs."""
        hw = _hw()
        seen = {}

        def fake_claim(instruction):
            seen["claimed"] = instruction
            return {"status": "error", "content": [{"text": "stop here"}]}

        hw._claim_task = fake_claim  # type: ignore[method-assign]
        result = hw._execute_task_sync("pick", policy_provider="lerobot_local", duration=1.0, policy_object=object())
        assert seen == {"claimed": "pick"}
        assert _text(result) == "stop here"


def test_the_quickstart_hands_start_task_the_checkpoint_it_trained():
    text = QUICKSTART.read_text()
    assert 'policy_provider="lerobot_local",\n                    pretrained_name_or_path="/tmp/pick_ckpt"' in text
