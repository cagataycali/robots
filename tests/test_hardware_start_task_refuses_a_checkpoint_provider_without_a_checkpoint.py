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

import re
from pathlib import Path

import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.registry.policies import get_policy_provider, list_policy_providers
from tests._hardware_robot import hardware_robot_on

_DOCS = Path(__file__).resolve().parents[1] / "docs"
#: The pages that carry the quickstart's record, train, run story now: the old
#: getting-started/quickstart.md redirects to start/first-robot.md and the
#: record-train-deploy recipe to learn/training/lerobot.md, which hands the trained
#: checkpoint to the provider.
QUICKSTART_PAGES = (_DOCS / "start" / "first-robot.md", _DOCS / "learn" / "training" / "lerobot.md")
#: A documented lerobot_local build or task start, with its full argument list.
_LEROBOT_LOCAL_CALL = re.compile(
    r"(?:create_policy|start_task|run_policy)\((?:[^()]|\([^()]*\))*?lerobot_local(?:[^()]|\([^()]*\))*\)",
    re.DOTALL,
)


class _Arm:
    name = "so101"
    robot_type = "so_follower"
    is_connected = False
    config = type("Cfg", (), {"port": "/dev/null", "cameras": {}})()

    def connect(self, *a, **k):  # pragma: no cover - the point is that this is never reached
        raise AssertionError("the arm was connected for a task that could not act")


def _hw() -> HwRobot:
    hw = hardware_robot_on(_Arm(), tool_name="so101")
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
    """Every documented ``lerobot_local`` build or task start names its checkpoint.

    The quickstart used to hand ``start_task`` the checkpoint it had just trained
    in one call; the same claim is graded on the pages that carry that story now,
    so no page shows the provider being started without ``pretrained_name_or_path``,
    which is exactly the call the guard above refuses.
    """
    calls = [
        (page.relative_to(_DOCS).as_posix(), match.group(0))
        for page in QUICKSTART_PAGES
        for match in _LEROBOT_LOCAL_CALL.finditer(page.read_text(encoding="utf-8"))
    ]
    assert calls, "premise: the quickstart pages no longer build or start a lerobot_local policy"
    missing = [(page, call) for page, call in calls if "pretrained_name_or_path" not in call]
    assert not missing, f"lerobot_local is handed out without the checkpoint it needs: {missing}"
