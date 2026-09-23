"""A provider's own preflight is consulted on the arm, before the policy is built.

Every other pre-construction check on the hardware task path runs before the
policy exists - ``_policy_provider_error`` resolves the name,
``_policy_port_error`` and ``_policy_requires_error`` read the registry - and
each exists because the alternative energizes the arm, asks the operator to
approve the rollout, and only then refuses. The one check written FOR that
purpose was reached on the sim path alone: ``SimEngine._preflight_policy_config``
runs :meth:`~strands_robots.policies.base.Policy.preflight` before
``create_policy``, while ``Robot._get_policy`` called ``create_policy``
directly. So a configuration the provider can refuse without constructing - a
chunk count the consumer cannot execute, an ``image_keys`` list that withholds a
feature the embodiment feeds, camera names that cannot be routed to a VLA's
declared image inputs - was refused on the physical arm only after the
multi-minute weight download.

Four rules, driven through the real rollout dispatch on an in-memory arm:

1. A preflight that refuses ends the task with the provider's own message and
   ``create_policy`` is never reached.
2. A preflight that passes is handed the arm's OWN observation keys, camera keys
   included - that routing information is what such a hook validates.
3. A provider that leaves the default no-op hook in place has no observation
   read on its behalf: the read warms and grabs a frame from every configured
   camera, gathered purely to be discarded.
4. An observation the arm cannot serve is not a verdict on the policy
   configuration: the build goes ahead and ``_initialize_policy``, which owns
   that same read, reports it.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState, TaskStatus
from strands_robots.policies import factory as policy_factory
from strands_robots.policies import register_policy
from strands_robots.policies.mock import MockPolicy

_CAMERA = "front"
_REFUSAL = "preflight: image feature 'observation.images.wrist_image' has no camera to feed it"
_REFUSING = "hw_preflight_refusing_probe"
_OBSERVING = "hw_preflight_observing_probe"
#: A port is supplied because a runtime-registered provider is absent from the
#: JSON registry, so ``_policy_port_error`` cannot learn it needs none.
_PORT = 5555
_STEPS = 2


class _RefusingPolicy(MockPolicy):
    """Refuses its configuration without constructing, as a real hook does."""

    @classmethod
    def preflight(cls, observation_keys: set[str], **policy_config: Any) -> None:
        raise ValueError(_REFUSAL)


class _ObservingPolicy(MockPolicy):
    """Records the observation keys the preflight is handed."""

    seen_keys: list[set[str]] = []

    @classmethod
    def preflight(cls, observation_keys: set[str], **policy_config: Any) -> None:
        cls.seen_keys.append(set(observation_keys))


class _Arm:
    """In-memory stand-in for a connected lerobot arm with one camera."""

    def __init__(self, log: list[str], *, readable: bool = True) -> None:
        self.name = "fake_arm"
        self.robot_type = "fake_arm"
        self.sent_actions: list[dict[str, Any]] = []
        self.config = type("Cfg", (), {"cameras": {_CAMERA: object()}})()
        self._log = log
        self._readable = readable

    def get_observation(self) -> dict[str, Any]:
        self._log.append("observe")
        if not self._readable:
            raise RuntimeError("motors bus did not answer")
        return {"shoulder_pan.pos": 0.0, "gripper.pos": 0.0, _CAMERA: object()}

    def send_action(self, action: dict[str, Any]) -> None:
        self.sent_actions.append(action)


@pytest.fixture
def provider() -> Any:
    """Register the two probe providers, and take them back out again."""
    _ObservingPolicy.seen_keys.clear()
    register_policy(_REFUSING, lambda: _RefusingPolicy)
    register_policy(_OBSERVING, lambda: _ObservingPolicy)
    try:
        yield
    finally:
        for name in (_REFUSING, _OBSERVING):
            policy_factory._runtime_registry.pop(name, None)
        _ObservingPolicy.seen_keys.clear()


@pytest.fixture
def hw(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A ``Robot`` on an in-memory arm, with the real ``_get_policy`` path.

    Returns a callable taking ``readable=`` (whether the arm serves an
    observation) and handing back the robot plus the event log every
    observation read and every ``create_policy`` call is appended to, in order.
    """
    log: list[str] = []
    real_create = policy_factory.create_policy

    def recording_create(provider_name: str, **kwargs: Any) -> Any:
        log.append("create_policy")
        return real_create(provider_name, **kwargs)

    monkeypatch.setattr("strands_robots.policies.create_policy", recording_create)
    built: list[Any] = []

    def construct(*, readable: bool = True) -> tuple[Any, list[str]]:
        robot = HwRobot.__new__(HwRobot)
        robot.tool_name_str = "test_arm"
        robot.action_horizon = 1
        robot.data_config = None
        robot.control_frequency = 50.0
        robot.action_sleep_time = 1.0 / 50.0
        robot._task_state = RobotTaskState()
        robot._executor = ThreadPoolExecutor(max_workers=1)
        robot._shutdown_event = threading.Event()
        robot._stop_requested = threading.Event()
        robot._task_admission = threading.Lock()
        robot._task_claimed = False
        robot.mesh = None
        robot.peer_id = None
        robot.robot = _Arm(log, readable=readable)

        async def _connected() -> tuple[bool, str]:
            return (True, "")

        robot._connect_robot = _connected  # type: ignore[method-assign]

        def _no_telemetry(observation: dict[str, Any], *, skip_images: bool = False) -> None:
            return None

        robot._publish_ros_telemetry = _no_telemetry  # type: ignore[method-assign]
        built.append(robot)
        return robot, log

    try:
        yield construct
    finally:
        for robot in built:
            robot._shutdown_event.set()
            robot._task_state.status = TaskStatus.STOPPED
            robot._executor.shutdown(wait=False)


def _drive(robot: Any, provider_name: str) -> dict[str, Any]:
    """Run one rollout through the real dispatch, as ``start_task`` would."""
    return robot._run_control_loop("pick up the cube", _PORT, "localhost", provider_name, 10.0, n_steps=_STEPS)


def _text(result: dict[str, Any]) -> str:
    return " ".join(c.get("text", "") for c in result.get("content", []) if isinstance(c, dict))


class TestARefusedConfigurationNeverBuilds:
    def test_the_providers_own_message_ends_the_task(self, hw: Any, provider: Any) -> None:
        robot, log = hw()
        result = _drive(robot, _REFUSING)

        assert _REFUSAL in _text(result)
        assert robot._task_state.status is TaskStatus.ERROR
        assert "create_policy" not in log, "the refused configuration must not be built"
        assert robot.robot.sent_actions == [], "a refused configuration commands the arm zero times"


class TestAPassingPreflightSeesTheArmsOwnKeys:
    def test_the_camera_key_reaches_the_hook_before_the_build(self, hw: Any, provider: Any) -> None:
        robot, log = hw()
        result = _drive(robot, _OBSERVING)

        assert result["status"] == "success", _text(result)
        assert len(_ObservingPolicy.seen_keys) == 1
        assert {"shoulder_pan.pos", "gripper.pos", _CAMERA} <= _ObservingPolicy.seen_keys[0]
        assert log[:2] == ["observe", "create_policy"], f"the hook is read before the build; got {log[:3]}"


class TestANoOpHookHasNothingGatheredForIt:
    def test_mock_is_built_without_a_preflight_read(self, hw: Any) -> None:
        robot, log = hw()
        result = _drive(robot, "mock")

        assert result["status"] == "success", _text(result)
        assert log[0] == "create_policy", f"no observation may be read for a no-op preflight; got {log[:3]}"


class TestAnUnreadableArmIsNotAPolicyVerdict:
    def test_the_build_goes_ahead_and_the_read_is_reported_by_its_owner(self, hw: Any, provider: Any) -> None:
        robot, log = hw(readable=False)
        result = _drive(robot, _OBSERVING)

        assert "create_policy" in log, "a failed read must not become a refusal of the configuration"
        assert _ObservingPolicy.seen_keys == []
        assert "Failed to initialize policy" in _text(result)
        assert robot._task_state.status is TaskStatus.ERROR
