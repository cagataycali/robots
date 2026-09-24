# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A native driver's ``start_task`` consults the provider's preflight first.

:meth:`~strands_robots.policies.base.Policy.preflight` is the seam a provider
uses to refuse a configuration WITHOUT constructing, and therefore before any
weight download. The simulation engine ran it; the physical arm gained it; the
two drivers that build a policy from the registry themselves - UR and Feetech -
called ``create_policy`` directly. So the one failure mode
:func:`~strands_robots.drivers.rollout.policy_from_provider` exists to prevent
was still reachable through it.

Measured on an SO-101 over the MuJoCo twin, before this change::

    start_task(policy_provider="lerobot_local", policy_type="smolvla",
               pretrained_name_or_path="lerobot/smolvla_base",
               embodiment="so_real")
    -> status: success           after 20.1 s of model loading
    get_task_status()
    -> running: False, steps: 0, exit_reason: "policy",
       refusal: "Robot supplies 0 camera(s) [] but the policy requires image
                 input(s) ['observation.images.camera1', ...]"

The verb answered "started" on an energized arm for a rollout that could not
take one step: the driver's observation is joints only, and no rename can feed
an image feature from it. Afterwards the same call is refused in 2.0 s with the
provider's own message and no rollout at all.

Four rules, driven through the real ``start_task`` of both drivers that build:

1. A preflight that refuses ends the verb with the provider's own message, and
   ``create_policy`` is never reached.
2. A preflight that passes is handed the driver's OWN observation keys - the
   ones the rollout will hand the policy, which is what such a hook validates.
3. A provider that leaves the default no-op hook in place has no observation
   read on its behalf: that read crosses the wire.
4. An observation the robot cannot serve is not a verdict on the policy
   configuration: the build goes ahead and the rollout reports the read.
"""

from __future__ import annotations

import inspect
import sys
import types
from typing import Any

import pytest

import strands_robots.policies as policies
from strands_robots.drivers import rollout as rollout_module
from strands_robots.drivers.feetech import driver as feetech_module
from strands_robots.drivers.feetech.driver import FeetechDriver
from strands_robots.drivers.ur import URDriver
from strands_robots.policies import factory as policy_factory
from strands_robots.policies import register_policy
from strands_robots.policies.mock import MockPolicy
from tests.mocks.ur_rtde import FakeRTDE, text_of

#: The message a refusing hook raises, asserted verbatim so the caller is shown
#: the provider's own words rather than a generic "could not build".
_REFUSAL = "preflight: image feature 'observation.images.wrist_image' has no camera to feed it"
_REFUSING = "driver_preflight_refusing_probe"
_OBSERVING = "driver_preflight_observing_probe"
#: A runtime-registered provider is absent from the JSON registry, so the
#: required-keyword guard ahead of the preflight has nothing to ask of it.
ROLLED_OUT = {"status": "success", "content": [{"text": "rollout started"}]}


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


class _Probe:
    """A connected driver whose rollout is stubbed, plus the event log."""

    def __init__(self, driver: Any, log: list[str], joint_keys: set[str]) -> None:
        self.driver = driver
        self.log = log
        self.joint_keys = joint_keys

    def start_task(self, **kwargs: Any) -> dict[str, Any]:
        return self.driver.start_task("pick up the cube", duration=1.0, **kwargs)


def _feetech_probe(monkeypatch: pytest.MonkeyPatch, *, readable: bool) -> _Probe:
    """A Feetech SO-101 over the MuJoCo twin - no serial bus needed."""
    driver = FeetechDriver(tool_name="so101", transport="twin")
    assert driver.connect_eagerly() is None
    log: list[str] = []
    keys = set(feetech_module.read_joints(driver))
    assert keys, "the twin served no joints, so nothing below grades a key set"

    def read(_driver: Any) -> dict[str, float]:
        log.append("observe")
        if not readable:
            raise RuntimeError("motors bus did not answer")
        return dict.fromkeys(keys, 0.0)

    monkeypatch.setattr(feetech_module, "read_joints", read)
    monkeypatch.setattr(FeetechDriver, "run_policy", lambda *a, **k: ROLLED_OUT)
    return _Probe(driver, log, keys)


def _ur_probe(monkeypatch: pytest.MonkeyPatch, *, readable: bool) -> _Probe:
    """A UR5e over a fake RTDE controller."""
    fake = FakeRTDE()
    control = types.ModuleType("rtde_control")
    receive = types.ModuleType("rtde_receive")
    control.RTDEControlInterface = fake.make_control  # type: ignore[attr-defined]
    receive.RTDEReceiveInterface = fake.make_receive  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rtde_control", control)
    monkeypatch.setitem(sys.modules, "rtde_receive", receive)
    driver = URDriver(tool_name="ur5e", port="192.168.1.10")
    assert driver.connect_eagerly() is None
    log: list[str] = []
    keys = set(driver.get_observation())
    assert keys, "the fake controller served no joints, so nothing below grades a key set"

    def read(_self: Any) -> dict[str, float]:
        log.append("observe")
        if not readable:
            raise RuntimeError("the controller dropped the RTDE link")
        return dict.fromkeys(keys, 0.0)

    monkeypatch.setattr(URDriver, "get_observation", read)
    monkeypatch.setattr(URDriver, "run_policy", lambda *a, **k: ROLLED_OUT)
    return _Probe(driver, log, keys)


#: The two drivers whose ``start_task`` builds from the policy registry. The
#: other eleven refuse the verb outright, so they have no build to gate.
BUILDERS = {"feetech": _feetech_probe, "ur": _ur_probe}


@pytest.fixture(params=sorted(BUILDERS))
def probe(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, provider: Any) -> _Probe:
    """One connected builder driver, with its wire read and rollout recorded."""
    built = BUILDERS[request.param](monkeypatch, readable=True)
    monkeypatch.setattr(
        policies,
        "create_policy",
        _recording_create(built.log, policy_factory.create_policy),
    )
    return built


def _recording_create(log: list[str], real: Any) -> Any:
    """``create_policy``, logging the call so its ORDER against the read is graded."""

    def recording(provider_name: str, **kwargs: Any) -> Any:
        log.append("create_policy")
        return real(provider_name, **kwargs)

    return recording


class TestARefusedConfigurationNeverReachesTheBuild:
    """Rule 1: the provider's own message, and no policy constructed."""

    def test_the_verb_answers_the_providers_refusal(self, probe: _Probe) -> None:
        envelope = probe.start_task(policy_provider=_REFUSING)
        assert envelope["status"] == "error"
        text = text_of(envelope)
        assert text == f"start_task: {_REFUSAL}", text

    def test_the_policy_is_never_built(self, probe: _Probe) -> None:
        """ "Before the build" is the point, not a nicer message.

        The old answer came out of the rollout a *successful* build had already
        started, so a check that ran after ``create_policy`` would still have
        paid the weight download and energized the loop.
        """
        probe.start_task(policy_provider=_REFUSING)
        assert probe.log == ["observe"], probe.log

    def test_no_rollout_is_started(self, probe: _Probe, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            type(probe.driver),
            "run_policy",
            lambda *a, **k: pytest.fail("a rollout was started for a refused configuration"),
        )
        assert probe.start_task(policy_provider=_REFUSING)["status"] == "error"


class TestTheHookJudgesTheObservationTheRolloutWillHand:
    """Rule 2: the driver's own keys, not a guess at them."""

    def test_the_drivers_own_joint_keys_are_handed_to_the_hook(self, probe: _Probe) -> None:
        assert probe.start_task(policy_provider=_OBSERVING) == ROLLED_OUT
        assert _ObservingPolicy.seen_keys == [probe.joint_keys], _ObservingPolicy.seen_keys

    def test_a_passing_hook_still_reaches_the_build(self, probe: _Probe) -> None:
        """Non-vacuity: the check refuses a configuration, not every configuration."""
        assert probe.start_task(policy_provider=_OBSERVING) == ROLLED_OUT
        assert probe.log == ["observe", "create_policy"], probe.log


class TestNoHookMeansNoRead:
    """Rule 3: a wire read is not paid for a provider that cannot use it."""

    def test_a_provider_without_a_hook_is_not_read_for(self, probe: _Probe) -> None:
        assert probe.start_task(policy_provider="mock") == ROLLED_OUT
        assert probe.log == ["create_policy"], probe.log


class TestAnUnreadableRobotIsNotAVerdict:
    """Rule 4: a read that fails leaves the configuration unjudged."""

    @pytest.mark.parametrize("name", sorted(BUILDERS))
    def test_the_build_goes_ahead_when_the_robot_cannot_be_read(
        self, monkeypatch: pytest.MonkeyPatch, provider: Any, name: str
    ) -> None:
        built = BUILDERS[name](monkeypatch, readable=False)
        monkeypatch.setattr(policies, "create_policy", _recording_create(built.log, policy_factory.create_policy))
        assert built.start_task(policy_provider=_OBSERVING) == ROLLED_OUT
        assert built.log == ["observe", "create_policy"], built.log
        assert _ObservingPolicy.seen_keys == [], _ObservingPolicy.seen_keys


class TestTheSeamHasNoOptionalEscape:
    """A driver cannot acquire this build without acquiring the check.

    ``observe`` carries no default on purpose: an optional reader would let the
    next driver to grow a task verb pass ``policy_from_provider`` everything but
    the one argument that makes it check, and silently be back where this
    started - with a green build and a rollout that faults at step 0.
    """

    def test_observe_is_required(self) -> None:
        parameter = inspect.signature(rollout_module.policy_from_provider).parameters["observe"]
        assert parameter.default is inspect.Parameter.empty
