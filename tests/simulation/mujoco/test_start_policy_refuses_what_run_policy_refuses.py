# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A rollout ``run_policy`` refuses is refused by ``start_policy`` too.

``start_policy`` submits the rollout to a worker thread and returns
immediately, and **nothing reads that worker's result** - the ``Future`` is
tracked only to answer "is this robot busy", and is pruned once done. So a
refusal produced on the worker is discarded: the caller keeps the
``status="success"``, "Policy started on 'arm' (async)" it was handed, and a
moment later ``list_policies_running`` reports "No policies running." - the
same reading a rollout that ran to completion gives.

Every other knob on this surface is therefore checked before the submit (the
horizon, the duration, the seed, the video config, the provider keyword bags,
the recording rate). The policy configuration was checked only half-way: the
provider name was resolved, but the provider's own class-level
:meth:`~strands_robots.policies.base.Policy.preflight` hook - which is what
refuses a camera the model's image features cannot be routed from, or an
action-chunk count the consumer cannot execute - ran on the worker. A
``policy_config`` that ``run_policy`` refuses up front, with a message naming
the parameter, was reported as a started rollout.

Pinned as an equality between the two surfaces rather than as a message
substring: the point is not that ``start_policy`` says something, it is that it
says exactly what the blocking surface says for the same request. The refusal
must also leave the robot unclaimed - a guard that returns after announcing the
rollout marks a robot busy forever.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies import factory as policy_factory  # noqa: E402
from strands_robots.policies import register_policy  # noqa: E402
from strands_robots.policies.mock import MockPolicy  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_CAMERA = "overhead"
_REFUSING = "start_policy_preflight_refuses_probe"
_ACCEPTING = "start_policy_preflight_accepts_probe"

# Configurations the probe's hook refuses, one per reason a real hook refuses
# one: a parameter shape it can see, and a camera it cannot route from.
REFUSED_CONFIGS: list[dict[str, Any]] = [
    {"actions_per_step": 0},
    {"image_keys": ["missing-camera"]},
]


class _RefusingPolicy(MockPolicy):
    """Refuses a configuration its ``preflight`` can judge without weights."""

    @classmethod
    def preflight(cls, observation_keys: set[str], **policy_config: Any) -> None:
        if policy_config.get("actions_per_step") == 0:
            raise ValueError("probe: actions_per_step must be a positive integer, got 0")
        for key in policy_config.get("image_keys", []):
            if key not in observation_keys:
                raise ValueError(f"probe: image_keys names '{key}', which the scene cannot feed")


class _AcceptingPolicy(MockPolicy):
    """Overrides ``preflight`` and accepts, so the hook is not the refusal."""

    @classmethod
    def preflight(cls, observation_keys: set[str], **policy_config: Any) -> None:
        return None


def _text(result: dict[str, Any]) -> str:
    """The human-readable half of an agent-tool envelope."""
    return " ".join(c["text"] for c in result.get("content", []) if "text" in c)


def _wait_until_idle(sim: Simulation, timeout: float = 10.0) -> str:
    """Block until no rollout is in flight, then report what the surface says."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        reported = _text(sim.list_policies_running())
        if "No policies running" in reported:
            return reported
        time.sleep(0.02)
    raise AssertionError(f"a rollout is still in flight after {timeout}s: {reported}")


@pytest.fixture
def sim():
    """One arm and one camera, so an observation has a frame to route."""
    engine = Simulation(tool_name="start_policy_preflight_probe", mesh=False)
    engine.create_world()
    engine.add_robot(name="arm", data_config="so100")
    engine.add_camera(_CAMERA, position=[1.0, 0.0, 0.6], target=[0.0, 0.0, 0.2])
    yield engine
    engine.cleanup()


@pytest.fixture(autouse=True)
def probe_providers():
    register_policy(_REFUSING, lambda: _RefusingPolicy)
    register_policy(_ACCEPTING, lambda: _AcceptingPolicy)
    try:
        yield
    finally:
        policy_factory._runtime_registry.pop(_REFUSING, None)
        policy_factory._runtime_registry.pop(_ACCEPTING, None)


class TestBothSurfacesGiveOneVerdict:
    @pytest.mark.parametrize("config", REFUSED_CONFIGS, ids=["parameter-shape", "unroutable-camera"])
    def test_the_async_surface_repeats_the_blocking_surface_verbatim(self, sim, config):
        blocking = sim.run_policy(robot_name="arm", policy_provider=_REFUSING, policy_config=config, n_steps=2)
        started = sim.start_policy(robot_name="arm", policy_provider=_REFUSING, policy_config=config, n_steps=2)

        assert blocking["status"] == "error", blocking
        assert started["status"] == "error", started
        assert _text(started) == _text(blocking)

    @pytest.mark.parametrize("config", REFUSED_CONFIGS, ids=["parameter-shape", "unroutable-camera"])
    def test_a_refused_request_leaves_the_robot_unclaimed(self, sim, config):
        """A guard that returned after the claim would mark the arm busy forever."""
        sim.start_policy(robot_name="arm", policy_provider=_REFUSING, policy_config=config, n_steps=2)

        assert "No policies running" in _text(sim.list_policies_running())
        # The proof the claim is really free: the next rollout is admitted.
        accepted = sim.start_policy(robot_name="arm", policy_provider=_ACCEPTING, n_steps=2)
        assert accepted["status"] == "success", accepted
        _wait_until_idle(sim)

    def test_a_configuration_the_hook_accepts_still_starts(self, sim):
        started = sim.start_policy(robot_name="arm", policy_provider=_ACCEPTING, n_steps=2)

        assert started["status"] == "success", started
        _wait_until_idle(sim)

    def test_a_prebuilt_policy_skips_the_check_the_provider_is_unused_for(self, sim):
        """``run_policy`` skips the pre-flight for a ``policy_object``; so does this."""
        started = sim.start_policy(
            robot_name="arm",
            policy_provider="a-name-no-spelling-resolves",
            policy_object=MockPolicy(),
            n_steps=2,
        )

        assert started["status"] == "success", started
        _wait_until_idle(sim)


class TestTheCheckCostsNothingWithoutAHook:
    def test_a_provider_with_no_hook_has_no_frame_rendered_for_it(self, sim, monkeypatch):
        """``mock`` leaves ``preflight`` alone, so the observation that feeds the
        hook - which renders every camera in the scene - is never gathered, on
        this surface any more than on the blocking one.
        """
        asked_for_images: list[bool] = []
        real = sim.get_observation

        def spy(robot_name=None, skip_images=False, *args, **kwargs):
            asked_for_images.append(not skip_images)
            return real(robot_name, *args, skip_images=skip_images, **kwargs)

        monkeypatch.setattr(sim, "get_observation", spy)

        started = sim.start_policy(robot_name="arm", policy_provider="mock", n_steps=2)
        assert started["status"] == "success", started
        _wait_until_idle(sim)

        assert asked_for_images, "the rollout must have read the observation at all"
        assert not any(asked_for_images), (
            f"no call may render camera frames for a no-op preflight; got {asked_for_images}"
        )
