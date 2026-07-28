# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The success predicate must not trigger a second camera render per step.

Both branches of legacy ``evaluate()`` called ``resolved_check(_observation_fn())``.
``_observation_fn`` fetches with ``skip_images = not policy.requires_images``, so
for any image-consuming policy (every VLA: ACT, SmolVLA, MolmoAct2) that rendered
EVERY camera a second time per control step purely to hand an observation to a
predicate that does not read pixels.

No in-tree predicate reads them: the built-in check is
``def _contact_check(_obs)`` and ignores its argument entirely, and everything in
``predicates.py`` takes ``sim`` and reads sim state directly.

Measured over 20 eval steps with one 224x224 camera plus the default camera::

    success_fn='contact' -> image-rendering observation calls = 40   (20 suffice)
    success_fn=None      -> image-rendering observation calls = 20

and end to end on a 30-step eval: **0.85s -> 0.56s**, a 34% saving. ``requires_images``
exists precisely to avoid this cost - ``policies/base.py`` documents "a ~10x
throughput win at 500Hz when no cameras are needed".

The fix gives the predicate its own image-free fetch, with an opt-in escape hatch:
a caller-supplied ``success_fn`` carrying ``requires_images = True`` still receives
pixels, so the cheap default does not silently withhold data from a predicate that
wants it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.base import Policy  # noqa: E402
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402
from strands_robots.simulation.policy_runner import PolicyRunner  # noqa: E402


class _ImagePolicy(Policy):
    """An image-consuming policy - the case that paid the double render."""

    def __init__(self, keys) -> None:
        super().__init__()
        self._keys = list(keys)

    @property
    def provider_name(self) -> str:
        return "image"

    def set_robot_state_keys(self, keys) -> None:
        pass

    @property
    def requires_images(self) -> bool:
        return True

    async def get_actions(self, observation, instruction, **kwargs):
        return [dict.fromkeys(self._keys, 0.0)]


class _StatePolicy(_ImagePolicy):
    """A policy that needs no pixels; both fetches should skip images."""

    @property
    def requires_images(self) -> bool:
        return False


def _sim():
    sim = MuJoCoSimEngine()
    sim.create_world()
    assert sim.add_robot("so101")["status"] == "success"
    assert (
        sim.add_camera("wrist", position=[0.3, 0.0, 0.3], target=[0.0, 0.0, 0.1], width=64, height=64)["status"]
        == "success"
    )
    return sim


def _instrument(sim) -> dict[str, int]:
    """Count get_observation calls by whether they render images."""
    counts = {"with_images": 0, "skip_images": 0}
    original = sim.get_observation

    def counting(*args, **kwargs):
        key = "skip_images" if kwargs.get("skip_images", False) else "with_images"
        counts[key] += 1
        return original(*args, **kwargs)

    sim.get_observation = counting
    return counts


def _evaluate(sim, policy, success_fn, max_steps: int = 5):
    return PolicyRunner(sim).evaluate(
        "so101", policy, n_episodes=1, max_steps=max_steps, action_horizon=1, success_fn=success_fn
    )


class TestTheSuccessCheckDoesNotRenderImages:
    def test_an_image_policy_renders_once_per_step_not_twice(self):
        """Regression: 20 steps cost 40 image-rendering observation calls."""
        sim = _sim()
        try:
            counts = _instrument(sim)
            policy = _ImagePolicy(sim.robot_action_keys("so101"))

            result = _evaluate(sim, policy, "contact", max_steps=5)

            assert result["status"] == "success", result
            assert counts["with_images"] == 5, f"{counts['with_images']} image-rendering calls for 5 steps (expected 5)"
            assert counts["skip_images"] == 5, f"the predicate fetch is not image-free: {counts}"
        finally:
            sim.destroy()

    def test_the_no_predicate_path_is_unchanged(self):
        """success_fn=None never had the extra fetch; it must stay that way."""
        sim = _sim()
        try:
            counts = _instrument(sim)
            policy = _ImagePolicy(sim.robot_action_keys("so101"))

            _evaluate(sim, policy, None, max_steps=5)

            assert counts["with_images"] == 5
            assert counts["skip_images"] == 0
        finally:
            sim.destroy()

    def test_a_state_only_policy_never_renders(self):
        """requires_images=False must keep BOTH fetches image-free."""
        sim = _sim()
        try:
            counts = _instrument(sim)
            policy = _StatePolicy(sim.robot_action_keys("so101"))

            _evaluate(sim, policy, "contact", max_steps=5)

            assert counts["with_images"] == 0, counts
            assert counts["skip_images"] == 10, counts
        finally:
            sim.destroy()

    def test_a_callable_predicate_also_gets_the_cheap_fetch(self):
        """Not just the built-in 'contact' string."""
        sim = _sim()
        try:
            counts = _instrument(sim)
            policy = _ImagePolicy(sim.robot_action_keys("so101"))

            _evaluate(sim, policy, lambda obs: False, max_steps=5)

            assert counts["with_images"] == 5, counts
            assert counts["skip_images"] == 5, counts
        finally:
            sim.destroy()


class TestPredicatesThatWantPixelsCanOptIn:
    def test_requires_images_on_the_predicate_delivers_image_keys(self):
        """The escape hatch: no silent withholding from a predicate that asks."""
        sim = _sim()
        seen: list[list[str]] = []

        def pixel_predicate(observation):
            seen.append([key for key in observation if "wrist" in key or "image" in key])
            return False

        pixel_predicate.requires_images = True
        try:
            policy = _ImagePolicy(sim.robot_action_keys("so101"))

            _evaluate(sim, policy, pixel_predicate, max_steps=3)

            assert seen, "the predicate was never called"
            assert all(keys for keys in seen), f"opt-in predicate got no image keys: {seen}"
        finally:
            sim.destroy()

    def test_a_plain_predicate_gets_no_image_keys(self):
        """The counterpart: the default really is image-free."""
        sim = _sim()
        seen: list[list[str]] = []

        def plain_predicate(observation):
            seen.append([key for key in observation if "wrist" in key or "image" in key])
            return False

        try:
            policy = _ImagePolicy(sim.robot_action_keys("so101"))

            _evaluate(sim, policy, plain_predicate, max_steps=3)

            assert seen, "the predicate was never called"
            assert not any(keys for keys in seen), f"images were still rendered for the predicate: {seen}"
        finally:
            sim.destroy()

    def test_the_opt_in_renders_twice_by_design(self):
        """Opting in costs the second render - that is the documented trade."""
        sim = _sim()
        try:
            counts = _instrument(sim)

            def pixel_predicate(observation):
                return False

            pixel_predicate.requires_images = True
            policy = _ImagePolicy(sim.robot_action_keys("so101"))

            _evaluate(sim, policy, pixel_predicate, max_steps=5)

            assert counts["with_images"] == 10, counts
            assert counts["skip_images"] == 0, counts
        finally:
            sim.destroy()


class TestSuccessDetectionIsUnchanged:
    def test_a_predicate_that_fires_still_ends_the_episode_successfully(self):
        """Throughput fix only: no success-rate change."""
        sim = _sim()
        calls = {"n": 0}

        def succeeds_after_two(observation):
            calls["n"] += 1
            return calls["n"] >= 2

        try:
            policy = _ImagePolicy(sim.robot_action_keys("so101"))

            result = _evaluate(sim, policy, succeeds_after_two, max_steps=20)

            payload = next((block["json"] for block in result["content"] if "json" in block), {})
            assert payload.get("success_rate") == pytest.approx(1.0)
            assert payload.get("success_measured") is True
            assert calls["n"] == 2, f"the episode did not stop on success ({calls['n']} calls)"
        finally:
            sim.destroy()

    def test_a_predicate_that_never_fires_reports_zero_but_measured(self):
        sim = _sim()
        try:
            policy = _ImagePolicy(sim.robot_action_keys("so101"))

            result = _evaluate(sim, policy, lambda obs: False, max_steps=5)

            payload = next((block["json"] for block in result["content"] if "json" in block), {})
            assert payload.get("success_rate") == pytest.approx(0.0)
            assert payload.get("success_measured") is True
        finally:
            sim.destroy()

    def test_the_predicate_still_receives_robot_state(self):
        """Image-free does not mean empty: joint keys must still be there."""
        sim = _sim()
        seen: list[list[str]] = []

        def state_predicate(observation):
            seen.append(sorted(observation))
            return False

        try:
            policy = _ImagePolicy(sim.robot_action_keys("so101"))

            _evaluate(sim, policy, state_predicate, max_steps=3)

            assert seen, "the predicate was never called"
            assert len(seen[0]) > 0, "the predicate got an empty observation"
        finally:
            sim.destroy()
