"""``MicroduckPolicy.reset`` restores the command instead of orphaning it.

``reset`` clears two pieces of per-episode state. ``get_actions`` rebuilds one
of them lazily (``last_action``, zeroed on the next tick) and has no such
rebuild for the other: the command is built once inside ``_ensure_config``,
which early-returns for the rest of the policy's life once ``_configured`` is
set. So on an already-configured policy the sequence *get_actions -> reset ->
get_actions* left the next tick with no command at all.

That sequence is the ordinary one, not an edge case. ``PolicyRunner`` forwards
``policy.reset(seed=...)`` before a seeded rollout and again per episode in a
multi-episode eval, so reusing one policy across two seeded rollouts - or
reaching episode 2 of a seeded eval - produced no valid action. The runner
wraps that ``reset`` in a best-effort ``try/except``, and the ``reset`` itself
succeeded, so the failure landed one call later with an empty-message
``AssertionError`` naming neither the policy nor the cause. Under ``python -O``
it did not raise at all: the command block collapsed to a single element and
the policy was handed a 49-wide observation where the graph declares 61.

What ``reset`` should restore is decided by the path that already worked. A
``reset`` before the first ``get_actions`` leaves ``_configured`` unset, so
``_ensure_config`` runs and the next tick starts from the constructor's
command; anything else here would give ``reset`` two meanings depending on
whether the policy had run yet. Both paths now read one helper, so they cannot
drift, and the helper returns a copy because ``_apply_command_kwargs`` writes
the twist slots in place over memory ``_initial_command`` shares.
"""

from __future__ import annotations

import asyncio
import inspect

import numpy as np
import pytest

from strands_robots.policies.microduck import (
    MICRODUCK_JOINT_NAMES,
    MicroduckPolicy,
    MicroduckPolicyBundle,
)

from .test_microduck_policy import _obs_dict, _StubSession

# A non-default constructor command: a forward twist plus a yaw, in a 13-wide
# vector (twist + head_pose + body_pose, the default `command_names`). Chosen
# non-zero so "restored" is distinguishable from "zeroed".
_CTOR_COMMAND = [0.4, 0.0, 0.1] + [0.0] * 10
_DECLARED_OBS_WIDTH = 61


def _policy(**kwargs: object) -> MicroduckPolicy:
    return MicroduckPolicy(session=_StubSession(), **kwargs)  # type: ignore[arg-type]


class TestTheCommandSurvivesAReset:
    """The regression: an already-configured policy still acts after a reset."""

    def test_a_tick_after_a_reset_returns_actions(self):
        policy = _policy()
        asyncio.run(policy.get_actions(_obs_dict(), ""))
        policy.reset(seed=7)
        actions = asyncio.run(policy.get_actions(_obs_dict(), ""))
        assert set(actions[0]) == set(MICRODUCK_JOINT_NAMES)

    def test_the_reset_leaves_a_command_in_place(self):
        policy = _policy()
        asyncio.run(policy.get_actions(_obs_dict(), ""))
        policy.reset(seed=7)
        assert policy._command is not None

    def test_the_restored_command_is_the_one_the_constructor_asked_for(self):
        policy = _policy(command=_CTOR_COMMAND)
        asyncio.run(policy.get_actions(_obs_dict(), ""))
        policy.reset(seed=7)
        np.testing.assert_allclose(policy._command, _CTOR_COMMAND, atol=1e-6)

    def test_a_reset_after_running_matches_a_reset_before_running(self):
        """The two orderings agree, so ``reset`` has one meaning."""
        before = _policy(command=_CTOR_COMMAND)
        before.reset(seed=7)
        asyncio.run(before.get_actions(_obs_dict(), ""))

        after = _policy(command=_CTOR_COMMAND)
        asyncio.run(after.get_actions(_obs_dict(), ""))
        after.reset(seed=7)
        asyncio.run(after.get_actions(_obs_dict(), ""))

        assert before._command is not None and after._command is not None
        np.testing.assert_allclose(before._command, after._command, atol=1e-6)

    def test_the_observation_width_is_unchanged_across_a_reset(self):
        """The width is what a ``-O`` build loses instead of raising."""
        stub = _StubSession()
        policy = MicroduckPolicy(session=stub)
        asyncio.run(policy.get_actions(_obs_dict(), ""))
        first = stub.last_input.shape  # type: ignore[union-attr]
        policy.reset(seed=7)
        asyncio.run(policy.get_actions(_obs_dict(), ""))
        assert stub.last_input.shape == first  # type: ignore[union-attr]
        assert first[-1] == _DECLARED_OBS_WIDTH

    def test_last_action_is_rebuilt_on_the_tick_after_a_reset(self):
        """The lazy rebuild the command lacked: ``get_actions`` re-zeroes this."""
        policy = _policy()
        asyncio.run(policy.get_actions(_obs_dict(), ""))
        policy.reset(seed=7)
        asyncio.run(policy.get_actions(_obs_dict(), ""))
        assert policy._last_action is not None

    def test_a_bundle_child_that_has_run_survives_the_bundles_reset(self):
        """``MicroduckPolicyBundle.reset`` forwards to every held skill."""
        walk, stand = _policy(), _policy()
        bundle = MicroduckPolicyBundle({"walk": walk, "stand": stand}, active="walk")
        asyncio.run(bundle.get_actions(_obs_dict(), ""))
        assert walk._configured and not stand._configured
        bundle.reset(seed=7)
        for child in (walk, stand):
            assert set(asyncio.run(child.get_actions(_obs_dict(), ""))[0]) == set(MICRODUCK_JOINT_NAMES)


class TestTheRestoredCommandIsNotAliased:
    """One tick's ``target_velocity`` must not become every later episode's start."""

    def test_a_twist_kwarg_does_not_rewrite_what_a_reset_restores(self):
        policy = _policy(command=_CTOR_COMMAND)
        asyncio.run(policy.get_actions(_obs_dict(), "", target_velocity=[0.9, 0.8, 0.7]))
        policy.reset(seed=7)
        np.testing.assert_allclose(policy._command, _CTOR_COMMAND, atol=1e-6)

    def test_the_restored_command_shares_no_memory_with_the_constructors(self):
        policy = _policy(command=_CTOR_COMMAND)
        asyncio.run(policy.get_actions(_obs_dict(), ""))
        policy.reset(seed=7)
        assert policy._initial_command is not None and policy._command is not None
        assert not np.shares_memory(policy._command, policy._initial_command)


class TestTheEpisodeStartCommandHasOneOwner:
    """Structural: the first episode and every later one read one helper."""

    def test_the_command_is_built_in_exactly_one_place(self):
        source = inspect.getsource(MicroduckPolicy)
        assert source.count("def _episode_start_command") == 1
        # `_ensure_config` and `reset` are the only readers, so neither can
        # grow its own copy of the width check or the zero fallback.
        assert source.count("_episode_start_command()") == 2

    def test_ensure_config_defers_to_the_helper(self):
        assert "_episode_start_command()" in inspect.getsource(MicroduckPolicy._ensure_config)

    def test_reset_defers_to_the_helper(self):
        assert "_episode_start_command()" in inspect.getsource(MicroduckPolicy.reset)


class TestWhatIsUnchangedEitherWay:
    """Controls: these held before the fix and must go on holding."""

    def test_a_reset_before_the_first_tick_still_defers_to_ensure_config(self):
        """Unconfigured, the width is not yet known, so the build stays deferred."""
        policy = _policy(command=_CTOR_COMMAND)
        policy.reset(seed=7)
        assert policy._command is None
        asyncio.run(policy.get_actions(_obs_dict(), ""))
        np.testing.assert_allclose(policy._command, _CTOR_COMMAND, atol=1e-6)

    def test_a_reset_still_clears_last_action(self):
        policy = _policy()
        asyncio.run(policy.get_actions(_obs_dict(), ""))
        policy.reset(seed=7)
        assert policy._last_action is None

    def test_two_ticks_without_a_reset_are_unchanged(self):
        stub = _StubSession()
        policy = MicroduckPolicy(session=stub)
        asyncio.run(policy.get_actions(_obs_dict(), ""))
        asyncio.run(policy.get_actions(_obs_dict(), ""))
        fed_last_action = stub.last_input.reshape(-1)[34:48]  # type: ignore[union-attr]
        np.testing.assert_allclose(fed_last_action, np.arange(14) * 0.01, atol=1e-6)

    def test_a_mismatched_constructor_command_is_still_refused_at_configure(self):
        policy = _policy(command=[0.1, 0.2])
        with pytest.raises(ValueError, match="initial command width 2"):
            asyncio.run(policy.get_actions(_obs_dict(), ""))

    def test_a_command_kwarg_override_still_applies(self):
        policy = _policy()
        wide = [0.5] * 13
        asyncio.run(policy.get_actions(_obs_dict(), "", command=wide))
        np.testing.assert_allclose(policy._command, wide, atol=1e-6)
