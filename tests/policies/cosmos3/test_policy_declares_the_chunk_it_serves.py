# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``Cosmos3Policy`` declares the chunk it serves, so the runner keeps the whole chunk and can mask its latency.

Measured on main 9e4f0a3d0: a Cosmos 3 inference returned a 16-step chunk
against a wire-faithful RoboLab server (32 by the DROID embodiment's
``action_chunk_size``), yet the policy declared ``execution_horizon == 1`` and
``is_chunk_emitting() == False``, because it never set ``actions_per_step``.
The runner therefore consumed ``max(action_horizon, 1)`` actions per chunk (8 by
default, dropping the tail) and ``run_policy(async_rtc=None)`` never enabled
async RTC for a diffusion policy that pays hundreds of milliseconds per chunk.
(The GR00T half of the finding left with the provider in #4256.)

Now the constructor declares ``actions_per_step`` (the embodiment default, or a
value the caller pins, on the shared ``chunk_count_error`` domain), and the
served chunk length replaces the default the first time the server answers
unless the caller pinned one. The rollout surface asks nothing new: it reads
``execution_horizon`` through ``resolve_chunk_length`` after every call.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.base import resolve_chunk_length
from strands_robots.policies.cosmos3 import Cosmos3Policy
from strands_robots.policies.cosmos3.embodiments import get_embodiment
from tests.policies.cosmos3.test_policy import FakeClient, _droid_chunk, _obs_with_state_and_images

KEYS = [f"joint_{i}" for i in range(7)] + ["gripper"]


def _client(action: np.ndarray) -> Any:
    """The fake client, typed as the duck the constructor quacks at."""
    return FakeClient(action)


def _policy(served: int, **kwargs: Any) -> Cosmos3Policy:
    policy = Cosmos3Policy(embodiment="droid", client=_client(_droid_chunk(served, 8)), **kwargs)
    policy.set_robot_state_keys(KEYS)
    return policy


def _infer(policy: Cosmos3Policy) -> list[dict[str, Any]]:
    return asyncio.run(policy.get_actions(_obs_with_state_and_images(), "pick up the cube"))


class TestTheDeclarationBeforeTheFirstAnswer:
    def test_the_embodiment_default_is_the_declared_chunk(self) -> None:
        policy = _policy(32)
        assert policy.actions_per_step == get_embodiment("droid").action_chunk_size == 32
        assert policy.execution_horizon == 32
        assert policy.is_chunk_emitting() is True

    def test_every_embodiment_declares_its_own_default(self) -> None:
        for name in ("droid", "av"):
            emb = get_embodiment(name)
            policy = Cosmos3Policy(embodiment=name, client=_client(_droid_chunk(emb.action_chunk_size, 9)))
            assert policy.execution_horizon == emb.action_chunk_size

    def test_a_pinned_value_is_kept(self) -> None:
        policy = _policy(32, actions_per_step=8)
        assert policy.execution_horizon == 8

    @pytest.mark.parametrize("value", [0, -1, 2.5, True, "8", float("nan")], ids=repr)
    def test_an_unusable_pin_is_refused_on_the_shared_domain(self, value: Any) -> None:
        with pytest.raises(ValueError, match="actions_per_step"):
            Cosmos3Policy(embodiment="droid", client=_client(_droid_chunk()), actions_per_step=value)


class TestTheServedChunkSetsTheInterval:
    def test_a_shorter_served_chunk_is_adopted(self, caplog: pytest.LogCaptureFixture) -> None:
        """The measured case: the server serves 16 where the embodiment says 32."""
        policy = _policy(16)
        with caplog.at_level(logging.INFO, logger="strands_robots.policies.cosmos3.policy"):
            out = _infer(policy)
        assert len(out) == 16
        assert policy.execution_horizon == 16
        assert policy.is_chunk_emitting() is True
        assert any("serves 16-step chunks" in rec.getMessage() for rec in caplog.records)

    def test_the_runner_keeps_the_whole_served_chunk(self) -> None:
        """``resolve_chunk_length`` after the call reads the adopted interval, so no tail is dropped."""
        policy = _policy(16)
        out = _infer(policy)
        assert len(out[: resolve_chunk_length(policy, action_horizon=8)]) == 16

    def test_a_pinned_value_survives_a_different_served_chunk(self, caplog: pytest.LogCaptureFixture) -> None:
        policy = _policy(16, actions_per_step=32)
        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.cosmos3.policy"):
            _infer(policy)
            _infer(policy)
        assert policy.execution_horizon == 32
        warnings = [
            rec for rec in caplog.records if "was pinned but the server serves 16-step chunks" in rec.getMessage()
        ]
        assert len(warnings) == 1, "the pinned-versus-served notice is said once"

    def test_a_pin_below_the_served_chunk_is_honoured_quietly(self, caplog: pytest.LogCaptureFixture) -> None:
        policy = _policy(32, actions_per_step=8)
        with caplog.at_level(logging.INFO, logger="strands_robots.policies.cosmos3.policy"):
            out = _infer(policy)
        assert len(out) == 32
        assert policy.execution_horizon == 8
        assert len(out[: resolve_chunk_length(policy, action_horizon=1)]) == 8
        assert not any("pinned" in rec.getMessage() for rec in caplog.records)

    def test_an_equal_served_chunk_says_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        policy = _policy(32)
        with caplog.at_level(logging.INFO, logger="strands_robots.policies.cosmos3.policy"):
            _infer(policy)
        assert policy.execution_horizon == 32
        assert not any("chunk" in rec.getMessage() for rec in caplog.records)

    def test_a_single_row_answer_is_taken_as_served(self) -> None:
        """A 1-D action is one step; the interval follows it like any other served length."""
        policy = Cosmos3Policy(embodiment="droid", client=_client(np.zeros(8, dtype=np.float32)))
        policy.set_robot_state_keys(KEYS)
        _infer(policy)
        assert policy.execution_horizon == 1
        assert policy.is_chunk_emitting() is False
