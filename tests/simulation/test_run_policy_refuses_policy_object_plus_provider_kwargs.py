# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""``policy_object`` with ``policy_provider`` / ``policy_config`` is refused.

The pre-built-policy and the resolver paths of ``run_policy`` / ``start_policy``
/ ``eval_policy`` / ``evaluate_benchmark`` are the two call shapes the surface
documents, and the ``if policy_object is None:`` branch in base.py routes the
rollout into one or the other. Passing BOTH takes the pre-built branch and
silently discards the two provider kwargs the OTHER branch would have used -
no warning, no field in the result json, nothing a caller who edited a prior
provider-only call by inserting ``policy_object=`` can read.

The three symptoms this pins:

* ``run_policy(policy_object=p, policy_provider="###NOT_A_REAL_PROVIDER###",
  policy_config={"pretrained_name_or_path": "DOES_NOT_EXIST"}, ...)`` ran the
  pre-built MockPolicy to ``status="success"``, with no sign of the discarded
  provider/config in the json. A bogus provider the SAME call rejects on the
  provider-only path was accepted here.
* ``start_policy(policy_object=p, policy_provider="BOGUS", ...)`` returned
  ``status="success"`` / "Policy started" for the same combination - its own
  contract pins its refusals to ``run_policy``, so a surface silent on
  run_policy is silent on start_policy too.
* ``eval_policy(policy_object=p, policy_provider="BOGUS", ...)`` is the eval-
  path counterpart (code shape identical: ``if policy_object is None:`` branch
  at base.py's eval_policy entry).

The guard is a mutex: ``policy_object`` AND ``policy_provider!="mock"`` (the
default sentinel) OR ``policy_config is not None`` is refused up front, with
a message naming the two kwargs and the "pass EITHER / OR" remedy. The two
unambiguous shapes (bare ``policy_object=p`` with defaults; or
``policy_provider=/policy_config=`` with no object) are NOT refused.
"""
from __future__ import annotations

import pytest

from strands_robots import Robot
from strands_robots.policies.mock import MockPolicy


@pytest.fixture
def sim():
    engine = Robot("so101", mesh=False)
    yield engine
    engine.destroy()


@pytest.fixture
def mock():
    return MockPolicy()


# ---------- run_policy --------------------------------------------------------


def test_run_policy_refuses_policy_object_plus_policy_provider(sim, mock):
    """Explicit non-default policy_provider beside policy_object is refused."""
    result = sim.run_policy(
        robot_name="so101",
        policy_object=mock,
        policy_provider="lerobot_local",
        instruction="noop",
        n_steps=2,
        control_frequency=30.0,
    )
    assert result["status"] == "error"
    msg = result["content"][0]["text"]
    assert "run_policy" in msg
    assert "policy_object" in msg
    assert "policy_provider" in msg
    assert "'lerobot_local'" in msg
    # the remedy
    assert "EITHER" in msg and "OR" in msg


def test_run_policy_refuses_policy_object_plus_policy_config(sim, mock):
    """Explicit policy_config beside policy_object is refused."""
    result = sim.run_policy(
        robot_name="so101",
        policy_object=mock,
        policy_config={"host": "127.0.0.1"},
        instruction="noop",
        n_steps=2,
        control_frequency=30.0,
    )
    assert result["status"] == "error"
    msg = result["content"][0]["text"]
    assert "policy_object" in msg and "policy_config" in msg


def test_run_policy_refuses_policy_object_plus_bogus_provider_and_config(sim, mock):
    """Both kwargs set beside policy_object: message names BOTH."""
    result = sim.run_policy(
        robot_name="so101",
        policy_object=mock,
        policy_provider="###NOT_A_REAL_PROVIDER###",
        policy_config={"pretrained_name_or_path": "DOES_NOT_EXIST"},
        instruction="noop",
        n_steps=2,
        control_frequency=30.0,
    )
    assert result["status"] == "error"
    msg = result["content"][0]["text"]
    assert "policy_provider" in msg and "policy_config" in msg


def test_run_policy_accepts_bare_policy_object(sim, mock):
    """policy_object alone (both provider/config defaulted) still runs."""
    result = sim.run_policy(
        robot_name="so101",
        policy_object=mock,
        instruction="noop",
        n_steps=2,
        control_frequency=30.0,
    )
    assert result["status"] == "success"


def test_run_policy_accepts_provider_only_path(sim):
    """No policy_object + provider/config is unaffected by the guard."""
    result = sim.run_policy(
        robot_name="so101",
        policy_provider="mock",
        policy_config={},
        instruction="noop",
        n_steps=2,
        control_frequency=30.0,
    )
    assert result["status"] == "success"


# ---------- start_policy ------------------------------------------------------


def test_start_policy_refuses_policy_object_plus_policy_provider(sim, mock):
    """Same mutex as run_policy, pinned on the async entry point."""
    result = sim.start_policy(
        robot_name="so101",
        policy_object=mock,
        policy_provider="kimodo",
        instruction="noop",
        control_frequency=30.0,
    )
    assert result["status"] == "error"
    msg = result["content"][0]["text"]
    assert "start_policy" in msg
    assert "policy_object" in msg and "policy_provider" in msg


def test_start_policy_refuses_policy_object_plus_policy_config(sim, mock):
    """policy_config beside policy_object on the async entry point."""
    result = sim.start_policy(
        robot_name="so101",
        policy_object=mock,
        policy_config={"host": "127.0.0.1"},
        instruction="noop",
        control_frequency=30.0,
    )
    assert result["status"] == "error"
    msg = result["content"][0]["text"]
    assert "start_policy" in msg
    assert "policy_object" in msg and "policy_config" in msg


# ---------- eval_policy -------------------------------------------------------


def test_eval_policy_refuses_policy_object_plus_policy_provider(sim, mock):
    """eval_policy shares the same silent-wrong code shape as run_policy."""
    result = sim.eval_policy(
        robot_name="so101",
        policy_object=mock,
        policy_provider="protomotions",
        instruction="noop",
        max_steps=2,
        control_frequency=30.0,
    )
    assert result["status"] == "error"
    msg = result["content"][0]["text"]
    assert "eval_policy" in msg
    assert "policy_object" in msg and "policy_provider" in msg


# ---------- message surface ---------------------------------------------------


def test_mutex_message_is_text_only_envelope(sim, mock):
    """Pre-fix the silent-wrong carried a json block (rollout report); the
    refusal must be a text-only error envelope."""
    result = sim.run_policy(
        robot_name="so101",
        policy_object=mock,
        policy_provider="lerobot_local",
        instruction="noop",
        n_steps=2,
        control_frequency=30.0,
    )
    assert result["status"] == "error"
    assert all("json" not in b for b in result["content"])
