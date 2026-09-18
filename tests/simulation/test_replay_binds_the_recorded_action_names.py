# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay binds a recorded action column by the names the recording wrote for it.

``PolicyRunner.replay`` maps the recorded ``action`` vector onto action keys by
index. Before this change the keys were whatever ``robot_action_keys`` returned
*today*, and the recorder writes that same list into the dataset as
``features["action"]["names"]`` at *record* time. Those two agree only while
the backend's key order is fixed. #3851 moved the MuJoCo order from actuator
declaration to joint order on eleven robots, and the width guard cannot see the
difference: a ``dynamixel_2r`` episode recorded as ``[R2, R1]`` replayed under
the new ``[R1, R2]`` with every value on the other actuator and
``status="success"``, the same failure the reorder fixed on the state side.

The recording already carries the answer, so replay reads it: when the dataset
names its action columns and those names are this robot's actuators in any
order, the vector is bound by name. ``send_action`` resolves a dict by name, so
the recorded order is the right order whatever the backend says now. A dataset
without that schema (the column-only fakes below) or one whose columns are
another roster keeps the positional path, since neither is an ordering
question and ``action_key_map`` is the explicit answer there.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.simulation.policy_runner import PolicyRunner
from tests.simulation.test_policy_runner import FakeSim


class _KeyedSim(FakeSim):
    """FakeSim whose actuator roster is declared, so the test controls the order."""

    def __init__(self, action_keys: list[str]) -> None:
        super().__init__(joint_names=tuple(action_keys))
        self._action_keys = list(action_keys)

    def robot_action_keys(self, robot_name: str) -> list[str]:
        return list(self._action_keys)


class _Recording:
    """A one-frame dataset with an optional on-disk action schema."""

    fps = 30

    def __init__(self, values: list[float], names: list[str] | None) -> None:
        self._values = list(values)
        if names is not None:
            self.meta = SimpleNamespace(features={"action": {"dtype": "float32", "names": list(names)}})

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, list[float]]:
        return {"action": list(self._values)}


def _replay(monkeypatch: pytest.MonkeyPatch, sim: FakeSim, recording: _Recording, **kw: Any) -> dict[str, Any]:
    import strands_robots.dataset_recorder as dr

    monkeypatch.setattr(dr, "load_lerobot_episode", lambda repo_id, episode, root: (recording, 0, 1), raising=False)
    return PolicyRunner(sim).replay(repo_id="fake/recording", robot_name="fake_robot", speed=1000.0, **kw)


def _sent(sim: FakeSim) -> dict[str, float]:
    sent = [dict(action) for kind, action, _ in sim.calls if kind == "send_action"]
    assert len(sent) == 1, sent
    return sent[0]


def test_a_recording_made_under_the_old_order_lands_on_the_actuators_it_named(monkeypatch):
    """Recorded ``[R2, R1]``, robot now ``[R1, R2]``: each value reaches its own actuator."""
    sim = _KeyedSim(["R1", "R2"])
    recording = _Recording(values=[-0.7, 0.3], names=["R2", "R1"])

    result = _replay(monkeypatch, sim, recording)

    assert result["status"] == "success", result
    assert _sent(sim) == {"R2": -0.7, "R1": 0.3}


def test_a_recording_in_the_current_order_is_unchanged(monkeypatch):
    sim = _KeyedSim(["R1", "R2"])
    recording = _Recording(values=[0.3, -0.7], names=["R1", "R2"])

    result = _replay(monkeypatch, sim, recording)

    assert result["status"] == "success", result
    assert _sent(sim) == {"R1": 0.3, "R2": -0.7}


def test_a_dataset_without_an_action_schema_binds_positionally(monkeypatch):
    """The column-only fakes the other replay tests use keep working."""
    sim = _KeyedSim(["R1", "R2"])
    recording = _Recording(values=[0.3, -0.7], names=None)

    result = _replay(monkeypatch, sim, recording)

    assert result["status"] == "success", result
    assert _sent(sim) == {"R1": 0.3, "R2": -0.7}


def test_another_roster_is_not_an_ordering_question(monkeypatch):
    """Columns that are not this robot's actuators keep the positional path.

    A different robot's names, or a multi-robot recording's prefixed ones, are
    what ``action_key_map`` exists for; the by-name binding only claims a
    permutation of this robot's own roster.
    """
    sim = _KeyedSim(["R1", "R2"])
    recording = _Recording(values=[0.3, -0.7], names=["alice__R1", "alice__R2"])

    result = _replay(monkeypatch, sim, recording)

    assert result["status"] == "success", result
    assert _sent(sim) == {"R1": 0.3, "R2": -0.7}


@pytest.mark.parametrize("names", [[], ["R1", 2], "R1,R2"])
def test_a_malformed_schema_is_not_trusted(monkeypatch, names):
    sim = _KeyedSim(["R1", "R2"])
    recording = _Recording(values=[0.3, -0.7], names=None)
    recording.meta = SimpleNamespace(features={"action": {"names": names}})

    result = _replay(monkeypatch, sim, recording)

    assert result["status"] == "success", result
    assert _sent(sim) == {"R1": 0.3, "R2": -0.7}


def test_an_explicit_action_key_map_still_wins(monkeypatch):
    sim = _KeyedSim(["R1", "R2"])
    recording = _Recording(values=[-0.7, 0.3], names=["R2", "R1"])

    result = _replay(monkeypatch, sim, recording, action_key_map=["R1", "R2"])

    assert result["status"] == "success", result
    assert _sent(sim) == {"R1": -0.7, "R2": 0.3}
