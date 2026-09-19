# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A demo never announces a rollout the runtime refused.

``Robot.run_policy`` reports a refusal by RETURNING ``{"status": "error", ...}``
rather than raising, so a demo that discards the result prints its completion
line and exits 0 for a rollout that applied no action. Two demos did:

* ``examples/kimodo/kimodo_g1_walking.py`` printed ``Done. Video: <path>`` for
  the default ``nvidia/Kimodo-G1-RP-v1`` checkpoint, which is refused on the
  first sample (bare weights, no ``model_index.json``) - so the advertised MP4
  was never written and the exit status still said success;
* ``examples/wbc/wbc_g1_gait.py`` printed ``gait rollout complete`` for any
  refusal, including the 516-wide non-gait ONNX the docs warn does not load.

Both are graded here on the same rule, one row each: a refused rollout raises
and prints no claim, an accepted one prints the claim. No GPU, no weights and no
simulator - the runtime is stubbed at the ``Robot`` seam each demo imports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

import strands_robots

_REFUSED: dict[str, Any] = {
    "status": "error",
    "content": [{"text": "Policy failed: not a diffusers pipeline"}],
}
_ACCEPTED: dict[str, Any] = {"status": "success", "content": [{"json": {"steps_used": 200}}]}


class _StubSim:
    """The slice of ``Robot`` the demos drive, with a scripted rollout result."""

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result
        self.calls = 0

    def add_camera(self, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    def run_policy(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return self._result


@dataclass(frozen=True)
class _Demo:
    """One example's entry point, the claim it prints, and how to drive it."""

    label: str
    claim: str
    invoke: Callable[[pytest.MonkeyPatch], None]


def _run_kimodo_walking(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.kimodo import kimodo_g1_walking

    monkeypatch.setattr("sys.argv", ["kimodo_g1_walking.py", "--out", "/tmp/unwritten.mp4"])
    kimodo_g1_walking.main()


def _run_wbc_gait(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.wbc import wbc_g1_gait
    from strands_robots.policies import wbc

    monkeypatch.setattr(wbc, "WBCGaitPolicy", lambda **_kwargs: object())
    wbc_g1_gait.run_policy("/nonexistent/gait-g1", vx=0.5, freq=1.0, duration=1.0, mp4=None)


DEMOS = [
    _Demo("kimodo_g1_walking", "Done. Video:", _run_kimodo_walking),
    _Demo("wbc_g1_gait", "gait rollout complete", _run_wbc_gait),
]


@pytest.fixture
def stub_robot(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, Any]], _StubSim]:
    """Install a stub at the ``Robot`` seam the demos import at call time."""

    def install(result: dict[str, Any]) -> _StubSim:
        sim = _StubSim(result)
        monkeypatch.setattr(strands_robots, "Robot", lambda *_a, **_k: sim)
        return sim

    return install


@pytest.mark.parametrize("demo", DEMOS, ids=lambda d: d.label)
def test_a_refused_rollout_is_not_announced_as_done(demo, stub_robot, monkeypatch, capsys):
    """A refusal raises, and the demo's completion line is never printed."""
    sim = stub_robot(_REFUSED)

    with pytest.raises(RuntimeError, match="not a diffusers pipeline"):
        demo.invoke(monkeypatch)

    assert sim.calls == 1
    assert demo.claim not in capsys.readouterr().out


@pytest.mark.parametrize("demo", DEMOS, ids=lambda d: d.label)
def test_an_accepted_rollout_still_reports_done(demo, stub_robot, monkeypatch, capsys):
    """The guard reads the status only - a successful rollout is unchanged."""
    sim = stub_robot(_ACCEPTED)

    demo.invoke(monkeypatch)

    assert sim.calls == 1
    assert demo.claim in capsys.readouterr().out
