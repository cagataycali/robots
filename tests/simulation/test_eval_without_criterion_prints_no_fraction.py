"""eval_policy with no success criterion does not print a success fraction.

An agent asked for a baseline read ``Success: 0/3 (0.0%) [no success criterion
- not measured]`` as a 0% success rate and reported every episode as failed.
Nothing measured anything, so the text says exactly that and prints no
fraction; the json keeps its documented ``success_rate: 0.0`` +
``success_measured: false`` pair.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="t", mesh=False)
    s.create_world()
    s.add_robot("so101")
    yield s
    s.cleanup()


def _text(result):
    return next(c["text"] for c in result["content"] if "text" in c)


def _json(result):
    return next(c["json"] for c in result["content"] if "json" in c)


def test_no_criterion_prints_not_measured_and_no_fraction(sim):
    result = sim.eval_policy(robot_name="so101", policy_provider="mock", n_episodes=2, max_steps=3)
    assert result["status"] == "success"
    text = _text(result)
    assert "Success: not measured" in text
    assert "0/2" not in text and "0.0%" not in text, "a fraction nobody measured reads as a 0% baseline"
    assert "success_fn" in text and "benchmark" in text, "the remedy names both ways to measure"
    payload = _json(result)
    assert payload["success_measured"] is False
    assert payload["success_rate"] == 0.0, "the json contract is unchanged"


def test_criterion_keeps_the_fraction(sim):
    result = sim.eval_policy(
        robot_name="so101", policy_provider="mock", n_episodes=2, max_steps=3, success_fn="contact"
    )
    text = _text(result)
    assert "Success: " in text and "/2 (" in text
    assert "not measured" not in text
    assert _json(result)["success_measured"] is True
