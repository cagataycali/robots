"""Isaac's colour randomization says it leaves the robot alone.

MuJoCo's ``randomize_colors`` recolours "every non-ground geom", the robot's
included (docs/learn/simulation/randomization.md); Isaac's recolours registered
objects only, and reported ``Colors: 1 object(s) resampled`` as if that were the
whole scene. It cannot do more safely: converted robot visuals are USD instance
proxies, and de-instancing them mid-simulation invalidated the PhysX tensor view
(measured on 6.1). So the envelope names the robots it skipped.
"""

from __future__ import annotations

import types

from strands_robots.simulation.isaac import randomization as rnd_module

from .test_randomize_and_obs_noise import _engine, _stage_free  # noqa: F401 - autouse fixture: stage writes inert


def _run(robots: dict) -> dict:
    engine = _engine()
    engine._robots = robots
    return engine.randomize(randomize_colors=True, randomize_lighting=False, seed=0)


def test_the_report_names_the_skipped_robot() -> None:
    result = _run({"so100": types.SimpleNamespace()})
    assert result["status"] == "success", result
    text = " ".join(b.get("text", "") for b in result["content"])
    assert "robot visuals are not recoloured on Isaac: so100" in text
    payload = next(b["json"] for b in result["content"] if "json" in b)
    assert payload["robots_not_recoloured"] == ["so100"]


def test_no_robot_no_caveat() -> None:
    result = _run({})
    text = " ".join(b.get("text", "") for b in result["content"])
    assert "not recoloured" not in text


def test_the_module_documents_the_scope() -> None:
    assert "ROBOT is not recoloured" in (rnd_module.__doc__ or "")
