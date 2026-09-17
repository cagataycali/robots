"""``eval_policy`` scores episodes by a predicate clause via ``success_when``.

Before, the only success criterion an agent-tool call could express was
``success_fn='contact'``; a predicate spelled as a string
(``'base_beyond_x:0.5'``) was refused with ``Unknown success_fn string`` and
no word on what IS accepted, and the DSL every neighbouring surface speaks
(``run_policy``'s ``stop_when``, a benchmark's ``success`` clause) was not
available here at all.

Pinned: ``success_when`` takes that DSL; it is compiled through the closed
registry (bad shape refused with the shape), probed against the live scene
(missing body refused up front; a base clause on a fixed-base arm refused before
the first episode), evaluated per step (a clause that never holds
scores 0/N with success_measured true, one that holds scores 1/N), and is an
alternative to ``success_fn`` (both -> refused). The unknown-string refusal
names ``'contact'`` and points at ``success_when``. The tool spec publishes
``success_when`` as an object.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation


@pytest.fixture
def sim():
    s = Simulation(tool_name="eval_success_when", mesh=False)
    s.create_world()
    assert s.add_robot(name="so101", data_config="so101")["status"] == "success"
    assert s.add_robot(name="go2", data_config="unitree_go2", position=[1.0, 0.0, 0.0])["status"] == "success"
    yield s
    s.cleanup(policy_stop_timeout=0.5)


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _json(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


def _eval(sim, **kw):
    kw.setdefault("policy_provider", "mock")
    kw.setdefault("n_episodes", 1)
    kw.setdefault("max_steps", 10)
    kw.setdefault("control_frequency", 50.0)
    return sim.eval_policy(**kw)


def test_unknown_success_fn_string_names_contact_and_success_when(sim):
    r = _eval(sim, robot_name="go2", success_fn="base_beyond_x:0.5")
    assert r["status"] == "error"
    text = _text(r)
    assert "'contact'" in text and "success_when" in text, text


def test_success_when_that_holds_scores_the_episode_and_flags_true_at_reset(sim):
    r = _eval(sim, robot_name="go2", success_when={"predicate": "base_beyond_x", "x": 0.5})
    assert r["status"] == "success", _text(r)
    payload = _json(r)
    assert payload["success_rate"] == 1.0
    assert payload["success_measured"] is True
    assert "at reset" in _text(r)  # the go2 starts at x=1.0, so the clause already held


def test_success_when_that_never_holds_scores_zero_with_success_measured(sim):
    r = _eval(sim, robot_name="go2", success_when={"predicate": "base_beyond_x", "x": 5.0})
    assert r["status"] == "success", _text(r)
    payload = _json(r)
    assert payload["success_rate"] == 0.0
    assert payload["success_measured"] is True


def test_success_when_missing_body_is_refused_before_any_episode(sim):
    r = _eval(sim, robot_name="go2", success_when={"predicate": "body_above_z", "body": "nope", "z": 0.1})
    assert r["status"] == "error"
    text = _text(r)
    assert text.startswith("eval_policy: success_when references bodies not present"), text
    assert "['nope']" in text


def test_success_when_base_clause_on_a_fixed_base_arm_is_refused_before_any_episode(sim):
    """A base clause the robot under evaluation cannot satisfy is refused up front.

    Pinned on what ``success_when`` owns - that the refusal is attributed to
    ``success_when``, states the floating-base cause, and names the 0% this
    surface would otherwise have reported - not on how the robot is spelled in
    it. Which of the placeholder spellings speaks (``<the sole robot>``, the
    bare repr, or the "robot under evaluation" label) is settled by the
    open PRs that own that line, so asserting one here would make this cell a
    merge-order tripwire rather than a pin on this feature.
    """
    r = _eval(sim, robot_name="so101", success_when={"predicate": "base_beyond_x", "x": 0.5})
    assert r["status"] == "error"
    text = _text(r)
    assert text.startswith("eval_policy: success_when arms a base_* predicate"), text
    assert "has no floating base" in text
    # The consequence is stated in this surface's own terms - the success rate.
    assert "0% success rate" in text, text
    # Refused before the first episode: no results envelope was produced.
    assert not any("json" in c for c in r["content"] if isinstance(c, dict)), r["content"]


def test_success_when_bad_shape_is_refused_with_the_shape(sim):
    r = _eval(sim, robot_name="go2", success_when={"base_beyond_x": 1})
    assert r["status"] == "error"
    assert "success_when: unknown keys ['base_beyond_x']" in _text(r)


def test_success_fn_and_success_when_together_are_refused(sim):
    r = _eval(sim, robot_name="go2", success_fn="contact", success_when={"predicate": "base_beyond_x", "x": 0.5})
    assert r["status"] == "error"
    assert "not both" in _text(r)


def test_tool_spec_publishes_success_when_as_an_object_and_success_fn_names_contact():
    spec = json.loads((Path(Simulation.__module__.replace(".", "/")).parent / "tool_spec.json").read_text())
    props = spec["inputSchema"]["json"]["properties"] if "inputSchema" in spec else spec["properties"]
    assert props["success_when"]["type"] == "object"
    assert "stop_when" in props["success_when"]["description"]
    assert "'contact'" in props["success_fn"]["description"]
    assert "success_when" in props["success_fn"]["description"]
