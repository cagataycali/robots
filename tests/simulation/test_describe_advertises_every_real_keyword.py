"""Every keyword a described method really takes is in its advertised signature.

The describe() grader in ``test_sim_engine_describe_discovery.py`` pins the one
direction: nothing advertised may be stale. This pins the other: a keyword the
bound method accepts is discoverable from the advert, unless the advert elides
its tail with ``...`` on purpose (the four rollout verbs, whose full keyword
lists live in their docstrings). ``add_robot`` hid ``keyframe`` and ``render``
hid ``output_path`` on MuJoCo (#4154); an agent reading describe() could not
find either.
"""

from __future__ import annotations

import inspect
import os

import pytest

mj = pytest.importorskip("mujoco")

from tests.simulation.test_sim_engine_describe_discovery import _advertised_param_names  # noqa: E402


def _unadvertised_real_keywords(engine) -> list[str]:
    missing: list[str] = []
    for name, sig_str in engine.describe()["methods"].items():
        fn = getattr(engine, name, None)
        if not callable(fn):
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            continue
        head = sig_str.split("->", 1)[0]
        if "..." in head:
            continue  # the advert says it is abridged
        advertised = set(_advertised_param_names(sig_str))
        missing.extend(f"{name}(...{p}=...)" for p in params if p != "self" and p not in advertised)
    return missing


def test_mujoco_describe_advertises_every_real_keyword() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    try:
        sim.create_world()
        sim.add_robot("so100", data_config="so100")
        missing = _unadvertised_real_keywords(sim)
    finally:
        sim.destroy()
    assert missing == [], (
        "describe() hides keywords the real method accepts; an agent reading the advert "
        f"cannot discover them: {sorted(missing)}. Add them to the advert string, or end the "
        "advert's parameter list with '...' when it is abridged on purpose."
    )


def test_the_two_hidden_keywords_are_named() -> None:
    """The specific adverts #4154 named, pinned by text so the walk above has a witness."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from strands_robots.simulation import Simulation

    sim = Simulation()
    try:
        methods = sim.describe()["methods"]
    finally:
        sim.destroy()
    assert "keyframe=None" in methods["add_robot"]
    assert "output_path=None" in methods["render"]
