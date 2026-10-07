"""Every backend's ``list_objects`` returns a structured listing next to its text.

``list_bodies`` and Isaac's ``list_objects`` answer with a text block for the
agent and a ``{"json": ...}`` block for code. MuJoCo and Newton answered
``list_objects`` with text only, so a notebook or a recorder confirming that
``add_object(name="red_cube", ...)`` took had to regex-parse a sentence. One
pin per backend: the text block stays first (every existing reader indexes
``content[0]["text"]``), and the json block names each object with its shape,
static flag and the same live position the text reports.

MuJoCo runs a real world. Newton and Isaac run their unbound methods against a
small stand-in for ``self`` so the pin needs neither warp nor Isaac Sim.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.simulation.models import SimObject


def _mujoco() -> tuple[dict[str, Any], dict[str, Any]]:
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    sim = Simulation(tool_name="list_objects_json", mesh=False)
    try:
        sim.create_world(ground_plane=True)
        empty = sim.list_objects()
        sim.add_object(name="red_cube", shape="box", size=[0.05, 0.05, 0.05], position=[0.0, -0.2, 0.025])
        sim.add_object(name="shelf", shape="box", size=[0.2, 0.2, 0.02], position=[0.4, 0.0, 0.01], is_static=True)
        return empty, sim.list_objects()
    finally:
        sim.cleanup(policy_stop_timeout=0.5)


def _newton() -> tuple[dict[str, Any], dict[str, Any]]:
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    world = SimpleNamespace(objects={})
    stub = SimpleNamespace(
        _world=world,
        _model=object(),
        _lock=threading.Lock(),
        _object_body_map={"red_cube": 0},
        _live_body_position=lambda index: [0.0, -0.2, 0.025],
    )
    empty = NewtonSimEngine.list_objects(stub)  # type: ignore[arg-type]
    world.objects = {
        "red_cube": SimObject(name="red_cube", shape="box", position=[0.0, -0.2, 0.3], mass=0.1),
        "shelf": SimObject(name="shelf", shape="box", position=[0.4, 0.0, 0.01], is_static=True),
    }
    return empty, NewtonSimEngine.list_objects(stub)  # type: ignore[arg-type]


def _isaac() -> tuple[dict[str, Any], dict[str, Any]]:
    from strands_robots.simulation.isaac.simulation import IsaacSimulation

    poses = {"red_cube": [0.0, -0.2, 0.025], "shelf": [0.4, 0.0, 0.01]}
    stub = SimpleNamespace(
        _world_created=True,
        _lock=threading.Lock(),
        _objects={},
        get_body_state=lambda body_name: {"status": "success", "content": [{"json": {"position": poses[body_name]}}]},
    )
    empty = IsaacSimulation.list_objects(stub)  # type: ignore[arg-type]
    stub._objects = {
        "red_cube": SimpleNamespace(shape="box", is_static=False, prim_path="/World/red_cube"),
        "shelf": SimpleNamespace(shape="box", is_static=True, prim_path="/World/shelf"),
    }
    return empty, IsaacSimulation.list_objects(stub)  # type: ignore[arg-type]


@pytest.mark.parametrize("listing", [_mujoco, _newton, _isaac], ids=["mujoco", "newton", "isaac"])
def test_list_objects_returns_text_then_a_json_listing(listing) -> None:
    empty, full = listing()

    assert empty["status"] == "success"
    assert [list(block) for block in empty["content"]] == [["text"], ["json"]]
    assert empty["content"][1]["json"] == {"objects": {}}

    assert full["status"] == "success", full
    text, payload = full["content"][0]["text"], full["content"][1]["json"]["objects"]
    assert set(payload) == {"red_cube", "shelf"}
    assert (payload["red_cube"]["shape"], payload["red_cube"]["is_static"]) == ("box", False)
    assert payload["shelf"]["is_static"] is True
    for name, entry in payload.items():
        assert name in text
        assert len(entry["position"]) == 3
        assert all(f"{v:.3f}" in text or str(v) in text for v in entry["position"]), (name, entry, text)
