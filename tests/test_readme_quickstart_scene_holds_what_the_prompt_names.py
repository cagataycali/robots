"""The README quickstart builds a scene that holds what its prompt asks for."""

import re
from pathlib import Path

import pytest
import strands

README = Path(__file__).resolve().parents[1] / "README.md"


def test_the_quickstart_scene_holds_every_object_the_prompt_names(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mujoco")
    snippet = re.search(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert snippet, "README has no python quickstart block"
    asked: list[tuple[list, str]] = []

    class RecordingAgent:
        def __init__(self, tools: list) -> None:
            self.tools = tools

        def __call__(self, prompt: str) -> None:
            asked.append((self.tools, prompt))

    monkeypatch.setattr(strands, "Agent", RecordingAgent)
    exec(compile(snippet.group(1), str(README), "exec"), {})

    [(tools, prompt)] = asked
    objects = set(tools[0]._world.objects)
    named = {"_".join(m.split()) for m in re.findall(r"the ([a-z]+ [a-z]+)$", prompt)}
    assert named, f"no object named in {prompt!r}"
    assert named <= objects, f"prompt {prompt!r} names {named}, scene holds {objects or 'nothing'}"

    # The front camera's line of sight to each named object is not blocked by the arm.
    import mujoco
    import numpy as np

    world = tools[0]._world
    camera = world.cameras["front"]
    for name in named:
        eye = np.asarray(camera.position, dtype=float)
        ray = np.asarray(world.objects[name].position, dtype=float) - eye
        hit = np.array([-1], dtype=np.int32)
        mujoco.mj_ray(world._model, world._data, eye, ray, None, 1, -1, hit)
        seen = mujoco.mj_id2name(world._model, mujoco.mjtObj.mjOBJ_GEOM, int(hit[0])) or ""
        assert name in seen, f"camera 'front' looking at {name!r} first hits geom #{int(hit[0])} {seen!r}"
