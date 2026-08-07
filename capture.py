"""Capture what the VERA planner actually receives for each render_width."""

from __future__ import annotations

import asyncio
import json
import math
import pathlib
import sys
from typing import Any

import numpy as np

import strands_robots.policies.vera.provider as _P

TREE = str(pathlib.Path(_P.__file__).parents[3])
print("TREE:", TREE)

from strands_robots.policies.vera.config import VeraConfig  # noqa: E402
from strands_robots.policies.vera.provider import VeraPolicy  # noqa: E402

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


# ---- a real MuJoCo headless render as the camera observation ----------------
def sim_frame() -> np.ndarray:
    from strands_robots import Robot

    sim = Robot("panda", mode="sim", mesh=False)
    sim.create_world()
    sim.add_robot(name="panda", data_config="panda")
    sim.add_object(name="crate", shape="box", position=[0.55, 0.0, 0.06], size=[0.12, 0.12, 0.12],
                   color=[0.95, 0.45, 0.10, 1.0], mass=0.5)
    sim.add_object(name="ball", shape="sphere", position=[0.35, 0.22, 0.05], size=[0.10, 0.10, 0.10],
                   color=[0.15, 0.55, 0.90, 1.0], mass=0.3)
    sim.add_camera(name="wrist", position=[0.95, -0.55, 0.55], target=[0.45, 0.0, 0.10], fov=42)
    sim.step(120)
    r = sim.render(camera_name="wrist", width=480, height=480)
    png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    import io

    from PIL import Image

    arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.uint8)
    sim.cleanup()
    return arr


class _FakeClient:
    def __init__(self) -> None:
        self.infer_requests: list[dict] = []

    def get_server_metadata(self) -> dict:
        return {"action_space": "pos", "context_frames": 1}

    def infer(self, observation: dict) -> dict:
        self.infer_requests.append(observation)
        return {"action": np.zeros((1, 1), dtype=np.float32)}

    def reset(self, info: Any = None) -> None:
        pass

    def configure(self, params: dict) -> dict:
        return {"applied": params}

    def close(self) -> None:
        pass


CASES: list[tuple[str, Any]] = [
    ("128", 128),
    ("0", 0),
    ("2.7", 2.7),
    ("True", True),
]

frame = sim_frame()
np.save(OUT / "sim_frame.npy", frame)
facts: dict[str, Any] = {"tree": TREE, "sim_frame_shape": list(frame.shape), "cases": {}}

for label, value in CASES:
    rec: dict[str, Any] = {"requested": repr(value)}
    try:
        cfg = VeraConfig(embodiment="mimicgen", render_width=value, auto_launch_server=False)
        rec["config"] = "accepted"
        rec["stored"] = repr(cfg.render_width)
        client: Any = _FakeClient()
        pol = VeraPolicy(client=client, config=cfg)
        asyncio.run(pol.get_actions({"cam0": frame}, ""))
        wire = client.infer_requests[-1]
        ctx = np.asarray(wire["context_rgb"])
        view = ctx[-1]
        rec["result"] = "success"
        rec["view_widths"] = [int(w) for w in wire["view_widths"]]
        rec["view_shape"] = [int(v) for v in view.shape]
        np.save(OUT / f"view_{label}.npy", view)
    except Exception as exc:  # noqa: BLE001 - the outcome is the measurement
        rec.setdefault("config", "refused" if isinstance(exc, ValueError) and "render_width" in str(exc) else "accepted")
        rec["result"] = f"{type(exc).__name__}: {exc}"
    facts["cases"][label] = rec
    print(label, "->", json.dumps(rec)[:220])

(OUT / "facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
print("wrote", OUT / "facts.json")
_ = math
