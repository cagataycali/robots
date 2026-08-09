"""Capture the on_frame watchdog's behaviour for the artifact, on whichever tree runs it."""
from __future__ import annotations

import io
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import strands_robots.simulation.policy_runner as pr_mod

TREE = str(Path(pr_mod.__file__).parents[2])
print("TREE:", TREE, flush=True)

from PIL import Image
from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.policy_runner import PolicyRunner

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
W, H = 620, 560
CAM = dict(position=[0.62, -0.50, 0.40], target=[0.0, 0.0, 0.16], fov=42)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.emitted = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            record.getMessage()
            self.emitted += 1
        except (ValueError, OverflowError):
            pass  # what logging reports as "--- Logging error ---"


def _render(sim) -> np.ndarray:
    r = sim.render(camera_name="look", width=W, height=H)
    png = next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def run(tag: str, limit, broken: bool):
    sim = Simulation(tool_name="art", mesh=False)
    sim.create_world()
    sim.add_robot(name="alice", data_config="so100")
    sim.add_camera(name="look", **CAM)  # before the rollout: add_camera recompiles
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("alice"))

    home = _render(sim)
    captured: list[np.ndarray] = []
    failures: list[int] = []          # step index of each hook failure
    per_step_captured: list[int] = []  # cumulative captured, per step

    def hook(step: int, observation: dict, action: dict) -> None:
        if broken:
            failures.append(step)
            per_step_captured.append(len(captured))
            raise OSError("capture device not ready")
        captured.append(_render(sim))
        per_step_captured.append(len(captured))

    cap = _Capture()
    pr_mod.logger.addHandler(cap)
    old = pr_mod.logger.level
    pr_mod.logger.setLevel(logging.WARNING)
    try:
        result = PolicyRunner(sim).run(
            "alice", policy, duration=2.0, control_frequency=50,
            fast_mode=True, on_frame=hook, max_onframe_failures=limit,
        )
        status = result.get("status", "?")
        text = result["content"][0]["text"] if result.get("content") else ""
        refused = False
    except ValueError as e:
        status, text, refused = "refused", str(e), True
    finally:
        pr_mod.logger.removeHandler(cap)
        pr_mod.logger.setLevel(old)

    final = _render(sim)
    steps = len(per_step_captured)
    np.save(OUT / f"{tag}_final.npy", final)
    np.save(OUT / f"{tag}_home.npy", home)
    if captured:
        np.save(OUT / f"{tag}_frame.npy", captured[len(captured) // 2])
    sim.cleanup()
    return {
        "tag": tag, "limit": repr(limit), "status": status, "text": text[:190],
        "steps": steps, "captured": len(captured), "failures": len(failures),
        "warnings_emitted": cap.emitted, "refused": refused,
        "aborted": "times in a row" in text,
        "per_step_captured": per_step_captured,
    }


rows = [
    run("healthy", None, broken=False),   # what a working capture hook produces
    run("honored", 3, broken=True),       # a limit the counter can honor
    run("nonfinite", math.nan, broken=True),  # the defect
]
for r in rows:
    print(f"{r['tag']:10s} limit={r['limit']:8s} status={r['status']:9s} steps={r['steps']:<4d} "
          f"captured={r['captured']:<4d} failures={r['failures']:<4d} warn={r['warnings_emitted']:<4d} "
          f"aborted={r['aborted']}", flush=True)

(OUT / "facts.json").write_text(json.dumps({"tree": TREE, "rows": rows}, indent=2))
print("wrote", OUT / "facts.json")
