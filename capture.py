"""Measure both trees: what a caller is told when the rollout dispatch fails."""
from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import json
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

import strands_robots.hardware_robot as hr

ROOT = Path(hr.__file__).resolve().parents[1]
print("TREE:", ROOT, flush=True)

from strands_robots.hardware_robot import Robot as HwRobot  # noqa: E402
from strands_robots.hardware_robot import RobotTaskState  # noqa: E402
from tests.test_hardware_control_loop_rate_guard import _FakeArm  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
LABEL = sys.argv[2]
FACTS: dict[str, Any] = {"tree": str(ROOT), "label": LABEL}


def save() -> None:
    (OUT / f"facts_{LABEL}.json").write_text(json.dumps(FACTS, indent=2))


def make_robot() -> Any:
    r = HwRobot.__new__(HwRobot)
    r.tool_name_str = "thor_arm"
    r.action_horizon = 1
    r.data_config = None
    r.control_frequency = 50.0
    r.action_sleep_time = 1.0 / 50.0
    r._task_state = RobotTaskState()
    r._executor = ThreadPoolExecutor(max_workers=1)
    r._shutdown_event = threading.Event()
    r._stop_requested = threading.Event()
    r._task_admission = threading.Lock()
    r._task_claimed = False
    r.mesh = None
    r.peer_id = None
    r.robot = _FakeArm()

    async def _c() -> tuple[bool, str]:
        return (True, "")

    async def _r() -> bool:
        return True

    r._connect_robot = _c
    r._initialize_policy = lambda p: _r()
    r._publish_ros_telemetry = lambda o, **k: None
    return r


def drive(r: Any, duration: float) -> dict[str, Any]:
    return r._run_control_loop("pick up the cube", 5555, "localhost", "mock", duration)


# ---------------------------------------------------------------- healthy paths
r = make_robot()


async def _nested() -> dict[str, Any]:
    return drive(r, 0.6)


res = asyncio.run(_nested())
FACTS["healthy_nested"] = {
    "status": res["status"],
    "actions": len(r.robot.sent_actions),
    "final_action": dict(r.robot.sent_actions[-1]),
}
print("healthy nested:", FACTS["healthy_nested"]["status"], FACTS["healthy_nested"]["actions"], flush=True)
save()

r2 = make_robot()
res2 = drive(r2, 0.6)
FACTS["healthy_sync"] = {"status": res2["status"], "actions": len(r2.robot.sent_actions)}
save()

# ------------------------------------------------------- the failed dispatch
CAUSE = "can't start new thread"


class Refusing:
    def __init__(self, *a: Any, **k: Any) -> None:
        raise RuntimeError(CAUSE)


r3 = make_robot()
real_pool = concurrent.futures.ThreadPoolExecutor
concurrent.futures.ThreadPoolExecutor = Refusing  # type: ignore[misc]
raised: dict[str, Any] = {"type": None, "message": None}
leaked = 0
try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        async def _fail() -> dict[str, Any]:
            return drive(r3, 0.6)

        try:
            asyncio.run(_fail())
        except BaseException as exc:  # noqa: BLE001 - the report is what is measured
            raised = {"type": type(exc).__name__, "message": str(exc)}
        gc.collect()
    leaked = len([w for w in caught if "was never awaited" in str(w.message)])
finally:
    concurrent.futures.ThreadPoolExecutor = real_pool  # type: ignore[misc]

FACTS["failed_dispatch"] = {
    "injected_cause": CAUSE,
    "raised_type": raised["type"],
    "raised_message": raised["message"],
    "names_the_cause": bool(raised["message"] and CAUSE in raised["message"]),
    "names_the_asyncio_internal": bool(
        raised["message"] and "cannot be called from a running event loop" in raised["message"]
    ),
    "actions_commanded": len(r3.robot.sent_actions),
    "leaked_coroutines": leaked,
}
print("failed dispatch:", json.dumps(FACTS["failed_dispatch"], indent=2), flush=True)
save()

# ------------------------------------------- render what the dispatch delivers
import mujoco  # noqa: E402
import strands_robots as sr  # noqa: E402

sim = sr.Simulation(backend="mujoco", mesh=False)
sim.create_world()
assert sim.add_robot(name="so101")["status"] == "success"
assert sim.add_camera(name="look", position=[0.62, -0.52, 0.42], target=[0.0, 0.0, 0.16], fov=42)[
    "status"
] == "success"

model = sim._world._model
keys = sim.robot_action_keys("so101")
final = FACTS["healthy_nested"]["final_action"]
ordered = [final[f"joint_{i}"] for i in range(len(keys))]

# Map the rollout's generic joint targets positionally onto so101's actuators,
# clamped into each joint's own range so the render is exactly what is claimed.
mapped: dict[str, float] = {}
clamped = 0
for key, want in zip(keys, ordered, strict=True):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"so101/{key}")
    assert aid >= 0, key
    jid = int(model.actuator_trnid[aid, 0])
    lo, hi = (float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1]))
    val = float(np.clip(want, lo, hi))
    if abs(val - want) > 1e-9:
        clamped += 1
    mapped[key] = val

applied_ok = 0
for _ in range(60):
    if sim.send_action(mapped, robot_name="so101", n_substeps=10)["status"] == "success":
        applied_ok += 1

frame = sim.render(camera_name="look", width=760, height=680)
assert frame.get("status") == "success", frame
png = next(b["image"]["source"]["bytes"] for b in frame["content"] if "image" in b)
(OUT / f"pose_{LABEL}.png").write_bytes(png)

obs = sim.get_observation(robot_name="so101")
FACTS["render"] = {
    "mapped_targets": mapped,
    "clamped_components": clamped,
    "applied_ok": applied_ok,
    "achieved": {k: round(float(obs[k]), 4) for k in keys},
}
assert applied_ok == 60, applied_ok
sim.cleanup()
save()
print("render OK; clamped:", clamped, flush=True)
