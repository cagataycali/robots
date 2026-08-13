"""Measure the rate-limit reservation on both robot_mesh gate paths.

Run inside the tree under measurement (PYTHONPATH=.). Writes JSON to argv[1].
The mesh/audit patches are applied once on the main thread: unittest.mock
patching is not thread-safe, so a per-worker context would race.
"""
import json, os, pathlib, sys, threading
from unittest.mock import MagicMock, patch

import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)
import strands_robots.tools.robot_mesh as rmt

FN = getattr(rmt.robot_mesh, "__wrapped__", None) or rmt.robot_mesh
ACTION, LIMIT = "emergency_stop", 3


def race(hitl_env, n_threads=2, prefill=2):
    os.environ["STRANDS_MESH_HITL_ACTIONS"] = hitl_env
    os.environ["STRANDS_MESH_AUDIT_DIR"] = "/tmp/audit-art"
    rmt._reset_interrupt_actions_cache()
    rmt._reset_rate_limits()
    rmt._RATE_LIMITS[ACTION] = (LIMIT, 60.0)
    reserve = getattr(rmt, "_rate_limit_check_and_record", None) or rmt._rate_limit_record
    for _ in range(prefill):
        reserve(ACTION)

    arrived = threading.Barrier(n_threads + 1, timeout=30)
    real = rmt._numeric_option_error

    def gated(*a, **k):
        arrived.wait()          # hold every worker between check and reserve
        return real(*a, **k)

    out, lock, mesh = [], threading.Lock(), MagicMock()
    mesh.emergency_stop.return_value = [{"status": "ok"}]

    def worker():
        ctx = MagicMock(); ctx.interrupt.return_value = "y"
        r = FN(action=ACTION, tool_context=ctx)
        with lock:
            out.append((r["status"], r["content"][0]["text"]))

    with patch.object(rmt, "_resolve_mesh", return_value=mesh), \
         patch.object(rmt, "_numeric_option_error", gated):
        ts = [threading.Thread(target=worker, daemon=True) for _ in range(n_threads)]
        for t in ts:
            t.start()
        arrived.wait()
        for t in ts:
            t.join(timeout=30)

    slots = len(rmt._RATE_HISTORY.get(ACTION, ()))
    refusals = [t for s, t in out if s == "error"]
    return {
        "prefill": prefill, "concurrent": n_threads, "limit": LIMIT,
        "statuses": sorted(s for s, _ in out),
        "dispatched": mesh.emergency_stop.call_count,
        "slots_after": slots,
        "exceeds_limit": slots > LIMIT,
        "refusal": refusals[0][:150] if refusals else "",
    }


res = {"tree": TREE, "has_atomic_helper": hasattr(rmt, "_rate_limit_check_and_record"),
       "has_record_only": hasattr(rmt, "_rate_limit_record")}
for label, env in (("approved", ACTION), ("ungated", "none")):
    res[label] = race(env)
    print(label, json.dumps(res[label]))
pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=2))
