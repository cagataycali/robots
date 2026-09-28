"""H3: standalone Router.predict latency on real recorded states (no sim, no recording), GPU and CPU.

Profiles: 3 questions (joint/direction/size), 5 questions (+ progress_ok/cube_in_jaws), 1 question (judge noul).
Output: latency_smoke.json consumed by analyze.py, one row per (model, device, profile).
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import QUESTION as JUDGE_Q  # noqa: E402, I001
from laya_scene import ARM  # noqa: E402
from strands_robots.policies.laya.state_text import build_questions  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS = ["english", "multilingual", "typed-decisions"]


def states(n: int) -> list[dict]:
    out = []
    for line in (HERE / "judge" / "results" / "scripted_reach.jsonl").read_text().splitlines():
        e = json.loads(line)
        for t in e["ticks"]:
            out.append({**t["state"], "proposed_step": "wrist_flex decrease angle (small step)"})
            if len(out) >= n:
                return out
    return out


def main(devices: list[str], n: int) -> None:
    import laya

    profiles = {
        "3q joint/direction/size": build_questions(ARM, "joint_direction_size"),
        "5q gated profile": build_questions(ARM, "joint_direction_size_gated"),
        "1q judge noul": JUDGE_Q,
    }
    rows = {}
    sample = states(n)
    for device in devices:
        router = laya.Router(preload=False, device=device, max_loaded=3)
        for model in MODELS:
            router.predict(sample[0], profiles["3q joint/direction/size"], model=model)  # load + warm
            for name, qs in profiles.items():
                lat = []
                for st in sample:
                    t0 = time.perf_counter()
                    router.predict(st, qs, model=model)
                    lat.append((time.perf_counter() - t0) * 1000.0)
                lat_sorted = sorted(lat)
                key = f"{model}|{device}|{name}"
                rows[key] = {
                    "model": model,
                    "device": device,
                    "questions": name,
                    "n": len(lat),
                    "p50": statistics.median(lat),
                    "p95": lat_sorted[int(0.95 * len(lat)) - 1],
                    "max": lat_sorted[-1],
                    "note": f"{len(lat)} real recorded states",
                }
                print(key, f"p50 {rows[key]['p50']:.1f} p95 {rows[key]['p95']:.1f} ms", flush=True)
        del router
    (HERE / "latency_smoke.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(
        sys.argv[1].split(",") if len(sys.argv) > 1 else ["cuda", "cpu"], int(sys.argv[2]) if len(sys.argv) > 2 else 60
    )
