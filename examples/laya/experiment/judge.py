"""H2 proper: Laya as a System 1 JUDGE of another arm's proposed step.

For every recorded scripted/random tick (judge/results/*.jsonl, state text + proposed primitive + what the simulator
then measured) ask each Laya checkpoint ONE noul question: "will this proposed step reduce the distance to the cube?"
with the proposal appended to the state. Score P(yes) against the realized progress (>1 mm closer). This is the gate
use case (Laya in front of a proposal generator), separate from H1 where Laya itself proposes.
Output: judge/judge_<model>.jsonl (one row per tick) and judge/JUDGE_TABLES.md.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import PROGRESS_EPS_M, auroc, brier, ece, f3, mean, pct  # noqa: E402

HERE = Path(__file__).resolve().parent / "judge"
MODELS = ["english", "multilingual", "typed-decisions"]
SOURCES = [("scripted", "reach"), ("random", "reach"), ("scripted", "pick"), ("random", "pick")]
QUESTION = {
    "progress_ok": {
        "type": "noul",
        "instructions": (
            "The state describes a robot arm, its gripper and a cube (metres, degrees) and a proposed_step for one joint. "
            "Will applying the proposed_step reduce the distance between the gripper and the cube?"
        ),
    }
}


def rows_for(arm: str, task: str):
    p = HERE / "results" / f"{arm}_{task}.jsonl"
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        before = e["start_distance_m"]
        for t in e["ticks"]:
            prim = t["primitive"]
            if prim["joint"] == "none":
                step = "hold every joint (no movement)"
            elif prim["joint"] == "gripper":
                step = f"gripper {'open' if prim['direction'] > 0 else 'close'} ({prim['size']} step)"
            else:
                step = (
                    f"{prim['joint']} {'increase' if prim['direction'] > 0 else 'decrease'} angle ({prim['size']} step)"
                )
            yield {
                "arm": arm,
                "task": task,
                "episode": e["episode"],
                "tick": t["tick"],
                "state": {**t["state"], "proposed_step": step},
                "realized_progress": (before - t["distance_after"]) > PROGRESS_EPS_M,
                "delta_m": before - t["distance_after"],
            }
            before = t["distance_after"]


def main(device: str = "cuda") -> None:
    import laya

    router = laya.Router(preload=False, device=device, max_loaded=3)
    tables = [
        "| judge model | proposals from | task | ticks | realized progress rate | P(yes) mean | AUROC | Brier | ECE | p50 ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        out = HERE / f"judge_{model}.jsonl"
        done = set()
        if out.exists():
            for line in out.read_text().splitlines():
                r = json.loads(line)
                done.add((r["arm"], r["task"], r["episode"], r["tick"]))
        per = defaultdict(list)
        with out.open("a") as fh:
            for arm, task in SOURCES:
                for row in rows_for(arm, task):
                    key = (arm, task, row["episode"], row["tick"])
                    if key in done:
                        continue
                    t0 = time.perf_counter()
                    res = router.predict(row["state"], QUESTION, model=model)
                    row["p_yes"] = float(res["answers"]["progress_ok"]["noul"])
                    row["latency_ms"] = (time.perf_counter() - t0) * 1000.0
                    del row["state"]
                    fh.write(json.dumps(row) + "\n")
                print(f"[{model}] {arm}/{task} done", flush=True)
        for line in out.read_text().splitlines():
            r = json.loads(line)
            per[(r["arm"], r["task"])].append(r)
        for (arm, task), rs in sorted(per.items()):
            p = [r["p_yes"] for r in rs]
            y = [r["realized_progress"] for r in rs]
            lat = [r["latency_ms"] for r in rs[1:]]
            tables.append(
                f"| {model} | {arm} | {task} | {len(rs)} | {pct(mean([1.0 if v else 0.0 for v in y]))} | {f3(mean(p))} | "
                f"{f3(auroc(p, y))} | {f3(brier(p, y))} | {f3(ece(p, y))} | {statistics.median(lat):.1f} |"
            )
        # pooled
        rs = [r for v in per.values() for r in v]
        p = [r["p_yes"] for r in rs]
        y = [r["realized_progress"] for r in rs]
        tables.append(
            f"| {model} | ALL | both | {len(rs)} | {pct(mean([1.0 if v else 0.0 for v in y]))} | {f3(mean(p))} | {f3(auroc(p, y))} | {f3(brier(p, y))} | {f3(ece(p, y))} | |"
        )
        # reliability
        buckets = defaultdict(list)
        for r in rs:
            buckets[min(int(r["p_yes"] * 10), 9)].append(r["realized_progress"])
        tables.append("")
        tables.append(
            f"reliability {model} (pooled): "
            + "; ".join(
                f"[{b / 10:.1f},{(b + 1) / 10:.1f}) n={len(v)} realized={100 * mean([1.0 if x else 0.0 for x in v]):.0f}%"
                for b, v in sorted(buckets.items())
            )
        )
        tables.append("")
    (HERE / "JUDGE_TABLES.md").write_text("\n".join(tables) + "\n")
    print("\n".join(tables))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cuda")
