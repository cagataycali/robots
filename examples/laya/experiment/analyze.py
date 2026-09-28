"""Turn results/*.jsonl into REPORT.md tables: H1 (actuation), H2 (calibration), H3 (latency)."""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ARMS = ["scripted", "random", "english", "multilingual", "typed-decisions"]
TASKS = ["reach", "pick"]
PROGRESS_EPS_M = 0.001  # a tick "made progress" when the fingertip got >1 mm closer to the cube


def load(arm: str, task: str) -> list[dict]:
    p = RES / f"{arm}_{task}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f}%"


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def f1(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"


def wilson(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n} ({100 * max(0, c - h):.0f}-{100 * min(1, c + h):.0f}%)"


def tick_pairs(eps: list[dict]):
    """Yield (tick, distance_before) for every tick of every episode."""
    for e in eps:
        before = e["start_distance_m"]
        for t in e["ticks"]:
            yield t, before
            before = t["distance_after"]


def auroc(scores: list[float], labels: list[bool]) -> float:
    pos = [s for s, lab in zip(scores, labels) if lab]
    neg = [s for s, lab in zip(scores, labels) if not lab]
    if not pos or not neg:
        return float("nan")
    # rank-based (ties count half)
    wins = 0.0
    neg_sorted = sorted(neg)
    import bisect

    for s in pos:
        lo = bisect.bisect_left(neg_sorted, s)
        hi = bisect.bisect_right(neg_sorted, s)
        wins += lo + 0.5 * (hi - lo)
    return wins / (len(pos) * len(neg))


def brier(scores, labels):
    return mean([(s - (1.0 if lab else 0.0)) ** 2 for s, lab in zip(scores, labels)])


def ece(scores, labels, bins=10):
    tot = len(scores)
    if tot == 0:
        return float("nan")
    buckets = defaultdict(list)
    for s, lab in zip(scores, labels):
        buckets[min(int(s * bins), bins - 1)].append((s, lab))
    return sum(
        len(b) / tot * abs(mean([s for s, _ in b]) - mean([1.0 if lab else 0.0 for _, lab in b]))
        for b in buckets.values()
    )


def h1_table() -> str:
    rows = [
        "| arm | task | n | success (95% CI) | steps to success (median) | min distance m (mean) | final distance m (mean) | cube lifted max z m | hold ticks | gripper ticks | progress ticks (non-hold) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for task in TASKS:
        for arm in ARMS:
            eps = load(arm, task)
            if not eps:
                rows.append(f"| {arm} | {task} | 0 | (not run) | | | | | | | |")
                continue
            s = [e["summary"] for e in eps]
            k = sum(1 for x in s if x["success"])
            sts = [x["steps_to_success"] for x in s if x["steps_to_success"] is not None]
            nonhold = prog = 0
            for t, before in tick_pairs(eps):
                if t["applied"]["joint"] != "none":
                    nonhold += 1
                    prog += before - t["distance_after"] > PROGRESS_EPS_M
            rows.append(
                f"| {arm} | {task} | {len(eps)} | {wilson(k, len(eps))} | {statistics.median(sts) if sts else 'n/a'} | "
                f"{f3(mean([x['min_distance_m'] for x in s]))} | {f3(mean([x['final_distance_m'] for x in s]))} | "
                f"{f3(max(x['cube_max_z_m'] for x in s))} | {pct(mean([x['hold_fraction'] for x in s]))} | "
                f"{pct(mean([x['gripper_fraction'] for x in s]))} | {pct(prog / nonhold) if nonhold else 'n/a (all hold)'} |"
            )
    return "\n".join(rows)


def h1_primitives() -> str:
    rows = [
        "| arm | task | joint choice distribution (all ticks) | direction + | size small/medium/large |",
        "|---|---|---|---|---|",
    ]
    for task in TASKS:
        for arm in ARMS:
            eps = load(arm, task)
            if not eps:
                continue
            joints = Counter()
            dirs = Counter()
            sizes = Counter()
            for e in eps:
                for t in e["ticks"]:
                    joints[t["primitive"]["joint"]] += 1
                    dirs["+" if t["primitive"]["direction"] > 0 else "-"] += 1
                    sizes[t["primitive"]["size"]] += 1
            n = sum(joints.values())
            jd = ", ".join(f"{j} {100 * c / n:.0f}%" for j, c in joints.most_common())
            sz = "/".join(f"{100 * sizes[s] / n:.0f}%" for s in ("small", "medium", "large"))
            rows.append(f"| {arm} | {task} | {jd} | {pct(dirs['+'] / n)} | {sz} |")
    return "\n".join(rows)


def h2_table() -> tuple[str, dict]:
    rows = [
        "| model | task | ticks | P(progress_ok) mean | realized progress rate | AUROC progress_ok vs realized | Brier | ECE (10 bins) | P(cube_in_jaws) mean | finger contact rate | AUROC cube_in_jaws vs contact | hold-gate @0.5 would abstain |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    out = {}
    for task in TASKS:
        for arm in ARMS[2:]:
            eps = load(arm, task)
            if not eps:
                continue
            p_prog, y_prog, p_jaw, y_jaw = [], [], [], []
            for t, before in tick_pairs(eps):
                c = t["confidences"]
                if "progress_ok" not in c:
                    continue
                p_prog.append(c["progress_ok"])
                y_prog.append(before - t["distance_after"] > PROGRESS_EPS_M)
                p_jaw.append(c["cube_in_jaws"])
                y_jaw.append(bool(t["finger_contact"]))
            if not p_prog:
                continue
            gate = sum(1 for p in p_prog if p < 0.5) / len(p_prog)
            out[(arm, task)] = dict(
                auroc=auroc(p_prog, y_prog), brier=brier(p_prog, y_prog), ece=ece(p_prog, y_prog), n=len(p_prog)
            )
            rows.append(
                f"| {arm} | {task} | {len(p_prog)} | {f3(mean(p_prog))} | {pct(mean([1.0 if y else 0.0 for y in y_prog]))} | "
                f"{f3(auroc(p_prog, y_prog))} | {f3(brier(p_prog, y_prog))} | {f3(ece(p_prog, y_prog))} | {f3(mean(p_jaw))} | "
                f"{pct(mean([1.0 if y else 0.0 for y in y_jaw]))} | {f3(auroc(p_jaw, y_jaw))} | {pct(gate)} |"
            )
    return "\n".join(rows), out


def h2_reliability(arm: str, task: str) -> str:
    eps = load(arm, task)
    buckets = defaultdict(list)
    for t, before in tick_pairs(eps):
        c = t["confidences"]
        if "progress_ok" in c:
            buckets[min(int(c["progress_ok"] * 10), 9)].append(before - t["distance_after"] > PROGRESS_EPS_M)
    if not buckets:
        return ""
    rows = [f"reliability {arm}/{task}: bin -> n, realized progress rate"]
    for b in sorted(buckets):
        ys = buckets[b]
        rows.append(
            f"  [{b / 10:.1f}, {(b + 1) / 10:.1f}) -> n={len(ys)}, realized={100 * mean([1.0 if y else 0.0 for y in ys]):.0f}%"
        )
    return "\n".join(rows)


def h3_table(smoke: dict | None) -> str:
    rows = [
        "| model | source | questions | p50 ms | p95 ms | first tick ms (excl. warm-up) | notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS[2:]:
        lat = []
        first = []
        for task in TASKS:
            for e in load(arm, task):
                lat += [t["latency_ms"] for t in e["ticks"][1:]]
                if e["ticks"]:
                    first.append(e["ticks"][0]["latency_ms"])
        if lat:
            rows.append(
                f"| {arm} | in-loop, 10 Hz run_policy + 2 cameras recording, Thor GPU | 5 (gated profile) | {f1(statistics.median(lat))} | {f1(sorted(lat)[int(0.95 * len(lat)) - 1])} | {f1(statistics.median(first))} | {len(lat)} ticks |"
            )
    if smoke:
        for name, rec in smoke.items():
            rows.append(
                f"| {rec['model']} | standalone Router.predict, {rec['device']} | {rec['questions']} | {f1(rec['p50'])} | {f1(rec['p95'])} | | {rec.get('note', '')} |"
            )
    return "\n".join(rows)


def main() -> None:
    smoke_path = HERE / "latency_smoke.json"
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    h2, h2_stats = h2_table()
    parts = [
        "## H1 actuation (same seeds, same cube positions, same primitive vocabulary for every arm)",
        h1_table(),
        "",
        "Primitive choice distribution:",
        "",
        h1_primitives(),
        "",
        "## H2 calibration (progress_ok / cube_in_jaws vs what the simulator then measured)",
        h2,
        "",
        "```",
        "\n".join(filter(None, (h2_reliability(a, t) for a in ARMS[2:] for t in TASKS))),
        "```",
        "",
        "## H3 tick latency on Thor",
        h3_table(smoke),
    ]
    (HERE / "REPORT_TABLES.md").write_text("\n".join(parts) + "\n")
    print("\n".join(parts))


if __name__ == "__main__":
    sys.exit(main())
