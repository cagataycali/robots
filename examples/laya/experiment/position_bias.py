"""Position-bias probe: same state, joint-choice options in 6 orders. If the winner follows the SLOT not the label,
the zero-shot H1 result is an option-order artifact, not a decision about the robot."""

import copy
import json
import random
import sys

sys.path.insert(0, ".")
import laya

from strands_robots.policies.laya.state_text import build_questions

ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
router = laya.Router(preload=False, device="cuda", max_loaded=3)
recs = [json.loads(line) for line in open("results/english_reach.jsonl")]
states = [r["ticks"][k]["state"] for r in recs[:3] for k in (0, 50, 150)]
q = build_questions(ARM)["joint"]
labels = list(q["criteria"].keys())
rng = random.Random(0)
orders = [labels] + [rng.sample(labels, len(labels)) for _ in range(5)]
out = {}
for model in ("english", "multilingual", "typed-decisions"):
    rows = []
    for st in states:
        for order in orders:
            qq = copy.deepcopy(q)
            qq["criteria"] = {k: q["criteria"][k] for k in order}
            res = router.predict(st, {"joint": qq}, model=model)
            a = res["answers"]["joint"]
            ans = str(a["choice"])
            rows.append(
                {
                    "state_i": states.index(st),
                    "canonical": order == labels,
                    "order": order,
                    "answer": ans,
                    "slot": order.index(ans) if ans in order else None,
                    "p": {k: round(float(v), 3) for k, v in (a.get("probabilities") or {}).items()},
                }
            )
    from collections import Counter

    by_state = {}
    for r in rows:
        by_state.setdefault(r["state_i"], set()).add(r["answer"])
    invariant = sum(1 for v in by_state.values() if len(v) == 1)
    canon = Counter(r["answer"] for r in rows if r["canonical"])
    print(
        model,
        "order-invariant states",
        invariant,
        "/",
        len(by_state),
        "canonical-order answers",
        dict(canon),
        flush=True,
    )
    out[model] = {
        "rows": rows,
        "invariant_states": invariant,
        "n_states": len(by_state),
        "canonical": dict(canon),
        "answer_hist": Counter(r["answer"] for r in rows),
        "slot_hist": Counter(r["slot"] for r in rows),
        "n": len(rows),
    }
    print(model, "answers", dict(out[model]["answer_hist"]), "slots", dict(out[model]["slot_hist"]), flush=True)
json.dump(
    {
        m: {
            "answers": dict(v["answer_hist"]),
            "slots": {str(k): c for k, c in v["slot_hist"].items()},
            "n": v["n"],
            "invariant_states": v["invariant_states"],
            "n_states": v["n_states"],
            "canonical": v["canonical"],
            "rows": v["rows"],
        }
        for m, v in out.items()
    },
    open("position_bias.json", "w"),
    indent=1,
)
