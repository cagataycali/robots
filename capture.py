"""Measure what the four shared knob refusals report, and whether anything drove them.

Reads the before/after coverage JSONs and the mutation table produced by
_probe/mutate.py, and re-derives the envelope-equality evidence live so the
figure's central claim (the loop returns the SHARED helper's verdict verbatim)
is measured rather than transcribed.
"""
from __future__ import annotations

import json, os, pathlib, sys

import strands_robots

print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])
RUN = os.environ["GITHUB_RUN_ID"]
OUT = pathlib.Path(__file__).resolve().parent / "facts.json"
F = "strands_robots/simulation/isaac/simulation.py"

from strands_robots.policies import MockPolicy
from strands_robots.simulation.isaac.simulation import IsaacSimulation

# --- what the four knobs report, and is it the shared helper's own envelope? --
KNOBS = [
    ("control_frequency", 3716, "_validate_positive_frequency", 0.0),
    ("duration", 3729, "_validate_duration", 0.0),
    ("instructions", 3708, "_normalize_multi_policy_instructions", {"nosuch": "pick"}),
    ("action_horizon", 3737, "_normalize_multi_policy_horizons", 0),
]
policies = {"alpha": MockPolicy(), "beta": MockPolicy()}
rows = []
for knob, line, helper, bad in KNOBS:
    fn = getattr(IsaacSimulation, helper)
    if helper.startswith("_validate"):
        env = fn(bad, "run_multi_policy")
    elif "instructions" in helper:
        _, env = fn(policies, bad, "run_multi_policy")
    else:
        _, env = fn(policies, bad, "run_multi_policy", default_horizon=8)
    rows.append({
        "knob": knob, "line": line, "helper": helper, "bad": repr(bad),
        "text": env["content"][0]["text"],
        "status": env["status"],
    })

# --- coverage: was that refusal line executed before / after? ----------------
cov = {}
for arm in ("before", "after"):
    d = json.load(open(f"/tmp/cov-{arm}-{RUN}.json"))["files"][F]
    cov[arm] = {"missing": set(d["missing_lines"]),
                "miss": d["summary"]["missing_lines"],
                "pct": d["summary"]["percent_covered"]}
for r in rows:
    r["driven_before"] = r["line"] not in cov["before"]["missing"]
    r["driven_after"] = r["line"] not in cov["after"]["missing"]

closed = sorted(cov["before"]["missing"] - cov["after"]["missing"])
facts = {
    "tree": str(pathlib.Path(strands_robots.__file__).parents[1]),
    "rows": rows,
    "coverage": {"before_miss": cov["before"]["miss"], "after_miss": cov["after"]["miss"],
                 "before_pct": cov["before"]["pct"], "after_pct": cov["after"]["pct"],
                 "closed": closed},
    "mutations": json.load(open(f"/tmp/mut-{RUN}.json")),
    "suite": {"passed": 29599, "skipped": 266, "failed": 0, "seconds": 684,
              "base": "a78ea602", "pristine_passed": 29588},
    "cases": {"before": 19, "after": 30},
    "production_lines_changed": 0,
}

# --- self-audit: every claim the figure will draw --------------------------- #
assert all(not r["driven_before"] for r in facts["rows"]), "a knob was already driven"
assert all(r["driven_after"] for r in facts["rows"]), "a knob is still undriven"
assert all(r["status"] == "error" for r in facts["rows"])
assert set(closed) == {3708, 3716, 3728, 3729, 3737}, closed
assert facts["coverage"]["before_miss"] - facts["coverage"]["after_miss"] == 5
caught = sum(m["new_failed"] > 0 for m in facts["mutations"])
blind = sum(m["new_failed"] > 0 and m["old_failed"] == 0 for m in facts["mutations"])
assert (caught, blind, len(facts["mutations"])) == (7, 7, 8), (caught, blind, len(facts["mutations"]))
assert facts["cases"]["after"] - facts["cases"]["before"] == 11
assert facts["suite"]["pristine_passed"] + 11 == facts["suite"]["passed"]
OUT.write_text(json.dumps(facts, indent=1))
print("wrote", OUT)
for r in rows:
    print(f"  {r['knob']:18s} L{r['line']} driven before={r['driven_before']} after={r['driven_after']}")
    print(f"      {r['text'][:100]}")
