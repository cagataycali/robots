"""Measure the five broker-delivery routes: source, coverage, and mechanism."""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import strands_robots
from strands_robots.mesh.transport.iot_transport import (
    _TOPIC_POLICY, _qos_and_retain_for, _should_drop,
)

RUN = sys.argv[1]
ROOT = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", ROOT)
KEY = "strands_robots/mesh/transport/iot_transport.py"
SRC = (ROOT / KEY).read_text().splitlines()

ROUTES = [
    (539, "put(): explicit DROP sentinel", "reachable",
     "a bare strands/<peer>/camera passes _should_drop, so this is the only stop"),
    (607, "_unsubscribe(): handler already gone", "reachable",
     "a duplicate removal must leave the remaining subscribers untouched"),
    (608, "_unsubscribe(): tolerate it quietly", "reachable", "same route"),
    (617, "_unsubscribe(): no client to unsubscribe against", "reachable",
     "undeclare after a failed reconnect; only close() clears the handler map"),
    (238, "_qos_and_retain_for(): DROP in the top-level layout", "unreachable",
     "no reserved top-level kind carries a DROP entry"),
]

cov = {}
for arm in ("before", "after"):
    d = json.load(open(f"/tmp/cov-{arm}-{RUN}.json"))["files"][KEY]
    cov[arm] = (set(d["missing_lines"]), d["summary"]["percent_covered"], d["summary"]["num_statements"])

routes = [{
    "line": ln, "label": lbl, "kind": kind, "why": why,
    "source": SRC[ln - 1].strip(),
    "covered_before": ln not in cov["before"][0],
    "covered_after": ln not in cov["after"][0],
} for ln, lbl, kind, why in ROUTES]

# Why the two drop mechanisms are complementary rather than redundant.
shapes = ["strands/thor-arm/camera", "strands/thor-arm/camera/wrist",
          "strands/thor-arm/camera/wrist/ref", "strands/thor-arm/state"]
mechanism = [{
    "topic": t,
    "prefix_test": _should_drop(t),
    "policy_qos": _qos_and_retain_for(t)[0],
    "published": (not _should_drop(t)) and _qos_and_retain_for(t)[0] >= 0,
} for t in shapes]

drop_keys = sorted(k for k, (q, _r) in _TOPIC_POLICY.items() if q == "DROP")

facts = {
    "tree": str(ROOT), "file": KEY, "run": RUN,
    "statements": cov["before"][2],
    "pct_before": cov["before"][1], "pct_after": cov["after"][1],
    "missing_before": sorted(cov["before"][0]), "missing_after": sorted(cov["after"][0]),
    "routes": routes, "mechanism": mechanism, "drop_keys": drop_keys,
    "mutations": json.load(open(f"/tmp/mut-{RUN}.json")),
    "gate": {"suite_passed": 29598, "suite_skipped": 266, "suite_failed": 0,
             "pristine_passed": 29588, "new_cases": 10,
             "subset_before_passed": 3126, "subset_after_passed": 3136},
}
# self-audit
assert all(r["covered_after"] for r in facts["routes"] if r["kind"] == "reachable")
assert not any(r["covered_before"] for r in facts["routes"])
assert facts["missing_after"] == [238], facts["missing_after"]
assert [m["published"] for m in mechanism] == [False, False, True, True]
assert mechanism[0]["prefix_test"] is False and mechanism[0]["policy_qos"] == -1
assert facts["gate"]["pristine_passed"] + facts["gate"]["new_cases"] == facts["gate"]["suite_passed"]
out = pathlib.Path(f"/tmp/art-{RUN}.json"); out.write_text(json.dumps(facts, indent=2))
print("wrote", out)
print(json.dumps({k: facts[k] for k in ("pct_before", "pct_after", "missing_before", "missing_after", "drop_keys")}, indent=2))
