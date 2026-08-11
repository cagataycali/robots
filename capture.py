"""Measure, per branch: the reason a caller gets, and its coverage before/after."""
import json, os, pathlib, tempfile
import strands_robots
ROOT = pathlib.Path(strands_robots.__file__).parents[1]
import strands_robots.tools.harness_memory as hm

before = set(json.load(open("/tmp/d-before.json"))["files"]["strands_robots/tools/harness_memory.py"]["missing_lines"])
after = set(json.load(open("/tmp/d-after.json"))["files"]["strands_robots/tools/harness_memory.py"]["missing_lines"])


def reason(fn):
    try:
        return f"returns {fn()!r}"
    except Exception as e:
        return str(e)


rows = []

# 1. action vocabulary
with tempfile.TemporaryDirectory() as td:
    spec = pathlib.Path(td) / "tool_spec.json"
    spec.write_text(json.dumps({"properties": {"action": {}}}))
    orig, cache = hm._sim_tool_spec_path, hm._valid_actions_cache
    hm._sim_tool_spec_path = lambda: spec
    hm._valid_actions_cache = None
    r = reason(hm.get_valid_actions)
    hm._sim_tool_spec_path, hm._valid_actions_cache = orig, cache
rows.append(("get_valid_actions", "the tool spec carries no `action` enum",
             r.replace(str(spec), "<spec>"), (142, 143), "refuse"))

# 2/3. serializability
rows.append(("_validate_trace", "a trace entry cannot be serialized",
             reason(lambda: hm._validate_trace([{"action": "run_policy"}, {"action": "x", "h": object()}])),
             (205, 206), "refuse"))
rows.append(("_validate_summary", "a summary cannot be serialized",
             reason(lambda: hm._validate_summary({"why": object()})), (239, 240), "refuse"))

# 4. provenance fallback
real = hm._importlib_metadata.version
hm._importlib_metadata.version = lambda n: (_ for _ in ()).throw(hm._importlib_metadata.PackageNotFoundError(n))
rows.append(("_version_string", "the distribution metadata is absent",
             reason(hm._version_string), (277, 278), "degrade"))
hm._importlib_metadata.version = real

# 5. rule store
with tempfile.TemporaryDirectory() as td:
    m = hm.HarnessMemory(storage_dir=pathlib.Path(td))
    m._ensure_dirs()
    m.append_rule("failure_model", "a grasp that does not move the object is empty")
    (m.global_dir / hm._RULE_FILES["success_rule"]).write_bytes(b"ok\n\xff\xfe\n")
    rows.append(("HarnessMemory._read_rules", "a rule store is not valid UTF-8",
                 reason(m.load_rules), (490, 491), "refuse"))
    healthy_ok = hm.HarnessMemory._read_rules(m.global_dir / hm._RULE_FILES["failure_model"])

out = {
    "tree": str(ROOT),
    "branches": [
        {"fn": fn, "trigger": trig, "reason": rsn, "lines": list(lines), "kind": kind,
         "missing_before": sorted(set(lines) & before), "missing_after": sorted(set(lines) & after)}
        for fn, trig, rsn, lines, kind in rows
    ],
    "cov_before": {"missing": sorted(before), "pct": 96},
    "cov_after": {"missing": sorted(after), "pct": 100},
    "healthy_kind_readable": healthy_ok,
    "mutations": json.load(open("/tmp/mutations.json")),
}
pathlib.Path("/tmp/art_facts.json").write_text(json.dumps(out, indent=2))
print("TREE:", ROOT)
for b in out["branches"]:
    print(f"  {b['fn']:26s} {b['kind']:8s} before={b['missing_before']} after={b['missing_after']}")
    print(f"      {b['reason'][:100]}")
print("healthy kind readable:", healthy_ok)
