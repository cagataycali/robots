"""Measure the summary-budget behaviour of whichever tree imports it."""
import json, os, pathlib, sys, tempfile
import strands_robots.tools.harness_memory as hm

out_path = sys.argv[1]
tree = str(pathlib.Path(hm.__file__).parents[2])
print("TREE:", tree)
d = tempfile.mkdtemp(); os.environ["STRANDS_MEMORY_DIR"] = d
mem = hm.HarnessMemory(); CAP = hm._MAX_SUMMARY_BYTES
TRACE = [{"action": "run_policy", "instruction": "grasp the bowl", "n_steps": 20}]

def s_of(n):
    s = {"why": "a"}; s["why"] = "a" * (n - len(json.dumps(s, sort_keys=True)) + 1); return s

# 1. sweep the boundary: which sizes save, and of those which load back
sweep = []
for n in range(CAP - 240, CAP + 41, 4):
    try:
        mem.save_trace("sweep", TRACE, s_of(n)); saved = True
    except ValueError:
        sweep.append({"size": n, "saved": False, "loadable": None}); continue
    try:
        mem.load_trace("sweep"); loadable = True
    except ValueError:
        loadable = False
    sweep.append({"size": n, "saved": saved, "loadable": loadable})
lost = [r["size"] for r in sweep if r["saved"] and r["loadable"] is False]

# 1b. the same sweep through the agent tool, which validates the caller's
# payload first, so it isolates the documented-budget window exactly.
tsweep = []
for n in range(CAP - 200, CAP + 9, 1):
    r = hm.harness_memory(action="save_trace", task="tsweep", trace=TRACE, summary=s_of(n))
    if r["status"] != "success":
        tsweep.append({"size": n, "saved": False, "loadable": None}); continue
    r2 = hm.harness_memory(action="load_trace", task="tsweep")
    tsweep.append({"size": n, "saved": True, "loadable": r2["status"] == "success"})
tlost = [r["size"] for r in tsweep if r["saved"] and r["loadable"] is False]

# 2. the remedy the load failure names, followed exactly, three times
remedy = []
s = s_of(CAP)
try:
    mem.save_trace("remedy", TRACE, s); remedy.append("save: accepted")
    for attempt in (1, 2, 3):
        try:
            mem.load_trace("remedy"); remedy.append(f"attempt {attempt}: loaded"); break
        except ValueError:
            remedy.append(f"attempt {attempt}: refused on load")
            mem.delete_trace("remedy"); mem.save_trace("remedy", TRACE, s)
except ValueError as e:
    remedy.append(f"save: refused -- {str(e)}")

# 3. the invariant: does the save side measure what the load side will
shapes = {
    "ascii": {"why": "plain ascii"},
    "wide chars": {"why": "wide \u00e9\u00e8 \u4e2d\u6587", "n": 12},
    "nested": {"nested": {"a": [1, 2, {"b": 0.125}]}, "flag": True, "none": None},
    "floats": {"floats": [0.1, 1e-09, 1234567.891], "big": 2**53},
}
inv = {}
for label, s in shapes.items():
    measured = []
    real = hm._validate_summary
    hm._validate_summary = lambda p, _r=real, _m=measured: (_m.append(len(json.dumps(p, sort_keys=True))), _r(p))[1]
    try:
        mem.save_trace(f"i{len(inv)}", TRACE, s)
        on_save = measured[-1] if measured else None
        measured.clear()
        mem.load_trace(f"i{len(inv)}")
        on_load = measured[-1] if measured else None
    finally:
        hm._validate_summary = real
    inv[label] = {"on_save": on_save, "on_load": on_load}

# 4. the agent-tool envelope for a summary at the documented budget
r = hm.harness_memory(action="save_trace", task="viatool", trace=TRACE, summary=s_of(CAP))
tool_save = r["status"]
r2 = hm.harness_memory(action="load_trace", task="viatool")
tool_load = r2["status"]

pathlib.Path(out_path).write_text(json.dumps({
    "tree": tree, "cap": CAP, "sweep": sweep, "lost": lost,
    "remedy": remedy, "invariant": inv, "tsweep": tsweep, "tlost": tlost,
    "tool_save": tool_save, "tool_load": tool_load,
}, indent=1))
print(f"direct-API lost={len(lost)} tool-surface lost={len(tlost)} window={(min(tlost),max(tlost)) if tlost else None} tool: save={tool_save} load={tool_load}")
