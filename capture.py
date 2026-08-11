"""Derive every figure number: cross-backend matrix, coverage delta, mutations."""
import ast, json, os, pathlib, re, subprocess, sys

RUN = os.environ["GITHUB_RUN_ID"]
BEFORE = json.load(open(f"/tmp/cov-{RUN}.json"))
AFTER = json.load(open(f"/tmp/cov-after-{RUN}.json"))
ISAAC = "strands_robots/simulation/isaac/recording.py"
NEWTON = "strands_robots/simulation/newton/recording.py"
NEWTON_TEST = "tests/simulation/newton/test_recording_lifecycle_guards.py"
BASE_PROBE = "tests/simulation/newton/test_zz_base_lifecycle_probe.py"

# Contract -> the line whose execution proves it ran, located by anchor text so
# the matrix cannot go stale against a shifted file.
CONTRACTS = [
    ("start_recording: a failed recorder creation resets `recording`",
     {ISAAC: ('state["recording"] = False', 401, 0),
      NEWTON: ('world._backend_state["recording"] = False', 371, 0)}),
    ("start_recording: an existing dataset resumes (appends)",
     {ISAAC: ('state["dataset_recorder"] = resumed', 371, 0),
      NEWTON: ('world._backend_state["dataset_recorder"] = resumed', 340, 0)}),
    ("capture hook: no-op when the recorder is absent",
     {ISAAC: ("if rec is None:", 560, 1),
      NEWTON: ("if rec is None:", 494, 1)}),
    ("capture hook: no-op when recording has stopped",
     {ISAAC: ('if not state.get("recording", False):', 557, 1),
      NEWTON: ('if not world._backend_state.get("recording", False):', 491, 1)}),
]

def cell(cov, path, line):
    return line not in set(cov["files"][path]["missing_lines"])

matrix = []
for label, per in CONTRACTS:
    row = {"contract": label, "backends": {}}
    for path, (anchor, ln, off) in per.items():
        src = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
        assert src[ln - 1].strip() == anchor, f"{path}:{ln} is {src[ln-1].strip()!r} not {anchor!r}"
        target = ln + off
        be = "isaac" if "isaac" in path else "newton"
        row["backends"][be] = {"line": target,
                               "before": cell(BEFORE, path, target),
                               "after": cell(AFTER, path, target)}
    matrix.append(row)

cov = {}
for path in (ISAAC, NEWTON):
    n = BEFORE["files"][path]["summary"]["num_statements"]
    bm, am = BEFORE["files"][path]["missing_lines"], AFTER["files"][path]["missing_lines"]
    cov[path] = {"stmts": n, "before_miss": len(bm), "after_miss": len(am),
                 "before_pct": 100 * (n - len(bm)) / n, "after_pct": 100 * (n - len(am)) / n,
                 "closed": sorted(set(bm) - set(am)), "remaining": sorted(am)}

# ---- mutation table (re-run so every cell is measured) ----------------------
MUTATIONS = [
    ("M1  isaac: drop the recording-flag early-out", ISAAC, "_hook",
     [(557, 'if not state.get("recording", False):', None), (558, "return", None)]),
    ("M2  isaac: keep the flag check, discard its return", ISAAC, "_hook",
     [(558, "return", "pass")]),
    ("M3  isaac: drop the recorder-absent early-out", ISAAC, "_hook",
     [(560, "if rec is None:", None), (561, "return", None)]),
    ("M4  isaac: a failed create no longer resets the flag", ISAAC, "start_recording",
     [(401, 'state["recording"] = False', None)]),
    ("M5  isaac: the resume result is never attached", ISAAC, "start_recording",
     [(371, 'state["dataset_recorder"] = resumed', None)]),
    ("M6  newton: drop the recording-flag early-out", NEWTON, "_hook",
     [(491, 'if not world._backend_state.get("recording", False):', None), (492, "return", None)]),
]
NEW_ARM = ["tests/simulation/isaac/test_recording_lifecycle_guards.py",
           NEWTON_TEST + "::TestRunPolicyHookGuards::test_hook_is_noop_after_recording_stops"]
OLD_ARM = ["tests/simulation/isaac/test_dataset_recording.py",
           "tests/simulation/newton/test_dataset_recording.py", BASE_PROBE]

def run(targets):
    r = subprocess.run([sys.executable, "-m", "pytest", *targets, "-q", "--no-cov",
                        "-p", "no:randomly", "--timeout=120"],
                       capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "egl"})
    out = r.stdout + r.stderr
    f, p = re.search(r"(\d+) failed", out), re.search(r"(\d+) passed", out)
    return {"failed": int(f.group(1)) if f else 0, "passed": int(p.group(1)) if p else 0}

pathlib.Path(BASE_PROBE).write_text(
    subprocess.run(["git", "show", f"upstream/main:{NEWTON_TEST}"],
                   capture_output=True, text=True, check=True).stdout, encoding="utf-8")
saved = {f: pathlib.Path(f).read_text(encoding="utf-8") for f in (ISAAC, NEWTON)}
muts = []
try:
    muts.append({"label": "(unmutated control)", "new": run(NEW_ARM), "old": run(OLD_ARM)})
    for label, f, fname, edits in MUTATIONS:
        src = saved[f]; lines = src.splitlines(keepends=True)
        fn = [n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == fname]
        assert len(fn) == 1
        lo, hi = fn[0].lineno, fn[0].end_lineno
        drop, repl = [], {}
        for ln, expect, new in edits:
            assert lo <= ln <= hi and lines[ln - 1].strip() == expect, (label, ln)
            (drop.append(ln) if new is None
             else repl.__setitem__(ln, lines[ln - 1].replace(expect, new)))
        mutated = "".join(repl.get(i + 1, l) for i, l in enumerate(lines) if (i + 1) not in drop)
        assert mutated != src; ast.parse(mutated)
        pathlib.Path(f).write_text(mutated, encoding="utf-8")
        try:
            muts.append({"label": label, "new": run(NEW_ARM), "old": run(OLD_ARM)})
            print("  ", label, muts[-1]["new"], muts[-1]["old"], flush=True)
        finally:
            pathlib.Path(f).write_text(src, encoding="utf-8")
finally:
    for f, s in saved.items():
        pathlib.Path(f).write_text(s, encoding="utf-8")
    pathlib.Path(BASE_PROBE).unlink(missing_ok=True)

facts = {"tree": str(pathlib.Path(__file__).resolve().parents[1]), "matrix": matrix,
         "coverage": cov, "mutations": muts,
         "suite": {"before": 28174, "after": 28179, "skipped": 257},
         "totals": {"before_miss": BEFORE["totals"]["missing_lines"],
                    "after_miss": AFTER["totals"]["missing_lines"]}}
pathlib.Path(f"/tmp/facts-{RUN}.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
print("\nWROTE facts; holes before/after:",
      sum(1 for r in matrix for b in r["backends"].values() if not b["before"]),
      sum(1 for r in matrix for b in r["backends"].values() if not b["after"]))
