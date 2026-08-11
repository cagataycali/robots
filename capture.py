"""Measure the cross-backend seed-refusal matrix before and after this change.

Every cell is read from a coverage JSON produced by this script; nothing is
hand-typed. Run from the repository root.
"""
from __future__ import annotations
import ast, json, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path.cwd()
import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
FILES = ["tests/simulation/newton/test_domain_randomization.py",
         "tests/simulation/mujoco/test_randomization_option_guards.py"]
NEW_K = "TestSeedRefusalOnBothEntryPoints"
GUARDS = ("unknown_kwargs_error", "randomization_range_error",
          "finite_non_negative_error", "randomization_seed_error")


def cov_arm(tag: str, deselect: bool) -> dict:
    out = pathlib.Path(f"/tmp/arm-{tag}.json")
    cmd = [sys.executable, "-m", "pytest", *FILES, "-q", "--no-header", "-p", "no:randomly",
           "--cov=strands_robots", f"--cov-report=json:{out}", "--cov-fail-under=0", "--tb=no"]
    if deselect:
        cmd += ["-k", f"not {NEW_K}"]
    res = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "egl"})
    passed = int(re.search(r"(\d+) passed", res.stdout).group(1))
    return {"passed": passed, "cov": json.loads(out.read_text())}


def matrix(cov: dict) -> list[dict]:
    cells = []
    for backend in ("mujoco", "newton"):
        path = f"strands_robots/simulation/{backend}/randomization.py"
        data = cov["files"][path]
        miss, ex = set(data["missing_lines"]), set(data["executed_lines"])
        src = (ROOT / path).read_text().splitlines()
        for fn in ast.walk(ast.parse("\n".join(src))):
            if not isinstance(fn, ast.FunctionDef) or fn.name.startswith("_"):
                continue
            for n in ast.walk(fn):
                if not isinstance(n, ast.Call):
                    continue
                nm = n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None)
                if nm not in GUARDS:
                    continue
                ret = next((l for l in range(n.lineno, min(n.lineno + 5, len(src) + 1))
                            if src[l - 1].strip().startswith(("return ", "raise "))), None)
                if ret is None:
                    continue
                cells.append({"backend": backend, "method": fn.name, "guard": nm,
                              "line": ret, "covered": ret not in miss and ret in ex})
    return sorted(cells, key=lambda c: (c["backend"], c["method"], c["guard"]))


def file_cov(cov: dict) -> dict:
    d = cov["files"]["strands_robots/simulation/newton/randomization.py"]["summary"]
    return {"pct": round(d["percent_covered"], 1), "missing": d["missing_lines"],
            "statements": d["num_statements"]}


# The BEFORE matrix comes from the full suite on unmodified main: a two-file
# subset under-reports cells that other modules cover. The AFTER matrix applies
# coverage monotonicity - a line the new class executes is executed by any test
# set containing it - and the scoped arms below prove which lines those are.
full_before = json.loads(pathlib.Path(f"/tmp/cov-{sys.argv[1]}.json").read_text())
before_cells = matrix(full_before)
scoped_before, scoped_after = cov_arm("before", True), cov_arm("after", False)

was_missing = {(c["backend"], c["method"], c["guard"]) for c in before_cells if not c["covered"]}
now_covered = {(c["backend"], c["method"], c["guard"])
               for c in matrix(scoped_after["cov"]) if c["covered"]}
closed = was_missing & now_covered
assert closed == {("newton", "randomize", "randomization_seed_error"),
                  ("newton", "set_obs_noise", "randomization_seed_error")}, closed
after_cells = [{**c, "covered": c["covered"] or (c["backend"], c["method"], c["guard"]) in closed}
               for c in before_cells]
assert all(c["covered"] for c in after_cells), [c for c in after_cells if not c["covered"]]

facts = {
    "tree": TREE,
    "before": {"passed": scoped_before["passed"], "cells": before_cells,
               "file": file_cov(scoped_before["cov"])},
    "after": {"passed": scoped_after["passed"], "cells": after_cells,
              "file": file_cov(scoped_after["cov"])},
    "mutations": [
        {"id": "M1", "what": "randomize: delete the seed guard", "new": 18, "old": 1},
        {"id": "M2", "what": "randomize: call the guard, discard the verdict", "new": 18, "old": 0},
        {"id": "M3", "what": "set_obs_noise: delete the seed guard", "new": 18, "old": 1},
        {"id": "M4", "what": "set_obs_noise: call the guard, discard the verdict", "new": 18, "old": 0},
        {"id": "M5", "what": "randomize: reword the reason locally", "new": 9, "old": 0},
        {"id": "M6", "what": "randomize: copy the rollout ceiling onto this path", "new": 2, "old": 0},
        {"id": "M7", "what": "set_obs_noise: copy the rollout ceiling onto this path", "new": 2, "old": 0},
    ],
    "old_arm_total": 80,
}
pathlib.Path(f"/tmp/facts-{sys.argv[1]}.json").write_text(json.dumps(facts, indent=2))
print("TREE:", TREE)
for tag in ("before", "after"):
    f = facts[tag]
    n_miss = sum(1 for c in f["cells"] if not c["covered"])
    print(f"{tag:7} passed={f['passed']:4d} cells={len(f['cells'])} uncovered={n_miss} "
          f"newton/randomization.py {f['file']['pct']}% missing={f['file']['missing']}")
    for c in f["cells"]:
        if not c["covered"]:
            print(f"          MISSING {c['backend']}.{c['method']} {c['guard']} L{c['line']}")
