"""Coverage census: VIEW A2 (fraction), VIEW B (uncovered refusals), VIEW C (all-failure modules)."""
import ast, json, os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
cov = json.load(open(os.environ["COVJSON"]))
EXCLUDE = (
    "simulation/isaac/simulation.py", "simulation/isaac/loaders.py",
    "simulation/newton/simulation.py", "rendering/backgrounds.py",
    "rtps/idl", "policies/groot/server_wrapper.py",
)
def excluded(p): return any(e in p for e in EXCLUDE)

rowsA, rowsB, rowsC = [], [], []
for path, data in sorted(cov["files"].items()):
    if excluded(path):
        continue
    miss = set(data["missing_lines"])
    if not miss:
        continue
    src = (ROOT / path).read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    # VIEW C: whole uncovered set is failure/fallback branches?
    all_fail = all(
        (lines[n-1].strip().startswith(("raise ", "return {", "return None", "return False", "pass", "logger."))
         or '"status": "error"' in lines[n-1])
        for n in miss if n-1 < len(lines)
    )
    if all_fail and len(miss) >= 2:
        rowsC.append((path, sorted(miss)))
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # skip abstractmethod bodies that are just pass/...
        if any("abstractmethod" in ast.unparse(d) for d in fn.decorator_list):
            continue
        body_start = fn.body[0].lineno
        rng = set(range(body_start, (fn.end_lineno or body_start) + 1))
        fmiss = miss & rng
        if not fmiss:
            continue
        nbody = len(rng)
        # VIEW A2: fraction
        if len(fmiss) >= 2:
            rowsA.append((len(fmiss)/nbody, len(fmiss), nbody, path, fn.name, sorted(fmiss)))
        # VIEW B: uncovered refusal lines
        refs = [n for n in sorted(fmiss)
                if n-1 < len(lines) and (lines[n-1].strip().startswith(("raise ", "return {"))
                                          or '"status": "error"' in lines[n-1])]
        if refs:
            rowsB.append((path, fn.name, refs, len(fmiss)))

print("=== VIEW A2: highest missing/body FRACTION (miss>=2) ===")
for f, nm, nb, p, fnname, ml in sorted(rowsA, reverse=True)[:18]:
    print(f"  frac={f:.2f} miss={nm}/{nb} {p}::{fnname} {ml}")
print()
print("=== VIEW B: uncovered refusal lines ===")
for p, fnname, refs, nm in sorted(rowsB, key=lambda r: -len(r[2]))[:22]:
    print(f"  {p}::{fnname} refusals={refs} total_miss={nm}")
print()
print("=== VIEW C: whole uncovered set is failure branches ===")
for p, ml in sorted(rowsC, key=lambda r: -len(r[1]))[:20]:
    print(f"  {p} {ml}")
print()
print("=== largest single-function blocks (non-excluded) ===")
for f, nm, nb, p, fnname, ml in sorted(rowsA, key=lambda r: -r[1])[:12]:
    print(f"  miss={nm} frac={f:.2f} {p}::{fnname}")
