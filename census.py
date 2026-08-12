"""Coverage census: VIEW A (contiguous), A2 (fraction), B (uncovered refusals), C (all-failure modules)."""
import ast, json, pathlib, sys, collections

COV = json.load(open(f"/tmp/cov-{sys.argv[1]}.json"))
ROOT = pathlib.Path("strands_robots")
# Optional-dep gated ONLY (do not widen: isaac/recording.py etc are NOT gated)
GATED = {
    "strands_robots/simulation/isaac/simulation.py",
    "strands_robots/simulation/isaac/loaders.py",
    "strands_robots/simulation/newton/simulation.py",
    "strands_robots/rendering/backgrounds.py",
    "strands_robots/rtps/idl.py",
    "strands_robots/policies/groot/server_wrapper.py",
}
# Regions my own open PRs occupy
MINE = {"strands_robots/ros_telemetry.py"}

def runs(sorted_lines):
    out, cur = [], []
    for n in sorted_lines:
        if cur and n == cur[-1] + 1:
            cur.append(n)
        else:
            if cur: out.append(cur)
            cur = [n]
    if cur: out.append(cur)
    return out

rowsA, rowsA2, rowsB, modC = [], [], [], []
for path, data in sorted(COV["files"].items()):
    if path in GATED or path in MINE: continue
    miss = set(data["missing_lines"])
    if not miss: continue
    src = pathlib.Path(path).read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    # VIEW C: whole uncovered set is failure/fallback branches?
    all_fail = True
    for n in sorted(miss):
        t = lines[n-1].strip()
        if not (t.startswith(("raise ", "return ", "pass", "logger.", "continue", "break")) or "status\": \"error" in t):
            all_fail = False; break
    if all_fail and len(miss) >= 2:
        modC.append((path, len(miss), sorted(miss)[:12]))
    # per-function
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        body = fn.body
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
            body = body[1:]
        if not body: continue
        # skip abstractmethod bodies that are just pass
        if len(body) == 1 and isinstance(body[0], ast.Pass): continue
        lo, hi = body[0].lineno, fn.end_lineno
        fmiss = sorted(l for l in miss if lo <= l <= hi)
        if not fmiss: continue
        stmts = sum(1 for l in range(lo, hi+1) if lines[l-1].strip() and not lines[l-1].strip().startswith("#"))
        rr = runs(fmiss)
        longest = max(len(r) for r in rr)
        rowsA.append((longest, path, fn.name, fmiss[:10]))
        if len(fmiss) >= 2:
            rowsA2.append((len(fmiss)/max(stmts,1), len(fmiss), path, fn.name, fmiss[:8]))
        # VIEW B: uncovered refusal lines
        ref = [l for l in fmiss if lines[l-1].strip().startswith(("raise ", "return {")) or "status\": \"error" in lines[l-1]]
        if ref: rowsB.append((path, fn.name, ref))

print("=== VIEW A: longest contiguous missing run (top 12) ===")
for longest, p, f, m in sorted(rowsA, reverse=True)[:12]:
    print(f"  {longest:2d}  {p}::{f}  {m}")
print("\n=== VIEW A2: highest missing/body fraction, miss>=2 (top 14) ===")
for frac, n, p, f, m in sorted(rowsA2, reverse=True)[:14]:
    print(f"  {frac:.2f} ({n:2d})  {p}::{f}  {m}")
print("\n=== VIEW B: uncovered refusal lines (top 20 by count) ===")
for p, f, r in sorted(rowsB, key=lambda x: -len(x[2]))[:20]:
    print(f"  {len(r):2d}  {p}::{f}  {r}")
print("\n=== VIEW C: modules whose whole uncovered set is failure branches ===")
for p, n, m in sorted(modC, key=lambda x: -x[1])[:16]:
    print(f"  {n:3d}  {p}  {m}")
