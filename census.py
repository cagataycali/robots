import ast, json, pathlib, sys, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]
print("TREE:", ROOT)
cov = json.load(open(sys.argv[1]))
EXCL = ("isaac/simulation.py","isaac/loaders.py","newton/simulation.py",
        "rendering/backgrounds.py","rtps/idl","groot/server_wrapper.py")
def excluded(p): return any(e in p for e in EXCL)

rowsA, rowsA2, rowsB, rowsC = [], [], [], []
for path, data in sorted(cov["files"].items()):
    if excluded(path): continue
    miss = set(data["missing_lines"])
    if not miss: continue
    src = (ROOT/path).read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    # VIEW C: whole uncovered set is failure/fallback branches?
    all_fail = True
    for ln in miss:
        t = lines[ln-1].strip()
        if not (t.startswith("raise ") or t.startswith("return ") or t.startswith("logger.")
                or '"status": "error"' in t or t.startswith("pass") or t.startswith("except")):
            all_fail = False; break
    if all_fail and len(miss) >= 2:
        rowsC.append((len(miss), path, sorted(miss)[:8]))
    for fn in [n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]:
        body_lo = fn.body[0].lineno; body_hi = fn.end_lineno
        fmiss = sorted(l for l in miss if body_lo <= l <= body_hi)
        if not fmiss: continue
        nbody = sum(1 for l in data["executed_lines"]+data["missing_lines"] if body_lo<=l<=body_hi)
        # longest contiguous run
        best = cur = 1
        for a,b in zip(fmiss, fmiss[1:]):
            cur = cur+1 if b==a+1 else 1
            best = max(best,cur)
        frac = len(fmiss)/max(nbody,1)
        rowsA.append((best,len(fmiss),path,fn.name,fmiss[:6]))
        if len(fmiss)>=2: rowsA2.append((round(frac,3),len(fmiss),path,fn.name,fmiss[:6]))
        # VIEW B: uncovered refusal lines
        for l in fmiss:
            t = lines[l-1].strip()
            if t.startswith("raise ") or t.startswith("return {") or '"status": "error"' in t:
                rowsB.append((path,fn.name,l,t[:70]))

print("\n=== VIEW A: longest contiguous missing run ===")
for r in sorted(rowsA, reverse=True)[:12]: print(f"  run={r[0]} miss={r[1]} {r[2]}::{r[3]} {r[4]}")
print("\n=== VIEW A2: highest missing/body fraction (miss>=2) ===")
for r in sorted(rowsA2, reverse=True)[:14]: print(f"  frac={r[0]} miss={r[1]} {r[2]}::{r[3]} {r[4]}")
print("\n=== VIEW B: uncovered refusal lines, grouped ===")
g = collections.defaultdict(list)
for p,f,l,t in rowsB: g[(p,f)].append((l,t))
for (p,f),v in sorted(g.items(), key=lambda kv:-len(kv[1]))[:16]:
    print(f"  {p}::{f}  n={len(v)}")
    for l,t in v[:4]: print(f"      {l}: {t}")
print("\n=== VIEW C: whole uncovered set is failure branches ===")
for r in sorted(rowsC, reverse=True)[:14]: print(f"  miss={r[0]} {r[1]} {r[2]}")
