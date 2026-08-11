import ast, json, pathlib, re, sys
COV = json.load(open(f"/tmp/cov-{sys.argv[1]}.json"))
files = COV["files"]
tot = COV["totals"]
print(f"TOTAL: {tot['num_statements']} stmts, {tot['missing_lines']} miss, {tot['percent_covered']:.2f}%\n")

OPTDEP = ("isaac/", "newton/", "rendering/backgrounds", "rtps/idl", "groot/server_wrapper", "cosmos3/policy_diffusers")
def optdep(p): return any(k in p for k in OPTDEP)

def contig(ms):
    ms = sorted(ms); runs=[]; cur=[ms[0]] if ms else []
    for a,b in zip(ms, ms[1:]):
        if b==a+1: cur.append(b)
        else: runs.append(cur); cur=[b]
    if cur: runs.append(cur)
    return runs

rows=[]
for path, d in files.items():
    miss = set(d["missing_lines"])
    if not miss or optdep(path): continue
    try: tree = ast.parse(pathlib.Path(path).read_text())
    except Exception: continue
    src = pathlib.Path(path).read_text().splitlines()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
        b0 = fn.body[0]
        start = (b0.end_lineno+1) if isinstance(b0, ast.Expr) and isinstance(b0.value, ast.Constant) else fn.lineno
        body = set(range(start, (fn.end_lineno or start)+1))
        m = miss & body
        if not m: continue
        nstmt = len([l for l in body if l in set(d["executed_lines"]) | miss])
        runs = contig(sorted(m))
        rows.append({"path":path,"fn":fn.name,"miss":len(m),"nstmt":max(nstmt,1),
                     "frac":len(m)/max(nstmt,1),"longest":max(len(r) for r in runs),
                     "lines":sorted(m)})

print("### VIEW A - longest contiguous missing run")
for r in sorted(rows, key=lambda r:-r["longest"])[:12]:
    print(f"  {r['longest']:2d}  {r['path']}::{r['fn']}  {r['lines'][:9]}")

print("\n### VIEW A2 - highest missing/body FRACTION (miss>=2)")
for r in sorted([r for r in rows if r["miss"]>=2], key=lambda r:-r["frac"])[:14]:
    print(f"  {r['frac']:.2f} ({r['miss']}/{r['nstmt']})  {r['path']}::{r['fn']}  {r['lines'][:8]}")

print("\n### VIEW B - uncovered REFUSAL lines (raise / return-dict / status:error)")
hits={}
for path,d in files.items():
    miss=set(d["missing_lines"])
    if not miss or optdep(path): continue
    src=pathlib.Path(path).read_text().splitlines()
    tree=ast.parse("\n".join(src))
    fnof={}
    for fn in ast.walk(tree):
        if isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)):
            for l in range(fn.lineno,(fn.end_lineno or fn.lineno)+1): fnof[l]=fn.name
    for l in sorted(miss):
        if l>len(src): continue
        t=src[l-1].strip()
        if t.startswith("raise ") or t.startswith("return {") or '"status": "error"' in t:
            hits.setdefault((path,fnof.get(l,"?")),[]).append(l)
for (p,f),ls in sorted(hits.items(), key=lambda kv:-len(kv[1]))[:16]:
    print(f"  {len(ls)}  {p}::{f}  {ls}")

print("\n### VIEW C - modules whose ENTIRE uncovered set is failure/fallback branches")
for path,d in files.items():
    miss=sorted(d["missing_lines"])
    if not miss or optdep(path) or len(miss)>12: continue
    src=pathlib.Path(path).read_text().splitlines()
    tree=ast.parse("\n".join(src))
    handler=set(); abstract=set()
    for n in ast.walk(tree):
        if isinstance(n,ast.ExceptHandler):
            for l in range(n.lineno,(n.end_lineno or n.lineno)+1): handler.add(l)
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
            if any(isinstance(x,ast.Name) and x.id=="abstractmethod" for x in n.decorator_list):
                for l in range(n.lineno,(n.end_lineno or n.lineno)+1): abstract.add(l)
    if all(l in handler or l in abstract for l in miss): continue
    ok=all(src[l-1].strip().startswith(("raise ","return {","return None","return False","logger.")) or l in handler for l in miss if l<=len(src))
    if ok:
        pct=d["summary"]["percent_covered"]
        print(f"  {pct:5.1f}%  {path}  {miss}")
