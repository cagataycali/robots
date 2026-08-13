"""Coverage census: VIEW A (contiguous), A2 (fraction), B (uncovered refusals), C (all-failure modules)."""
import ast, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
print("TREE:", ROOT)
cov = json.loads(pathlib.Path(sys.argv[1]).read_text())

# optional-dep gated ONLY
EXCL = (
    "isaac/simulation.py", "isaac/loaders.py", "newton/simulation.py",
    "rendering/backgrounds.py", "rtps/idl", "groot/server_wrapper.py",
)
# regions my own open PRs occupy
MINE_PRS = ("tools/lerobot_train.py",)

def gated(p): return any(e in p for e in EXCL)

rows = []
for path, data in cov["files"].items():
    if gated(path):
        continue
    miss = set(data["missing_lines"])
    if not miss:
        continue
    src = (ROOT / path).read_text()
    lines = src.splitlines()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        body_lo = fn.body[0].lineno
        span = set(range(body_lo, fn.end_lineno + 1))
        fmiss = sorted(miss & span)
        if not fmiss:
            continue
        # skip abstract `pass`-only bodies
        if len(fn.body) == 1 and isinstance(fn.body[0], ast.Pass):
            continue
        nbody = len(span)
        # longest contiguous run
        runs, cur = [], [fmiss[0]]
        for a, b in zip(fmiss, fmiss[1:]):
            if b == a + 1: cur.append(b)
            else: runs.append(cur); cur = [b]
        runs.append(cur)
        longest = max(len(r) for r in runs)
        # refusal-ish uncovered lines
        refus = [n for n in fmiss if re.match(r"\s*(raise |return \{|return _err|return err)", lines[n-1])]
        rows.append(dict(path=path, fn=fn.name, miss=len(fmiss), body=nbody,
                         frac=round(len(fmiss)/max(nbody,1), 3),
                         longest=longest, refusals=len(refus), lines=fmiss))

print("\n=== VIEW A: longest contiguous run (top 14) ===")
for r in sorted(rows, key=lambda r: -r["longest"])[:14]:
    print(f'{r["longest"]:>3}  {r["path"]}::{r["fn"]}  miss={r["miss"]} frac={r["frac"]} {r["lines"][:8]}')

print("\n=== VIEW A2: highest missing/body fraction (miss>=2, top 14) ===")
for r in sorted([r for r in rows if r["miss"] >= 2], key=lambda r: -r["frac"])[:14]:
    print(f'{r["frac"]:>5}  {r["path"]}::{r["fn"]}  miss={r["miss"]}/{r["body"]} {r["lines"][:8]}')

print("\n=== VIEW B: uncovered refusal lines (top 16) ===")
for r in sorted([r for r in rows if r["refusals"]], key=lambda r: -r["refusals"])[:16]:
    print(f'{r["refusals"]:>3}  {r["path"]}::{r["fn"]}  miss={r["miss"]} {r["lines"][:8]}')

print("\n=== VIEW C: modules whose WHOLE uncovered set is failure/fallback branches ===")
for path, data in cov["files"].items():
    if gated(path): continue
    miss = data["missing_lines"]
    if not miss or len(miss) > 12: continue
    src_lines = (ROOT / path).read_text().splitlines()
    kinds = []
    for n in miss:
        t = src_lines[n-1].strip()
        kinds.append(bool(re.match(r"(raise |return |pass$|logger\.|except |warnings\.)", t)))
    if all(kinds):
        pct = data["summary"]["percent_covered"]
        print(f'{len(miss):>3} {path}  {pct:.1f}%  {miss}')
