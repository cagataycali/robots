"""Coverage census over the whole package. Views 1-4."""
import ast, json, os, pathlib, re, sys, collections

import strands_robots
ROOT = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", ROOT)

cov = json.load(open(os.environ["COVJSON"]))
files = cov["files"]

# Optional-dep gated modules: unwinnable on this host.
OPTIONAL = ("simulation/isaac/", "simulation/newton/", "rendering/backgrounds",
            "rtps/", "policies/groot/server_wrapper", "policies/cosmos3/policy_diffusers")

def is_optional(p): return any(o in p for o in OPTIONAL)

def contiguous(nums):
    if not nums: return 0
    nums = sorted(nums); best = cur = 1
    for a, b in zip(nums, nums[1:]):
        cur = cur + 1 if b == a + 1 else 1
        best = max(best, cur)
    return best

rows = []
refusal_rows = collections.defaultdict(list)
guard_rows = []

GUARD_RE = re.compile(r"if\s+(?:\w+\s*:?=\s*)?(?:self\.)?(_?\w*(?:_error|_problems|_validate\w*|validate_\w*))\s*\(")

for path, data in sorted(files.items()):
    if is_optional(path): continue
    missing = set(data["missing_lines"])
    if not missing: continue
    src = (ROOT / path).read_text(encoding="utf-8")
    lines = src.splitlines()
    try: tree = ast.parse(src)
    except SyntaxError: continue

    # View 2/4: uncovered refusal lines
    for ln in sorted(missing):
        if ln - 1 >= len(lines): continue
        txt = lines[ln - 1].strip()
        if txt.startswith("raise ") or txt.startswith("return {") or '"status": "error"' in txt:
            refusal_rows[path].append((ln, txt[:100]))

    # View 3: guard-refusal asymmetry (the return/raise AFTER an `if err := guard(...)`)
    for i, ln in enumerate(lines, start=1):
        m = GUARD_RE.search(ln)
        if not m: continue
        # next non-blank line
        j = i
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines): continue
        nxt = lines[j].strip()
        if (nxt.startswith("return ") or nxt.startswith("raise ")) and (j + 1) in missing:
            guard_rows.append((path, i, m.group(1), nxt[:70]))

    # View 1: per-function
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
        lo, hi = node.lineno, (node.end_lineno or node.lineno)
        body = [n for n in range(lo, hi + 1) if n in data["executed_lines"] or n in missing]
        miss = [n for n in range(lo, hi + 1) if n in missing]
        if not miss or not body: continue
        rows.append((path, node.name, len(miss), len(body), len(miss) / len(body), contiguous(miss), min(miss), max(miss)))

print("\n=== VIEW 1a: longest contiguous missing run (top 14) ===")
for r in sorted(rows, key=lambda r: -r[5])[:14]:
    print(f"  run={r[5]:2d} miss={r[2]:2d}/{r[3]:<3d} frac={r[4]:.2f}  {r[0]}::{r[1]}  L{r[6]}-{r[7]}")

print("\n=== VIEW 1b: highest missing fraction, >=3 missing (top 12) ===")
for r in sorted([r for r in rows if r[2] >= 3], key=lambda r: -r[4])[:12]:
    print(f"  frac={r[4]:.2f} miss={r[2]:2d}/{r[3]:<3d} run={r[5]}  {r[0]}::{r[1]}  L{r[6]}-{r[7]}")

print("\n=== VIEW 3: guard-refusal asymmetry (uncovered return/raise after a guard call) ===")
if not guard_rows: print("  (empty)")
for p, ln, g, nxt in guard_rows: print(f"  {p}:{ln}  guard={g}  -> {nxt}")

print("\n=== VIEW 2/4: uncovered refusal lines by file (files with >=2) ===")
for p, hits in sorted(refusal_rows.items(), key=lambda kv: -len(kv[1])):
    if len(hits) < 2: continue
    print(f"  {p}  ({len(hits)})")
    for ln, txt in hits[:8]: print(f"      L{ln}: {txt}")
