"""Shared-guard refusal matrix: for each guard symbol, which callers' refusal lines ran?"""
import ast, json, os, pathlib, re, collections

ROOT = pathlib.Path(".")
cov = json.load(open(f"/tmp/cov-{os.environ['GITHUB_RUN_ID']}.json"))
GUARD = re.compile(r"^(.*_error|coerce_[a-z_]+|.*_problems|validate_[a-z_]+)$")

# guard symbol -> list of (file, func, refusal_line, covered)
by_guard = collections.defaultdict(list)
for path in sorted(cov["files"]):
    p = ROOT / path
    if not p.exists():
        continue
    src = p.read_text(); lines = src.splitlines()
    miss = set(cov["files"][path]["missing_lines"])
    tree = ast.parse(src)
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            # if err := guard(...)  /  if msg := guard(...)
            if not isinstance(node, ast.If):
                continue
            calls = [c for c in ast.walk(node.test)
                     if isinstance(c, ast.Call)]
            names = []
            for c in calls:
                nm = c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "")
                if nm and GUARD.match(nm):
                    names.append(nm)
            if not names:
                continue
            # find the refusal statement in the If body
            for st in node.body:
                if isinstance(st, (ast.Return, ast.Raise)):
                    ln = st.lineno
                    for nm in names:
                        by_guard[nm].append((path, fn.name, ln, ln not in miss))
                    break

print("=== guards where SOME refusals ran and others did not ===")
for nm, rows in sorted(by_guard.items()):
    cov_n = sum(1 for r in rows if r[3])
    if 0 < cov_n < len(rows):
        print(f"\n{nm}: {cov_n}/{len(rows)} refusals executed")
        for path, fnname, ln, ok in sorted(rows):
            print(f"   {'OK ' if ok else 'MISS'} {path}::{fnname} L{ln}")
print()
print("=== guards where NO refusal ran (>=2 sites) ===")
for nm, rows in sorted(by_guard.items()):
    if len(rows) >= 2 and not any(r[3] for r in rows):
        print(f"  {nm}: 0/{len(rows)}  " + ", ".join(f"{p.split('/')[-1]}::{f}" for p, f, _, _ in rows))
