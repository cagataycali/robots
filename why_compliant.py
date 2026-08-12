"""For every in-scope block naming multicast, print WHICH clause makes it compliant."""
from __future__ import annotations
import ast, pathlib, re
ROOT = pathlib.Path(__file__).resolve().parent.parent
print("TREE:", ROOT)
MULTICAST = re.compile(r"multicast", re.I)
CLAUSES = {
    "names-the-flag": re.compile(r"STRANDS_MESH_MULTICAST"),
    "multicast-then-off": re.compile(r"multicast[^.]{0,40}(?:off|disabled|not the default|opt-in|opt in)", re.I),
    "off-then-multicast": re.compile(r"(?:off|disabled|opt-in|opt in)[^.]{0,40}multicast", re.I),
}
def blocks(p):
    src = p.read_text(encoding="utf-8")
    if p.suffix == ".py":
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                d = ast.get_docstring(node, clean=False)
                if d: yield (f"docstring:{getattr(node,'name','<module>')}", d)
        run, start = [], 0
        for i, line in enumerate(src.splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                if not run: start = i
                run.append(s.lstrip("#").strip())
            elif run:
                yield (f"comment@L{start}", "\n".join(run)); run = []
        if run: yield (f"comment@L{start}", "\n".join(run))
    elif p.suffix == ".md":
        off = 0
        for para in src.split("\n\n"):
            yield (f"para@L{src[:off].count(chr(10))+1}", para); off += len(para)+2
    elif p.suffix == ".svg":
        for m in re.finditer(r"<text[^>]*>(.*?)</text>", src, re.S):
            yield (f"label@L{src[:m.start()].count(chr(10))+1}", m.group(1))

paths = (
    sorted((ROOT/"strands_robots"/"mesh").rglob("*.py"))
    + [ROOT/"README.md"]
    + sorted((ROOT/"examples").rglob("*.svg")) + sorted((ROOT/"docs").rglob("*.svg"))
)
n_naming = 0
for p in paths:
    for label, text in blocks(p):
        if not MULTICAST.search(text): continue
        n_naming += 1
        matched = [k for k, r in CLAUSES.items() if r.search(text)]
        flat = " ".join(text.split())
        verdict = ",".join(matched) if matched else "*** UNMARKED ***"
        print(f"{p.relative_to(ROOT)} [{label}] -> {verdict}")
        print(f"    {flat[:190]}")
print(f"\nTOTAL blocks naming multicast in scope: {n_naming}")
