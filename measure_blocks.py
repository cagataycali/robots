"""Measure a block-level 'multicast default' prose rule across candidate scopes."""
from __future__ import annotations
import ast, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
print("TREE:", ROOT)

MULTICAST = re.compile(r"multicast", re.I)
# A block is compliant if it names the opt-in knob or says multicast is off.
OFF = re.compile(
    r"STRANDS_MESH_MULTICAST"
    r"|multicast[^.]{0,40}(?:off|disabled|not the default|opt-in|opt in)"
    r"|(?:off|disabled|opt-in|opt in)[^.]{0,40}multicast"
    r"|multicast scouting is \*\*(?:off|disabled)",
    re.I,
)

def py_blocks(path: pathlib.Path):
    """Yield (label, text) for each docstring and each contiguous # comment run."""
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            d = ast.get_docstring(node, clean=False)
            if d:
                name = getattr(node, "name", "<module>")
                yield (f"docstring:{name}", d)
    lines = src.splitlines()
    run: list[str] = []
    start = 0
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith("#"):
            if not run:
                start = i
            run.append(s.lstrip("#").strip())
        else:
            if run:
                yield (f"comment@L{start}", "\n".join(run))
                run = []
    if run:
        yield (f"comment@L{start}", "\n".join(run))

def md_blocks(path: pathlib.Path):
    text = path.read_text(encoding="utf-8")
    off = 0
    for para in text.split("\n\n"):
        line_no = text[:off].count("\n") + 1
        off += len(para) + 2
        if para.strip():
            yield (f"para@L{line_no}", para)

def svg_blocks(path: pathlib.Path):
    text = path.read_text(encoding="utf-8")
    # Each <text>...</text> label is its own block: a diagram label stands alone.
    for m in re.finditer(r"<text[^>]*>(.*?)</text>", text, re.S):
        line_no = text[: m.start()].count("\n") + 1
        yield (f"label@L{line_no}", m.group(1))

def scan(paths):
    hits = []
    for p in sorted(paths):
        if p.suffix == ".py":
            blocks = py_blocks(p)
        elif p.suffix == ".md":
            blocks = md_blocks(p)
        elif p.suffix == ".svg":
            blocks = svg_blocks(p)
        else:
            continue
        for label, text in blocks:
            if MULTICAST.search(text) and not OFF.search(text):
                hits.append((str(p.relative_to(ROOT)), label, " ".join(text.split())[:150]))
    return hits

SCOPES = {
    "A mesh pkg only": list((ROOT / "strands_robots" / "mesh").rglob("*.py")),
    "B mesh + README": list((ROOT / "strands_robots" / "mesh").rglob("*.py")) + [ROOT / "README.md"],
    "C mesh + README + svg": (
        list((ROOT / "strands_robots" / "mesh").rglob("*.py"))
        + [ROOT / "README.md"]
        + list((ROOT / "examples").rglob("*.svg"))
        + list((ROOT / "docs").rglob("*.svg"))
    ),
    "D whole tree (.py/.md/.svg)": [
        p for p in ROOT.rglob("*")
        if p.suffix in {".py", ".md", ".svg"} and ".git" not in p.parts and "_probe" not in p.parts
    ],
}
for name, paths in SCOPES.items():
    hits = scan(paths)
    print(f"\n=== scope {name}: {len(paths)} files, {len(hits)} unmarked blocks ===")
    for f, label, snip in hits:
        print(f"  {f}  [{label}]  {snip}")
