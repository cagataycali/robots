"""Measure the guard census on both trees + the diagram band geometry. Writes JSON."""
from __future__ import annotations
import ast, json, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "mesh"))
import strands_robots  # noqa: E402
print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])
assert pathlib.Path(strands_robots.__file__).parents[1] == ROOT, "probe resolved the WRONG tree"

# Reuse the shipped guard's own readers, so the figure measures the shipped rule.
import test_discovery_posture_prose as guard  # noqa: E402

def base_text(rel: str) -> str:
    return subprocess.run(
        ["git", "show", f"upstream/main:{rel}"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout

TWO = ["examples/lerobot/architecture.svg", "strands_robots/mesh/iot/camera_offball.py"]
TWO[1] = "strands_robots/mesh/iot/camera_offload.py"

def census(base_side: bool) -> dict:
    """Count blocks naming multicast, and which are unmarked, over the guard's scope."""
    naming, unmarked = [], []
    for path in guard._SURFACES:
        rel = str(path.relative_to(ROOT))
        source = base_text(rel) if (base_side and rel in TWO) else path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            blocks = guard._python_blocks(source)
        elif path.suffix == ".md":
            blocks = guard._markdown_blocks(source)
        else:
            blocks = guard._svg_blocks(source)
        for label, text in blocks:
            if guard._MULTICAST.search(text):
                flat = " ".join(text.split())
                why = []
                if guard._NAMES_THE_FLAG.search(text):
                    why.append("names the flag")
                if guard._SAYS_IT_IS_OFF.search(text):
                    why.append("says it is off")
                naming.append({"file": rel, "label": label, "why": why, "snippet": flat[:70]})
                if not why:
                    unmarked.append({"file": rel, "label": label, "snippet": flat[:70]})
    return {"naming": naming, "unmarked": unmarked}

facts: dict = {"tree": str(ROOT)}
facts["before"] = census(base_side=True)
facts["after"] = census(base_side=False)

# --- the layer-5 diagram band, read out of each SVG's own geometry + CSS ---
def band(source: str) -> dict:
    css = dict(re.findall(r"\.([a-z-]+)\s*\{\s*fill:\s*(#[0-9A-Fa-f]{6})", source))
    rects = [
        {"x": float(m[0]), "y": float(m[1]), "w": float(m[2]), "h": float(m[3]), "attrs": m[4]}
        for m in re.findall(
            r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"([^/]*)/>', source
        )
    ]
    texts = [
        {"x": float(m[0]), "y": float(m[1]), "attrs": m[2], "body": m[3]}
        for m in re.findall(r'<text x="([\d.]+)" y="([\d.]+)"([^>]*)>(.*?)</text>', source, re.S)
    ]
    keep_r = [r for r in rects if 650 <= r["y"] <= 700]
    keep_t = [t for t in texts if 675 <= t["y"] <= 740]
    return {"css": css, "rects": keep_r, "texts": keep_t}

facts["band_before"] = band(base_text(TWO[0]))
facts["band_after"] = band((ROOT / TWO[0]).read_text(encoding="utf-8"))

out = ROOT / "_art" / "facts.json"
out.write_text(json.dumps(facts, indent=2), encoding="utf-8")
print(f"before: {len(facts['before']['naming'])} naming, {len(facts['before']['unmarked'])} unmarked")
print(f"after : {len(facts['after']['naming'])} naming, {len(facts['after']['unmarked'])} unmarked")
for u in facts["before"]["unmarked"]:
    print("  PRE-FIX UNMARKED:", u["file"], u["label"], "|", u["snippet"])
print("band rects:", len(facts["band_after"]["rects"]), "texts:", len(facts["band_after"]["texts"]))
for t in facts["band_after"]["texts"]:
    print(f"   text @({t['x']},{t['y']}) {t['attrs'].strip()[:60]} -> {t['body'][:40]!r}")
