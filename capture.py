"""Measure export_xml's destination handling. Runs in a worktree at main and on the branch."""
import base64, io, json, os, pathlib, shutil, sys, tempfile

import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)

OUT = pathlib.Path(sys.argv[1])
facts = {"tree": TREE}
def save():
    OUT.write_text(json.dumps(facts, indent=2))

from strands_robots import Robot

def _png(sim, cam):
    r = sim.render(camera_name=cam, width=560, height=440)
    assert r.get("status") == "success", r
    return next(c["image"]["source"]["bytes"] for c in r["content"] if "image" in c)

def _text(r):
    for b in r.get("content", []):
        if "text" in b:
            return b["text"]
    return ""

sim = Robot("so101", mesh=False)
sim.add_camera(name="look", position=[0.42, -0.40, 0.34], target=[0.05, 0.02, 0.14], fov=34)
sim.step(60)

# --- the honored path: a real export, and the frame it describes -------------
work = pathlib.Path(tempfile.mkdtemp(prefix="art-"))
good = work / "scene.xml"
r = sim.export_xml(output_path=str(good))
facts["honored"] = {
    "status": r["status"],
    "text": _text(r),
    "wrote": good.exists(),
    "bytes": good.stat().st_size if good.exists() else 0,
    "head": good.read_text().splitlines()[0][:70] if good.exists() else "",
}
save()

frame = _png(sim, "look")
facts["render_png_b64"] = base64.b64encode(frame).decode()
save()

# --- the symlink consequence: a file the caller never named -----------------
victim = work / "important-notes.txt"
victim.write_text("ORIGINAL CONTENT - the caller's own file\n")
link = work / "export-here.xml"
link.symlink_to(victim)
before = victim.read_text()
r = sim.export_xml(output_path=str(link))
after = victim.read_text()
facts["symlink"] = {
    "status": r["status"],
    "text": _text(r)[:150],
    "victim_before": before.strip(),
    "victim_after_first_60": after.strip()[:60],
    "victim_intact": after == before,
    "victim_bytes_before": len(before),
    "victim_bytes_after": len(after),
}
save()

# --- the other three vectors ------------------------------------------------
vectors = {}
esc = work / "escaped-by-traversal.xml"
if esc.exists():
    esc.unlink()
(work / "sub").mkdir(exist_ok=True)   # the parent must exist, or main raises before the vector lands
r = sim.export_xml(output_path=str(work / "sub" / ".." / esc.name))
vectors["traversal (`..`)"] = {"status": r["status"], "text": _text(r)[:90], "escaped": esc.exists()}

meta = work / "a;rm -rf ~;b.xml"
r = sim.export_xml(output_path=str(meta))
vectors["shell metacharacter (`;`)"] = {"status": r["status"], "text": _text(r)[:90], "escaped": meta.exists()}

r = sim.export_xml(output_path=str(work) + "\\nested\\scene.xml")
vectors["backslash separator"] = {"status": r["status"], "text": _text(r)[:90], "escaped": False}
facts["vectors"] = vectors
save()

# --- the envelope contract: an OSError from the caller's path ----------------
env = {}
for label, target in [
    ("missing parent directory", work / "deep" / "nested" / "s.xml"),
    ("destination is a directory", work),
]:
    try:
        r = sim.export_xml(output_path=str(target))
        env[label] = {"outcome": r["status"], "text": _text(r)[:80]}
    except Exception as e:                                    # noqa: BLE001 - the escape IS the finding
        env[label] = {"outcome": f"RAISED {type(e).__name__}", "text": str(e)[:80]}
facts["envelope"] = env
save()

sim.cleanup()
shutil.rmtree(work, ignore_errors=True)
print(json.dumps({k: v for k, v in facts.items() if k != "render_png_b64"}, indent=2))
