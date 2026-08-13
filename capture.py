"""Measure the bare-filename contract on both trees. Run with PYTHONPATH=<tree>."""
import json, os, pathlib, sys, tempfile
import strands_robots
TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)
from strands_robots.simulation.safe_output import validate_output_path

facts = {"tree": TREE, "rows": [], "render": {}}
OUT = pathlib.Path(sys.argv[1])

def save():
    OUT.write_text(json.dumps(facts, indent=2))

root = pathlib.Path(tempfile.mkdtemp()) / "renders"; root.mkdir()
cwd = os.getcwd()
facts["sandbox"] = str(root)
facts["cwd"] = cwd

CASES = [
    ("frame.png",     dict(sandbox_root=root, allow_abs=False), "bare name, confined"),
    ("sub/frame.png", dict(sandbox_root=root, allow_abs=False), "has a separator"),
    ("..",            dict(sandbox_root=root, allow_abs=False), "bare '..'"),
    ("/tmp/x.png",    dict(sandbox_root=root, allow_abs=False), "absolute outside"),
    ("frame;rm.png",  dict(sandbox_root=root, allow_abs=False), "metacharacter"),
    ("clip.mp4",      dict(sandbox_root=None, allow_abs=True),  "bare name, guards-only"),
]
for path, kw, label in CASES:
    row = {"input": path, "label": label, "confined": kw["sandbox_root"] is not None and not kw["allow_abs"]}
    try:
        r = validate_output_path(path, **kw)
        row["outcome"] = "accepted"
        row["dest"] = str(r)
        row["in_sandbox"] = str(r).startswith(str(root))
    except ValueError as e:
        row["outcome"] = "refused"
        row["reason"] = str(e)
    facts["rows"].append(row)
save()

# symlink planted at the anchored destination must still be refused
outside = pathlib.Path(tempfile.mkdtemp()) / "outside.png"; outside.write_bytes(b"x")
(root / "evil.png").symlink_to(outside)
try:
    validate_output_path("evil.png", sandbox_root=root, allow_abs=False)
    facts["symlink_at_anchored_dest"] = "ACCEPTED (hole)"
except ValueError as e:
    facts["symlink_at_anchored_dest"] = f"refused: {str(e)[:70]}"
save()

# end-to-end render with a bare filename
rroot = pathlib.Path(tempfile.mkdtemp()) / "rsandbox"; rroot.mkdir()
os.environ["STRANDS_ROBOTS_RENDER_ROOT"] = str(rroot)
os.environ.pop("STRANDS_ROBOTS_RENDER_ALLOW_ABS", None)
from strands_robots import Robot
sim = Robot("so101", mesh=False)
try:
    res = sim.render(camera_name="default", width=320, height=240, output_path="frame.png")
    facts["render"]["status"] = res["status"]
    if res["status"] == "success":
        j = next(b["json"] for b in res["content"] if "json" in b)
        png = next(b["image"]["source"]["bytes"] for b in res["content"] if "image" in b)
        facts["render"]["saved_path"] = j["saved_path"]
        facts["render"]["file_exists"] = (rroot / "frame.png").is_file()
        facts["render"]["bytes"] = len(png)
        (OUT.parent / f"render-{pathlib.Path(TREE).name}.png").write_bytes(png)
    else:
        facts["render"]["text"] = next(b["text"] for b in res["content"] if "text" in b)
        facts["render"]["file_exists"] = (rroot / "frame.png").is_file()
finally:
    sim.cleanup()
save()
print(json.dumps(facts, indent=2)[:1400])
