"""Measure the resolve/inspect split and the planted-cache outcome, per tree."""
from __future__ import annotations
import json, os, pathlib, re, shutil, subprocess, sys

TREE = pathlib.Path(__file__).resolve().parents[1]
CACHE = pathlib.Path.home() / ".cache/huggingface/lerobot"
PLANTS = [CACHE / "local" / "probe", CACHE / "user" / "data"]
PLUG = os.environ["PLUGDIR"]

FOUR = [
    "tests/test_dataset_schema_column_names_distinct.py",
    "tests/test_dataset_recorder_fps_domain.py",
    "tests/test_dataset_schema_frame_shape_domain.py",
    "tests/test_dataset_recorder.py",
]
TOOLS = [
    "tests/tools/test_lerobot_teleoperate.py",
    "tests/tools/test_lerobot_teleoperate_flag_domain.py",
    "tests/tools/test_lerobot_teleoperate_numeric_domain.py",
]

def pytest(paths, plugin=False):
    cmd = [sys.executable, "-m", "pytest", *paths, "-q", "--no-cov", "-p", "no:randomly"]
    env = {**os.environ, "MUJOCO_GL": "egl"}
    if plugin:
        cmd += ["-p", "inspectprobe"]
        env["PYTHONPATH"] = f"{PLUG}:{TREE}"
    out = subprocess.run(cmd, cwd=TREE, capture_output=True, text=True, env=env).stdout
    f = re.search(r"(\d+) failed", out)
    p = re.search(r"(\d+) passed", out)
    return {"failed": int(f.group(1)) if f else 0, "passed": int(p.group(1)) if p else 0,
            "file_exists_error": "FileExistsError" in out}

facts = {"tree": str(TREE)}

# 1. the resolve / inspect split over every module that touches the resolver
facts["split"] = {}
for label, paths in (("recorder_modules", FOUR), ("teleoperate_tool", TOOLS)):
    pytest(paths, plugin=True)
    d = json.loads(pathlib.Path("/tmp/split_hits.json").read_text())
    facts["split"][label] = {"resolves": len(d["resolves"]), "inspects": len(d["inspects"])}

# 2. the planted-cache outcome for the four recorder modules
for p in PLANTS:
    assert not p.exists(), f"refusing to touch existing {p}"
try:
    facts["clean"] = pytest(FOUR)
    for p in PLANTS:
        (p / "meta").mkdir(parents=True)
        (p / "meta" / "info.json").write_text(
            json.dumps({"fps": 30, "total_episodes": 0, "total_frames": 0}), encoding="utf-8")
    facts["planted"] = pytest(FOUR)
finally:
    for p in PLANTS:
        if p.exists():
            shutil.rmtree(p)
    ud = CACHE / "user"
    if ud.exists() and not any(ud.iterdir()):
        ud.rmdir()
    for p in PLANTS:
        assert not p.exists(), f"CLEANUP FAILED for {p}"
facts["cleanup_verified"] = all(not p.exists() for p in PLANTS)

out = pathlib.Path(sys.argv[1])
out.write_text(json.dumps(facts, indent=1), encoding="utf-8")
print(json.dumps(facts, indent=1))
