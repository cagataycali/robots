"""Measure the shipped scouting default and the four prose sites, both trees."""
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import strands_robots  # noqa: E402
from strands_robots.mesh import _zenoh_config as zc  # noqa: E402

facts: dict[str, object] = {"tree": str(pathlib.Path(strands_robots.__file__).parents[1])}

# --- source of truth: what the config actually emits ------------------------
os.environ.pop("STRANDS_MESH_MULTICAST", None)
facts["default"] = dict(zc.scouting_block())
os.environ["STRANDS_MESH_MULTICAST"] = "true"
facts["opted_in"] = dict(zc.scouting_block())
os.environ.pop("STRANDS_MESH_MULTICAST", None)

def main_text(rel: str) -> str:
    return subprocess.run(["git", "show", f"upstream/main:{rel}"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout

SITES = [
    ("README.md", r"Peers on the same LAN[\s\S]*?per process\.", r"Peers discover[\s\S]*?into the fleet\."),
    ("strands_robots/mesh/session.py", r"3\. Zenoh[^\n]*\n", r"3\. Zenoh gossip[\s\S]*?endpoints\.\n"),
    ("strands_robots/mesh/iot/camera_offload.py", r"The Zenoh path \([^)]*\)", r"The Zenoh path \([^)]*\)"),
    ("examples/lerobot/architecture.svg", r">Zenoh [^<]*<", r">Zenoh [^<]*<"),
]
sites = []
for rel, before_re, after_re in SITES:
    before = main_text(rel)
    after = (ROOT / rel).read_text(encoding="utf-8")
    b = re.search(before_re, before)
    a = re.search(after_re, after)
    assert b and a, f"{rel}: could not locate the prose"
    sites.append({
        "path": rel,
        "before": " ".join(b.group(0).split()).strip("><"),
        "after": " ".join(a.group(0).split()).strip("><"),
        "changed": before != after,
    })
facts["sites"] = sites

# --- guard verdict on each tree ---------------------------------------------
GUARD = "tests/mesh/test_multicast_prose_matches_scouting_default.py"
def guard_flags() -> int:
    out = subprocess.run([sys.executable, "-m", "pytest", GUARD, "-q", "--no-cov",
                          "-p", "no:randomly", "--tb=no"], cwd=ROOT,
                         capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "egl"})
    m = re.search(r"(\d+) failed", out.stdout)
    return int(m.group(1)) if m else 0

prose = [rel for rel, _, _ in SITES]
# The branch is committed, so a pathspec stash would be a no-op: check the
# prose out from the merge base instead, then restore it from HEAD.
base = subprocess.run(["git", "merge-base", "HEAD", "upstream/main"], cwd=ROOT,
                      capture_output=True, text=True, check=True).stdout.strip()
try:
    subprocess.run(["git", "checkout", "-q", base, "--", *prose], cwd=ROOT, check=True)
    assert "gossip scouting" not in (ROOT / "README.md").read_text(encoding="utf-8"), "revert failed"
    facts["guard_failures_before"] = guard_flags()
finally:
    subprocess.run(["git", "checkout", "-q", "HEAD", "--", *prose], cwd=ROOT, check=True)
    subprocess.run(["git", "reset", "-q", "HEAD", "--", *prose], cwd=ROOT, check=True)
assert "gossip scouting" in (ROOT / "README.md").read_text(encoding="utf-8"), "restore lost the fix"
facts["guard_failures_after"] = guard_flags()

out = pathlib.Path(f"/tmp/art-{os.environ.get('GITHUB_RUN_ID','x')}.json")
out.write_text(json.dumps(facts, indent=2), encoding="utf-8")
print(json.dumps(facts, indent=2))
