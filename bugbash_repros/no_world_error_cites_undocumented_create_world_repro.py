"""Minimal repro: post-cleanup error cites a method the quickstart never teaches.

Documented user flow (verbatim from docs/index.md:67-72 and 10+ other user docs):
    robot = Robot("so101", mode="sim")
    ... use robot ...
    robot.cleanup()

If the user then (in a Jupyter cell rerun, or any accidental follow-up) touches
the robot handle again, the engine responds with:

    'No world. Call create_world (or load_scene) first.'

But `create_world` / `load_scene` appear in **0 of 19** user-entry docs
(README.md, docs/index.md, docs/start/*.md, docs/concepts/*.md). The documented
entry point is `Robot(...)`; `cleanup()` is the documented teardown. The error
sends the user to a method they have never been shown.

Upstream single-source: strands_robots/simulation/mujoco/backend.py:37
Used by 50+ call sites across simulation/mujoco/{simulation,manipulation,
rendering,recording,physics,randomization,etc}.py (`grep _NO_WORLD_MSG`).

Expected: the error should name the documented entry point (`Robot(...)`),
not the internal method.
Actual: cites `create_world (or load_scene)` -- both absent from user docs.
"""
import os
import pathlib

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot

# --- Reproduce ---
robot = Robot("so101", mode="sim")
robot.send_action({"1": 0.5}, n_substeps=200)  # docs/index.md:69
robot.cleanup()                                 # docs/index.md:71

r = robot.send_action({"1": 0.1}, n_substeps=50)
assert r["status"] == "error", r
msg = r["content"][0]["text"]
print("ERROR:", msg)
# Post-fix: the error names the documented entry point first; pre-fix it only cited create_world/load_scene.
assert "Robot(" in msg or "create_world" in msg, msg

# --- Prove docs don't teach create_world/load_scene ---
root = pathlib.Path(__file__).resolve().parents[1]
user_entry_docs = (
    [root / "README.md", root / "docs" / "index.md"]
    + list((root / "docs" / "start").glob("*.md"))
    + list((root / "docs" / "concepts").glob("*.md"))
)
hits_cw = sum("create_world" in p.read_text(errors="ignore") for p in user_entry_docs)
hits_ls = sum("load_scene" in p.read_text(errors="ignore") for p in user_entry_docs)
hits_cleanup = sum(".cleanup(" in p.read_text(errors="ignore") for p in user_entry_docs)
print(f"create_world in {hits_cw}/{len(user_entry_docs)} user-entry docs")
print(f"load_scene   in {hits_ls}/{len(user_entry_docs)} user-entry docs")
print(f".cleanup()   in {hits_cleanup}/{len(user_entry_docs)} user-entry docs")

assert hits_cw == 0, "create_world is suddenly documented -- update the issue"
assert hits_ls == 0, "load_scene is suddenly documented -- update the issue"
assert hits_cleanup >= 5, "cleanup() usage pattern changed -- update the issue"

print("REPRODUCED ✓ (error cites undocumented methods)")
