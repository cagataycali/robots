"""Repro: SimEngine.start_recording(task=X) accepts non-str task.

Asymmetric guard. The design intent is codified in
strands_robots/simulation/base.py:3421 _validate_rollout_target:

    "A non-string instruction (None from a task lookup, a nested list)
     ran to status='success' and was written into the result metadata and
     the recorded ``task`` column, while run_multi_policy refused the
     same value."

That guard is applied to run_policy (base.py:3928), eval_policy (:5989)
and evaluate_benchmark (:6444) — every caller that writes into the
recorded ``task`` column *through the rollout*. But start_recording,
whose ``task`` kwarg is *also* written into the same recorded column
(it is stashed as ``state['recording_task']`` and forwarded to
DatasetRecorder.create(task=...) as ``default_task``, which
add_frame picks up at dataset_recorder.py:1540:

    frame["task"] = task or self.default_task or "untitled"

), has no corresponding validator — a clean asymmetric-guard shape.

start_recording's own body gates every other scalar/collection kwarg:
    - fps                 -> dataset_recording_option_error
    - push_to_hub         -> dataset_recording_posture_error
    - overwrite           -> dataset_recording_posture_error
    - cameras             -> name_list_error
    - repo_id/root        -> resolve_dataset_dir ValueError
    - rate consistency    -> _validate_recording_start_rate
    - already recording   -> _already_recording_error
    - camera schema       -> camera_schema_key_collision_error

Only ``task`` is passed through unchecked. Two silent-wrong outcomes:
    (A) Truthy non-str (int/float/bytes/list/dict/arbitrary object) is
        kept verbatim and lands in the parquet ``task`` column as that
        type -- which will later blow up on pyarrow's string-schema
        enforcement, OR produce a parquet with a malformed task column
        that LeRobot's own readers refuse.
    (B) Falsy value (0, False, [], {}, "", None) is silently rewritten
        to "untitled" by the ``task or default_task or 'untitled'``
        fall-through -- the caller's explicit intent is dropped with no
        notification.

The whole point of start_recording(task=X) IS to label every frame of
every episode in this recording with X -- so a non-str that silently
corrupts the label or a falsy one that silently drops it defeats the
method's documented contract on its central input.

Repro below demonstrates (1) no up-front refusal at the strands_robots
layer, and (2) the exact fall-through semantics the frame-task
composition at dataset_recorder.py:1540 applies.
"""
from __future__ import annotations

import ast
import os
import pathlib
import re
import tempfile


# ---------------------------------------------------------------------------
# (1) Static proof of the asymmetric guard shape.
# ---------------------------------------------------------------------------
PKG = pathlib.Path(__file__).resolve().parents[1] / "strands_robots"
rec_src = (PKG / "simulation" / "recording.py").read_text()
base_src = (PKG / "simulation" / "base.py").read_text()
dr_src = (PKG / "dataset_recorder.py").read_text()

# Isolate start_recording body.
start_idx = rec_src.index("def start_recording(")
end_idx = rec_src.index("def ", start_idx + 1)
start_body = rec_src[start_idx:end_idx]

print("=" * 72)
print("(1) start_recording up-front guards (via error-dict helpers)")
print("=" * 72)
for ln in start_body.splitlines():
    s = ln.strip()
    if (s.startswith("if error :=") or s.startswith("if cameras and")) and "error" in s:
        print(" ", s[:110])

print()
print("Guards above cover every scalar/collection kwarg EXCEPT `task`.")
print()

# Isolate _validate_rollout_target (the design intent).
m = re.search(r"def _validate_rollout_target\(.*?\n\s*return None\n", base_src, re.DOTALL)
assert m, "guard not found"
print("=" * 72)
print("(2) Design intent, codified elsewhere: _validate_rollout_target")
print("=" * 72)
print(m.group(0)[: m.group(0).index('"""', m.group(0).index('"""') + 3) + 3])

# Confirm it is called on every rollout entry point but NOT on start_recording.
calls = re.findall(r"self\._validate_rollout_target\([^)]*\)", base_src + rec_src)
print()
print("Call sites of _validate_rollout_target in the whole codebase:")
for c in calls:
    # Find line number
    for src_name, src in (("base.py", base_src), ("recording.py", rec_src)):
        for i, line in enumerate(src.splitlines(), 1):
            if c in line and "def " not in line:
                print(f"  {src_name}:{i}  {line.strip()[:90]}")
print()

# ---------------------------------------------------------------------------
# (3) Dynamic repro: start_recording accepts non-str task + fall-through
# ---------------------------------------------------------------------------
print("=" * 72)
print("(3) Dynamic repro: start_recording accepts non-str task values")
print("=" * 72)
os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot

sim = Robot("so101", mesh=False)

inputs = [
    123,                 # int (truthy)
    3.14,                # float (truthy)
    True,                # bool (truthy)
    ["pick", "cube"],    # list (truthy)
    {"a": 1},            # dict (truthy)
    b"bytes task",       # bytes (truthy)
    0,                   # int zero (falsy)
    False,               # bool False (falsy)
    [],                  # empty list (falsy)
    "",                  # empty str (falsy)
    None,                # None (falsy)
]

tmp = tempfile.mkdtemp(prefix="task_bug_")
try:
    for bad in inputs:
        d = os.path.join(tmp, "d")
        r = sim.start_recording(repo_id="local/task_bug", root=d, task=bad, fps=30, overwrite=True)
        # Any error here must mention the `task` param to count as a real refusal.
        text = ""
        try:
            text = r["content"][0]["text"]
        except Exception:
            text = str(r)
        refused_task = r.get("status") == "error" and "task" in text.lower() and "lerobot" not in text.lower()
        print(f"  task={bad!r:30s} -> status={r.get('status'):7s}  refused_as_task_bug={refused_task}")
        if r.get("status") == "success":
            sim.stop_recording()
finally:
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------------------
# (4) The silent-wrong fall-through at dataset_recorder.py:1540
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("(4) dataset_recorder.py:1540 frame-task composition semantics")
print("=" * 72)


def frame_task(per_frame_task, default_task):
    # verbatim: frame["task"] = task or self.default_task or "untitled"
    return per_frame_task or default_task or "untitled"


for v in inputs + ["normal label"]:
    out = frame_task(None, v)
    note = ""
    if v is None or v is False or v == 0 or v == "" or v == [] or v == {}:
        note = "  <-- FALSY: user intent silently dropped"
    elif not isinstance(v, str):
        note = "  <-- NON-STR: lands verbatim in parquet task column"
    print(f"  start_recording(task={v!r:30s}) -> parquet 'task' = {out!r}{note}")

print()
print("Summary: start_recording(task=X) accepts every X; the frame-task")
print("line at dataset_recorder.py:1540 then either (a) silently stores a")
print("non-str in the parquet 'task' column or (b) silently rewrites a")
print("falsy value to 'untitled', while run_policy / eval_policy /")
print("evaluate_benchmark refuse the same shapes up front via")
print("_validate_rollout_target. Fix: mirror the guard on start_recording.")
