"""
Repro: SimEngine.start_recording(root=<non-str>) raises raw TypeError out of
the agent-tool envelope, bypassing the structured {status:error, content:[...]}
every sibling kwarg (fps, push_to_hub, overwrite, cameras, repo_id) returns.

Sibling guards (see strands_robots/simulation/recording.py:1298-1317):
  - fps              -> dataset_recording_option_error  -> status=error envelope
  - push_to_hub      -> dataset_recording_posture_error -> status=error envelope
  - overwrite        -> dataset_recording_posture_error -> status=error envelope
  - cameras          -> name_list_error                 -> status=error envelope
  - repo_id (string) -> resolve_dataset_dir(ValueError) -> status=error envelope

root is unguarded:
  recording.py:1335-1339 catches ValueError only; dataset_source.py:193 guards
  isinstance(repo_id, str) but line 196 `directory = Path(root)` has no type
  check, so Path(42) raises TypeError which escapes the except block.

Expected: structured {"status": "error", "content": [...]} naming root=.
Actual:   TypeError leaks out of the tool call.

Run: python3 start_recording_root_type_repro.py
"""
import os
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot

r = Robot("so101", mesh=False)

bad_values = [42, True, 3.14, b"/tmp/x", ["/tmp/x"], {"path": "/tmp/x"}]

print("--- SimEngine.start_recording(root=<bad type>) ---")
leaks = 0
for v in bad_values:
    try:
        res = r.start_recording(repo_id="local/x", fps=30, task="t", root=v, overwrite=True)
        status = res.get("status", "?")
        text = res.get("content", [{}])[0].get("text", "")[:160]
        print(f"root={v!r:30s} status={status}")
        print(f"  envelope: {text}")
    except TypeError as e:
        leaks += 1
        print(f"root={v!r:30s} ** TypeError leaked **: {e}")

# Contrast: a str root with the same semantic problem is in-envelope.
print()
print("--- Contrast: str root with semantic problem is in-envelope ---")
res = r.start_recording(repo_id="local/x", fps=30, task="t", root="/", overwrite=True)
print(f"root='/'                      status={res.get('status')}")
print(f"  envelope: {res.get('content', [{}])[0].get('text', '')[:160]}")

# Contrast: non-str repo_id is in-envelope.
print()
print("--- Contrast: non-str repo_id IS in-envelope ---")
try:
    res = r.start_recording(repo_id=42, fps=30, task="t", overwrite=True)
    print(f"repo_id=42                    status={res.get('status')}")
    print(f"  envelope: {res.get('content', [{}])[0].get('text', '')[:200]}")
except TypeError as e:
    print(f"repo_id=42 ** TypeError leaked **: {e}")

# Contrast: non-bool overwrite is in-envelope.
print()
print("--- Contrast: non-bool overwrite IS in-envelope ---")
try:
    res = r.start_recording(repo_id="local/x", fps=30, task="t", overwrite="false")
    print(f"overwrite='false'             status={res.get('status')}")
    print(f"  envelope: {res.get('content', [{}])[0].get('text', '')[:200]}")
except TypeError as e:
    print(f"overwrite='false' ** TypeError leaked **: {e}")

print()
print(f"Leaked TypeErrors: {leaks}/{len(bad_values)}")
print("Expected: 0/{len(bad_values)} (all should return envelope like sibling guards).")
