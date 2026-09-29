"""Minimal repro: record.md quickstart claims 5 episodes saved but disk is empty.

Runs the *exact* documented pattern from docs/learn/data/record.md. The
`start_recording` reply flags an ImportError from a stale lerobot path, but
`run_policy` still returns status="success", "5/5 episode(s) completed". Only
`verify_dataset_episodes` (an opt-in check documented as "run every time") catches
the fabrication.

    $ cd <robots checkout>
    $ PYTHONPATH=. python3 /tmp/bugbash_repro/record_quickstart_silent_wrong.py

Expected: `run_policy` refuses to advance past a failed `start_recording`,
OR reports 0 episodes saved in its own summary.

Actual (0.5.3 head):
  start_recording => status="error", "Dataset init failed: No module named 'lerobot.utils.feature_utils'"
  run_policy     => status="success", "5/5 episode(s) completed"      <-- SILENT-WRONG
  stop_recording => status="success", "Was not recording."
  disk           => 0 parquet files
  verify(5)      => status="error", "actual: 0"                        <-- only truth

Root cause:
  strands_robots/dataset_recorder.py:1274
    from lerobot.utils.feature_utils import hw_to_dataset_features   # WRONG path
  Correct in installed lerobot 0.5.1:
    from lerobot.datasets.feature_utils import hw_to_dataset_features
  Sibling file strands_robots/streaming_dataset.py:348 already uses the correct path.
"""
import os, sys, tempfile
os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot

with tempfile.TemporaryDirectory() as td:
    root = os.path.join(td, "dataset")
    sim = Robot("so101")
    sim.add_camera("front", position=[0.6, 0.0, 0.4], target=[0.0, 0.0, 0.1])

    start = sim.start_recording(
        repo_id="you/so101_reach", task="reach the cube", fps=30,
        cameras=["front"], root=root,
    )
    print("start_recording status :", start["status"])
    print("start_recording text   :", start["content"][0]["text"][:120])

    rp = sim.run_policy(
        policy_provider="mock", instruction="reach the cube",
        duration=3.0, n_episodes=5, reset_between=True,
    )
    print("run_policy status      :", rp["status"])
    print("run_policy text        :", rp["content"][0]["text"][:120])

    stop = sim.stop_recording()
    print("stop_recording status  :", stop["status"])
    print("stop_recording text    :", stop["content"][0]["text"][:120])

    v = sim.verify_dataset_episodes(expected=5)
    print("verify(5) status       :", v["status"])
    print("verify(5) actual       :", v["content"][1]["json"]["actual"])

    # Assertion the user would want:
    assert rp["status"] == "success"      # succeeds (bug)
    assert v["content"][1]["json"]["actual"] == 0  # 0 episodes on disk
    print()
    print("BUG: run_policy reported 5/5 success while disk shows 0 episodes.")
