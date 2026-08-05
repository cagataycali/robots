"""Measure the dataset-integrity gate against a REAL MuJoCo recording.

Records a two-episode LeRobot dataset from a MuJoCo rollout, derives the
zero-length-episode corruption from it, and records what the gate says about
that dataset for each ``min_frames`` spelling. Run once per tree; each dump
names the tree it measured.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np

import strands_robots.verify_dataset as vd

TREE = str(pathlib.Path(vd.__file__).parents[1])
OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
FPS = 30

ARM = """<mujoco model="probe_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.05">
      <geom type="cylinder" size="0.05 0.05" rgba="0.30 0.34 0.42 1"/>
      <body name="link1" pos="0 0 0.06">
        <joint name="shoulder" type="hinge" axis="0 0 1" range="-2 2" damping="3"/>
        <geom type="capsule" fromto="0 0 0 0.24 0 0" size="0.032" rgba="0.24 0.68 0.94 1"/>
        <body name="link2" pos="0.24 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-2 2" damping="3"/>
          <geom type="capsule" fromto="0 0 0 0.20 0 0" size="0.026" rgba="0.99 0.62 0.20 1"/>
          <body name="tip" pos="0.20 0 0">
            <geom type="sphere" size="0.040" rgba="0.36 0.82 0.44 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="42" ctrlrange="-2 2"/>
    <position name="a_elbow" joint="elbow" kp="42" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""


def record_real_dataset(root: pathlib.Path) -> dict:
    """Record a genuine two-episode LeRobot dataset from a MuJoCo rollout."""
    from strands_robots import Simulation

    xml = OUT / "probe_arm.xml"
    xml.write_text(ARM, encoding="utf-8")
    sim = Simulation(backend="mujoco", tool_name="gate_probe", mesh=False)
    try:
        sim.create_world()
        sim.add_robot(name="arm", urdf_path=str(xml))
        sim.add_camera(name="rig", position=[0.62, -0.62, 0.46], target=[0.18, 0.0, 0.12], fov=40)
        sim.start_recording(repo_id="probe/gate", task="sweep the arm", fps=FPS, root=str(root))
        sim.run_policy(
            robot_name="arm", policy_provider="mock", control_frequency=FPS, n_steps=24, n_episodes=3
        )
        stopped = sim.stop_recording()
    finally:
        sim.cleanup()
    return {"stop_status": stopped["status"]}


def decode_a_frame(root: pathlib.Path) -> tuple[str, dict]:
    """Read one frame back out of the recording's own MP4 (round-trip proof)."""
    import imageio.v3 as iio

    mp4s = sorted(p for p in root.rglob("*.mp4") if p.stat().st_size > 0)
    frames = list(iio.imiter(mp4s[0]))
    npy = OUT / "frame.npy"
    np.save(npy, frames[len(frames) // 2])
    return str(npy), {
        "mp4": str(mp4s[0].relative_to(root)),
        "mp4_kb": round(mp4s[0].stat().st_size / 1024, 1),
        "decoded_frames": len(frames),
    }


def make_corrupt(src: pathlib.Path, dst: pathlib.Path) -> dict:
    """The same recording with episode 1's length column set to 0.

    That is how a zero-length / buffered episode presents on disk: the gate
    reads meta/episodes/**/*.parquet as ground truth.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    pqs = sorted(dst.glob("meta/episodes/**/*.parquet"))
    table = pq.read_table(pqs[0])
    cols = table.to_pydict()
    lengths = list(cols["length"])
    before = list(lengths)
    lengths[-1] = 0
    cols["length"] = lengths
    pq.write_table(pa.table(cols), pqs[0])
    # Keep meta/info.json consistent with the corrupted parquet, so the only
    # problem in this dataset is the zero-length episode itself.
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_frames"] = sum(lengths)
    info_path.write_text(json.dumps(info), encoding="utf-8")
    from strands_robots.dataset_recorder import read_dataset_episode_indices

    seen = read_dataset_episode_indices(dst)
    return {
        "episode_lengths_before": before,
        "episode_lengths_after": lengths,
        "reader_frames_per_episode": seen["frames_per_episode"],
        "reader_total_episodes": seen["total_episodes"],
    }


def gate_verdicts(root: pathlib.Path) -> list[dict]:
    """What does the gate say about this dataset, per min_frames spelling?"""
    values = [1, 0, -5, False, float("nan"), 2.7, "2", None]
    rows = []
    for v in values:
        try:
            r = vd.verify_dataset(root, min_frames=v, check_videos=False, check_stats=False)
            named = any("frame(s)" in p for p in r["problems"])
            rows.append(
                {
                    "value": repr(v),
                    "outcome": r["status"],
                    "named_the_short_episode": named,
                    "problem": (r["problems"][0][:88] if r["problems"] else ""),
                }
            )
        except BaseException as e:  # noqa: BLE001 - an escape past the report contract is the finding
            rows.append(
                {"value": repr(v), "outcome": f"raised {type(e).__name__}", "named_the_short_episode": False,
                 "problem": str(e)[:88]}
            )
    return rows


def cli_exit(root: pathlib.Path, *args: str) -> int:
    p = subprocess.run(
        [sys.executable, "-m", "strands_robots", "verify-dataset", str(root), "--no-check-videos",
         "--no-check-stats", *args],
        capture_output=True, text=True, cwd=TREE,
    )
    return p.returncode


healthy = OUT / "healthy"
shutil.rmtree(healthy, ignore_errors=True)
rec = record_real_dataset(healthy)
frame_path, mp4_facts = decode_a_frame(healthy)
corrupt = OUT / "corrupt"
corruption = make_corrupt(healthy, corrupt)

facts = {
    "tree": TREE,
    "recording": {**rec, **mp4_facts, "fps": FPS},
    "corruption": corruption,
    "frame_npy": frame_path,
    "verdicts": gate_verdicts(corrupt),
    "cli": {
        "corrupt_default": cli_exit(corrupt),
        "corrupt_min_frames_neg5": cli_exit(corrupt, "--min-frames", "-5"),
        "corrupt_min_frames_0": cli_exit(corrupt, "--min-frames", "0"),
        "healthy_default": cli_exit(healthy),
    },
}
(OUT / "facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
print("TREE:", TREE)
print(json.dumps(facts["recording"], indent=2))
print(json.dumps(facts["corruption"], indent=2))
print(json.dumps(facts["cli"], indent=2))
for row in facts["verdicts"]:
    print(f"  min_frames={row['value']:>8} -> {row['outcome']:<22} named_short={row['named_the_short_episode']!s:<5} {row['problem'][:60]}")
