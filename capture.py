"""Measure, dumping incrementally so a late failure cannot lose earlier work."""
import json, os, pathlib, re, subprocess, sys, tempfile
import numpy as np, imageio.v3 as iio
import strands_robots
ROOT = pathlib.Path(strands_robots.__file__).parents[1]
print("TREE:", ROOT, flush=True)
RUN = os.environ["GITHUB_RUN_ID"]
MOD = "tests/simulation/test_recording_dataset_stack_unavailable_across_backends.py"
BLOCK = f"/tmp/nomj-{RUN}"
DUMP = pathlib.Path(f"/tmp/art-{RUN}.json")
OUT = {"tree": str(ROOT)}
def save():
    DUMP.write_text(json.dumps(OUT, indent=2))

def run(target, blocked, k=None):
    env = {**os.environ, "MUJOCO_GL": "egl"}
    args = [sys.executable, "-m", "pytest", target, "-q", "--no-header", "-p", "no:randomly", "--no-cov", "-rs"]
    if k: args += ["-k", k]
    if blocked:
        env["PYTHONPATH"] = f"{BLOCK}:{ROOT}"
        args += ["-p", "nomujoco"]
    o = subprocess.run(args, capture_output=True, text=True, env=env, timeout=1200)
    g = lambda p: (int(m.group(1)) if (m := re.search(p, o.stdout)) else 0)
    return {"passed": g(r"(\d+) passed"), "skipped": g(r"(\d+) skipped"), "failed": g(r"(\d+) failed")}

scratch = pathlib.Path("tests/simulation/test_zz_mainver.py")
scratch.write_text(subprocess.run(["git", "show", f"upstream/main:{MOD}"], capture_output=True,
                                  text=True, check=True).stdout)
try:
    OUT["present"] = {"main": run(str(scratch), False), "branch": run(MOD, False)}; save()
    OUT["blocked"] = {"main": run(str(scratch), True), "branch": run(MOD, True)}; save()
    CELL = "test_the_call_is_refused_and_names_the_cause"
    OUT["nine_cells_blocked"] = {"main": run(str(scratch), True, CELL), "branch": run(MOD, True, CELL)}; save()
    P = pathlib.Path("strands_robots/simulation/mujoco/recording.py")
    src = P.read_text()
    OLDM = ('        if unavailable is None and _DatasetRecorder is None:\n'
            '            unavailable = "strands_robots.dataset_recorder did not provide DatasetRecorder."\n')
    assert src.count(OLDM) == 1
    try:
        P.write_text(src.replace(OLDM, "", 1)); assert P.read_text() != src
        OUT["mutation_blocked"] = {"main": run(str(scratch), True), "branch": run(MOD, True)}; save()
    finally:
        P.write_text(src); assert P.read_text() == src
finally:
    scratch.unlink(missing_ok=True)
print("pytest sections done", flush=True)

from strands_robots.simulation.mujoco.simulation import Simulation
rec = {}
with tempfile.TemporaryDirectory() as td:
    ds = pathlib.Path(td) / "ds"
    sim = Simulation(tool_name="art_probe", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm", urdf_path="_art/arm.xml")
    sim.add_camera(name="look", position=[0.30, -0.30, 0.24], target=[0.10, 0.0, 0.05], fov=42)
    FPS = 20
    rec["start"] = sim.start_recording(repo_id="local/art_probe", root=str(ds), fps=FPS, task="probe")["status"]
    rec["rollout"] = sim.run_policy(robot_name="arm", policy_provider="mock", duration=24 / FPS,
                                    control_frequency=float(FPS), action_horizon=1)["status"]
    rec["stop"] = sim.stop_recording()["status"]
    info = json.loads((ds / "meta" / "info.json").read_text())
    rec.update(episodes=info["total_episodes"], frames=info["total_frames"], fps=info["fps"])
    mp4 = ds / "videos" / "observation.images.look" / "chunk-000" / "file-000.mp4"
    rec["mp4_bytes"] = mp4.stat().st_size
    frames = list(iio.imiter(mp4))
    rec["decoded_frames"] = len(frames)
    f = np.asarray(frames[len(frames) - 1])
    np.save(f"/tmp/art-frame-{RUN}.npy", f)
    rec["frame_shape"] = list(f.shape)
    rec["saturated_frac"] = float((((f.max(2).astype(int) - f.min(2).astype(int)) > 45).mean()))
    sim.cleanup()
OUT["recording"] = rec; save()
print(json.dumps(OUT, indent=2))
