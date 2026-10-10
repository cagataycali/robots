"""RLENV R1: harvest cross-embodiment demonstrations that actually CARRY THEIR GOAL.

Why this exists. RLENV it1 audited the mjlab dataset factory and found 21,504 episodes /
3,225,600 frames whose per-episode target was recorded NOWHERE: proprio-only schema, no camera
keys, one constant task string, and frame-0 ``observation.state`` std exactly 0 across 8,577
episodes while the action label ranged over 0.0678 rad. Identical input, moving label, so the
behaviour-cloning optimum on that corpus IS the conditional action mean -- which is the
model-side failure CERT and LONELY-ELECTRON each measured from the other end. This script is the
replacement, and its one design rule is:

    EVERY EPISODE MUST CARRY ITS GOAL ON EVERY ROUTE THE LEARNER CAN READ.

Three routes, all three populated (``goal_route_audit.py`` scores a shard on exactly these):
  pixels  a VISIBLE red cube at the per-episode position, seen by `scene` and (where the asset
          has a gripper body) `wrist`. The mjlab reach target was an abstract point drawn only
          into the live viewer, so cameras alone would still not have contained the goal.
  text    a per-episode instruction that NAMES WHERE THE CUBE IS ("...the red cube at the near
          left"), from a 3x3 zone grid over the arm's own reachable box. LE's own generator
          passes `e.prompt() + TASKS[task]`, constant per (arm, task); varying it is this
          script's addition and it is what makes a wrong-INSTRUCTION probe possible at all.
  state   NOT POPULATED, and this is stated rather than claimed: `observation.state` is written by
          the recorder from the arm's joints, so the cube pose does not reach the parquet without a
          recorder change. it3 MEASURED the first shard and found the state carrying no goal, so the
          instruction above is corrected here instead of left standing. What the start jitter does
          buy is that the state is not CONSTANT at frame 0, which is rung-1, not a goal route.
A shard with only the pixel route is learnable but forces every blind ablation to collapse by
construction -- that is a property of the data, not a discovered property of the head.

What is REUSED rather than rebuilt (provenance, per D177c). The scene, the embodiment geometry,
the expert and the success test come from LONELY-ELECTRON's ``strands_vla`` package, imported
from a PINNED worktree (``--svla``, default ~/rlenv-svla-pin) rather than from that lane's live
checkout, so a shard is reproducible from a sha and cannot drift mid-harvest. That package is
also the code that generated the training data other heads were fitted on, so the camera names
and poses (`scene`, `wrist`, 128x128) MATCH TRAINING BY CONSTRUCTION -- the mis-feed D177 caught
cannot happen to a consumer of these shards. The pinned sha is written into every GEN-REPORT.

Two passes, because a single-pass "keep if success" filter is a lie (LE m3_record.py measured
koch touch keeping 3 while the parquet said 5):
  pass A  no recording -> collect the seeds whose success test passes
  pass B  recording on -> replay exactly those seeds, re-check success, and refuse to publish if
          a replayed seed disagrees (determinism is asserted, not assumed)

Rung-1 (D62/EVAL it20): every episode differs at the FIRST observation -- per-episode cube pose
AND a start-pose jitter of at least ``--jitter`` rad. it1 found the factory failing this with
zero margin (frame-0 state spread 0.0); here it is enforced and then MEASURED back off the
written parquet by ``goal_route_audit.py``.

Usage::

    python examples/rlenv/harvest.py --arm so101 --episodes 4 --out /tmp/rlenv_smoke   # smoke
    python examples/rlenv/harvest.py --arm so101 --episodes 200 --out ~/svla-rlenv-data/so101-touch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

# ---------------------------------------------------------------- goal -> words
# A 3x3 grid over the arm's OWN cube_box, named along the arm's approach axis. "near" is the
# base side, "far" the reach side, so the words mean the same thing on an arm whose workspace
# runs along -y (so101/koch/so100) and one along +x (fr3/panda).
_NEAR_FAR = ("near", "middle", "far")
_LAT = ("right", "centre", "left")


def zone_words(e, xy) -> tuple[str, str]:
    """(near/middle/far, right/centre/left) for a cube position in this arm's box."""
    (xlo, xhi), (ylo, yhi) = e.cube_box
    ax, ay = e.approach
    if abs(ax) > abs(ay):  # workspace runs along x; lateral axis is y
        depth = (xy[0] - xlo) / max(1e-9, xhi - xlo)
        lat = (xy[1] - ylo) / max(1e-9, yhi - ylo)
        if ax < 0:
            depth = 1.0 - depth
    else:  # workspace runs along y; lateral axis is x
        depth = (xy[1] - ylo) / max(1e-9, yhi - ylo)
        lat = (xy[0] - xlo) / max(1e-9, xhi - xlo)
        if ay < 0:
            depth = 1.0 - depth
    b = lambda t: 0 if t < 1 / 3 else (1 if t < 2 / 3 else 2)
    return _NEAR_FAR[b(depth)], _LAT[b(lat)]


def goal_instruction(e, task: str, xy, cams: tuple[str, ...]) -> str:
    """The per-episode instruction: embodiment prompt + task + WHERE the cube is."""
    nf, lr = zone_words(e, xy)
    verb = "touch" if task == "touch" else "push"
    return f"{e.prompt(cams)} {verb} the red cube at the {nf} {lr}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True)
    p.add_argument("--task", default="touch", choices=("touch", "push"))
    p.add_argument("--episodes", type=int, default=200, help="episodes wanted (of the --keep class)")
    p.add_argument("--force", action="store_true",
                   help="overwrite a shard already recorded in --out (the recorder wipes it)")
    p.add_argument("--keep", choices=("successes", "failures"), default="successes",
                   help="which outcome class to harvest. 'failures' records the episodes this harness has "
                        "been THROWING AWAY (~50-550 per shard pass): the same scripted expert on the same "
                        "bands, kept only when the success test FAILS. The pass-B gate stays symmetric -- a "
                        "SUCCESS inside a failure shard is exactly as false a label as the reverse.")
    p.add_argument("--max-tries", type=int, default=0, help="pass-A seed budget (default 4x episodes)")
    p.add_argument("--jitter", type=float, default=0.10, help="rung-1 start jitter, rad (D62)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--repo-id", default=None, help="LeRobot repo id (default local/rlenv-<arm>-<task>)")
    p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("--cam", type=int, default=128)
    p.add_argument("--duration-s", type=float, default=None,
                   help="rollout horizon in seconds, default the expert's EPISODE_S. This changes ONLY "
                        "how long the episode runs, NOT the plan: the scripted expert keeps its natural "
                        "speed and holds its last target afterwards. it7 MEASURED that so101 push is "
                        "complete at 6 s and perfectly stable to 18 s (180 steps verified, displacement "
                        "74.9 mm at both, 0/20 episodes disturbed afterwards), so a longer horizon buys "
                        "no task content -- it buys the DWELL that LE it37's 18 s venue will evaluate and "
                        "that 6 s demos never show, at 3x the frames")
    p.add_argument("--band-scale", type=float, default=1.0,
                   help="scale the arm's cube sampling box about its centre. it4 MEASURED that this "
                        "is the lever that makes the task goal-NECESSARY: on so101 touch a goal-blind "
                        "expert (aims at the box centre, reads nothing) succeeds 0.175 at 1.0 but only "
                        "0.025 at 2.0, while the real expert stays at 0.95 -- so 2.0 costs nothing and "
                        "raises the headroom a learner can earn by reading the goal from 0.775 to 0.925")
    p.add_argument("--svla", type=Path, default=Path.home() / "rlenv-svla-pin",
                   help="PINNED strands_vla worktree (provenance; its sha goes in the report)")
    p.add_argument("--report", type=Path, default=None)
    a = p.parse_args(argv)

    sys.path.insert(0, str(a.svla))
    import subprocess
    sha = subprocess.run(["git", "-C", str(a.svla), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()

    from strands_vla import scene as S
    from strands_vla import expert as X
    from strands_vla.embodiments import EMBODIMENTS

    if a.arm not in EMBODIMENTS:
        print(f"no embodiment '{a.arm}'. known: {sorted(EMBODIMENTS)}", file=sys.stderr)
        return 2
    e = EMBODIMENTS[a.arm]
    if a.band_scale != 1.0:
        import dataclasses

        (xlo, xhi), (ylo, yhi) = e.cube_box
        cx, cy = (xlo + xhi) / 2.0, (ylo + yhi) / 2.0
        k = a.band_scale
        e = dataclasses.replace(e, cube_box=((cx + (xlo - cx) * k, cx + (xhi - cx) * k),
                                             (cy + (ylo - cy) * k, cy + (yhi - cy) * k)))
        EMBODIMENTS[a.arm] = e  # scene.build and the expert both read the registry
    tries = a.max_tries or max(8, 4 * a.episodes)
    repo_id = a.repo_id or f"local/rlenv-{a.arm}-{a.task}"
    rng = np.random.default_rng(a.seed)

    # Seeds are drawn ONCE so pass A and pass B see the identical stream.
    seeds = [int(x) for x in rng.integers(0, 2**31 - 1, size=tries)]
    cams = tuple(c for c in ("scene", "wrist") if c == "scene" or e.wrist_parent)

    rec: dict = {
        "arm": a.arm, "task": a.task, "want": a.episodes, "jitter_rad": a.jitter,
        "svla_pin_sha": sha, "cams": list(cams), "cam_px": a.cam, "seed": a.seed,
        "tries_budget": tries, "repo_id": repo_id, "out": str(a.out),
        "band_scale": a.band_scale, "cube_box": e.cube_box,
        "duration_s": float(a.duration_s or X.EPISODE_S), "expert_episode_s": float(X.EPISODE_S),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps({"phase": "config", **rec}), flush=True)
    # it21: the LeRobot recorder WIPES --out, so re-running a harvest into the directory of an
    # already-published shard destroys its report and card while the Hub copy keeps claiming the old episode
    # count. It cost exactly that today (an 18-episode card over a 100-episode shard). Refuse unless forced.
    if (a.out / "ds").exists() or (a.out / "data").exists():
        if not a.force:
            print(json.dumps({"REFUSED": "%s already holds a recorded shard; the recorder would WIPE it "
                                         "(report + card included). Pass --force or choose a new --out."
                                         % a.out}), flush=True)
            return 2
        print(json.dumps({"WARNING": "--force: overwriting the shard already in %s" % a.out}), flush=True)
    (a.report or (a.out / "GEN-REPORT.json")).parent.mkdir(parents=True, exist_ok=True)
    return _run(a, e, S, X, seeds, cams, repo_id, rec)


def _run(a, e, S, X, seeds, cams, repo_id, rec) -> int:
    """Pass A then pass B, sharing one scene so geometry caches are built once."""
    robot, emb = S.build(a.arm, cam_size=(a.cam, a.cam), with_wrist=bool(e.wrist_parent))
    rec["wrist_present"] = "wrist" in cams
    t0 = time.perf_counter()

    rec["goal_pixel_contrast"] = goal_pixel_contrast(robot, emb, S, a, np.random.default_rng(a.seed + 7))
    print(json.dumps({"phase": "goal-contrast", **rec["goal_pixel_contrast"]}), flush=True)

    good: list[dict] = []
    for i, sd in enumerate(seeds):
        if len(good) >= a.episodes:
            break
        r = _episode(robot, emb, S, X, a, sd, cams, record=False)
        want = bool(r["ok"]) == (a.keep == "successes")
        if want:
            good.append({"seed": sd, "cube_xy": r["cube_xy"], "gap": r["gap"],
                         "ok": bool(r["ok"]), "instruction": r["instruction"]})
        if i % 20 == 0 or want:
            print(json.dumps({"phase": "A", "i": i, "seed": sd, "ok": r["ok"],
                              "gap": round(r["gap"], 4), "good": len(good)}), flush=True)
    rec["passA"] = {"tried": min(len(seeds), i + 1), "good": len(good),
                    # NOTE: in --keep failures this is the FAILURE rate, not the success rate. Same field,
                    # inverted meaning -- so the class is written next to it and readers must check it.
                    "keep_rate": round(len(good) / max(1, i + 1), 4),
                    "keep_rate_is_for": a.keep,
                    "seconds": round(time.perf_counter() - t0, 1)}
    print(json.dumps({"phase": "A-done", **rec["passA"]}), flush=True)
    if not good:
        rec["verdict"] = "NO-SUCCESSES"
        _write(a, rec)
        return 1
    kept = good[: a.episodes]
    rec["kept_seeds"] = [g["seed"] for g in kept]
    # it11: this line used to be the ONLY per-episode provenance written, while `kept` held the cube pose
    # and the measured gap for every episode. A shard's parquet carries no cube pose and no success flag, so
    # discarding these made every card claim depend on this one file -- and so101-push-venue18 lost its copy,
    # taking 200 goals with it. Keep the full records, and write them INTO the dataset so the goal travels
    # with the data instead of beside it.
    rec["kept_episodes"] = [{"episode_index": i, "seed": g["seed"], "cube_xy": g["cube_xy"],
                             "gap": round(float(g["gap"]), 6)} for i, g in enumerate(kept)]
    try:
        with open(os.path.join(a.out, "EPISODES.jsonl"), "w") as _f:
            for r in rec["kept_episodes"]:
                _f.write(json.dumps({**r, "arm": a.arm, "task": a.task}) + "\n")
    except OSError as exc:
        rec["episodes_jsonl_error"] = str(exc)
    rec["instruction_variety"] = len({g["instruction"] for g in kept})
    _write(a, rec)

    # ---- pass B: record EXACTLY those seeds, on a fresh scene, and re-check every one ----
    robot.destroy()
    robot, emb = S.build(a.arm, cam_size=(a.cam, a.cam), with_wrist=bool(e.wrist_parent))
    t1 = time.perf_counter()
    r = robot.start_recording(repo_id=repo_id, task=kept[0]["instruction"], fps=e.control_hz,
                              root=str(a.out), overwrite=True, cameras=list(cams))
    if r.get("status") != "success":
        rec["verdict"] = "START-RECORDING-FAILED"
        rec["passB"] = {"error": str(r.get("content"))[:400]}
        robot.destroy()
        _write(a, rec)
        return 1
    bad: list[dict] = []
    try:
        for g in kept:
            rb = _episode(robot, emb, S, X, a, g["seed"], cams, record=True)
            # A replayed seed that no longer succeeds means the harvest is not reproducible, and
            # "successes only" would be a false label on the shard. LE measured this failing
            # (koch kept 3, parquet said 5), so it is asserted here rather than assumed.
            if bool(rb["ok"]) != (a.keep == "successes") or abs(rb["gap"] - g["gap"]) > 1e-6:
                bad.append({"seed": g["seed"], "passA_gap": round(g["gap"], 6),
                            "passB_gap": round(rb["gap"], 6), "passB_ok": rb["ok"]})
            print(json.dumps({"phase": "B", "seed": g["seed"], "ok": rb["ok"],
                              "gap": round(rb["gap"], 4)}), flush=True)
    finally:
        srep = robot.stop_recording()
        robot.destroy()
    js = next((b["json"] for b in (srep.get("content") or [])
               if isinstance(b, dict) and "json" in b), {})
    rec["passB"] = {
        "recorded": len(kept), "parquet_episode_count": js.get("parquet_episode_count"),
        "frames": js.get("frame_count"), "seconds": round(time.perf_counter() - t1, 1),
        "replay_nondeterminism": bad,
        "outcome_class": a.keep,
        "single_class": bool(js.get("parquet_episode_count") == len(kept) and not bad),
        "successes_only": bool(js.get("parquet_episode_count") == len(kept) and not bad
                               and a.keep == "successes"),
    }
    print(json.dumps({"phase": "B-done", **{k: v for k, v in rec["passB"].items()
                                            if k != "replay_nondeterminism"}}), flush=True)
    rec["written_task_strings"] = _task_strings(a.out, repo_id)
    # it20: a shard whose IN-SHARD goal file failed to write still reported OK, because the error was
    # recorded in a field no gate read. The goals are the one thing a parquet cannot carry, so this degrades
    # the verdict rather than sitting in the report as trivia.
    if rec.get("episodes_jsonl_error"):
        rec["verdict_note"] = "GOALS NOT WRITTEN IN-SHARD: %s" % str(rec["episodes_jsonl_error"])[:120]
    rec["verdict"] = ("GOALS MISSING — rebuild the sidecar before publishing"
                      if rec.get("episodes_jsonl_error") else
                      "OK" if rec["passB"]["single_class"]
                      and len(rec["written_task_strings"].get("distinct", [])) > 1
                      else "CHECK-REPORT")
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write(a, rec)
    return 0 if rec["verdict"] == "OK" else 1


def _task_strings(out: Path, repo_id: str) -> dict:
    """Read the task strings BACK OFF THE WRITTEN SHARD.

    The text route only exists if the per-episode ``instruction`` actually reaches the parquet.
    ``start_recording`` also takes a single ``task``, and the recorder's own docstring leaves the
    precedence between the two ambiguous, so this is MEASURED rather than trusted: a shard whose
    tasks.parquet holds one string has no text route no matter what was passed in.
    """
    import pandas as pd

    root = out if (out / "meta").exists() else out / repo_id.replace("/", os.sep)
    hits = sorted(root.rglob("tasks.parquet")) if root.exists() else []
    if not hits:
        return {"error": f"no tasks.parquet under {root}"}
    df = pd.concat([pd.read_parquet(h) for h in hits])
    # LeRobot v3 writes meta/tasks.parquet with the task STRING as the INDEX and `task_index` as
    # the only column. Reading columns[0] therefore yields 0,1,2... and would report "2 distinct
    # tasks" for any shard with two episodes -- a check that passes on data it should fail. it3
    # hit exactly that and this is the fix; the index is tried first and named explicitly.
    if df.index.name == "task":
        vals = sorted({str(v) for v in df.index.dropna()})
    elif "task" in df.columns:
        vals = sorted({str(v) for v in df["task"].dropna()})
    else:
        return {"error": f"no task strings; index={df.index.name} cols={list(df.columns)}"}
    return {"n": len(vals), "distinct": vals[:12], "root": str(root)}


def _episode(robot, e, S, X, a, seed: int, cams, record: bool) -> dict:
    """One episode, following LE m3_record.py's PROVEN order (solve, re-place, replay, score).

    The expert is a keyframe plan replayed by ScriptedReplayPolicy; solve_keyframes MOVES the arm
    while solving, so the scene is reset and the cube re-placed before the replay. Success is the
    minimum arm-to-cube SURFACE gap over the episode (mj_geomDistance, so no end-effector offset
    to calibrate) plus, for push, the cube displacement along the approach axis.
    """
    rng = np.random.default_rng(90000 + seed)
    cube_xy = S.sample_cube(e, rng, a.task)

    def place():
        robot.reset()
        robot.move_object(S.CUBE, position=[cube_xy[0], cube_xy[1], e.cube_z])

    place()
    kf = X.solve_keyframes(robot, e, cube_xy, a.task)
    place()
    # RUNG-1 start jitter, applied AFTER the final reset and with reset_between=False below,
    # because run_policy(reset_between=True) resets the arm and silently wipes it -- measured in
    # it3: the first shard came out with frame-0 observation.state spread exactly 0.0.
    if a.jitter > 0:
        import mujoco

        m, d = robot.mj_model, robot.mj_data
        adr = S._qadr(robot, e)
        sign = rng.choice([-1.0, 1.0], size=len(adr))
        mag = a.jitter * (1.0 + 0.5 * rng.random(len(adr)))
        for k, ad in enumerate(adr):
            if ad >= 0:
                d.qpos[ad] = float(d.qpos[ad] + sign[k] * mag[k])
        mujoco.mj_forward(m, d)
    q0 = S.joints(robot, e).copy()
    pol = X.ScriptedReplayPolicy(hz=e.control_hz)
    pol.set_robot_state_keys(list(e.action_keys))
    pol.load_plan(kf, a.task)
    st = {"gap": 1e9, "c0": None}

    def obs(ev, st=st):
        n = type(ev).__name__
        if n == "RunPolicyStarted":
            st["c0"] = S.cube_pos(robot)
        elif n == "RunPolicyStep":
            st["gap"] = min(st["gap"], S.cube_surface_gap(robot, e))

    instr = goal_instruction(e, a.task, cube_xy, cams)
    robot.run_policy(robot_name=e.name, policy_object=pol, instruction=instr,
                     duration=(a.duration_s or X.EPISODE_S), control_frequency=e.control_hz,
                     n_episodes=1,
                     reset_between=(a.jitter <= 0), seed=seed, observer=obs, fast_mode=True)
    c1 = S.cube_pos(robot)
    c0 = st["c0"] if st["c0"] is not None else c1
    disp = float(np.dot(c1 - c0, np.array([e.approach[0], e.approach[1], 0.0])))
    ok = bool(X.success(a.task, st["gap"] <= X.TOUCH_GAP, disp))
    return {"ok": ok, "gap": float(st["gap"]), "disp": disp,
            "cube_xy": [round(float(x), 5) for x in cube_xy], "instruction": instr,
            "q0": [round(float(v), 4) for v in q0], "ik": kf.get("_ik_ok", {})}


def goal_pixel_contrast(robot, e, S, a, rng, n: int = 6) -> dict:  # noqa: C901
    """MEASURE how much signal the GOAL alone puts in the image, with the arm held still.

    Rung-1 asks that episodes differ at the first observation. RLENV it2 measured that this can be
    satisfied by variation that carries NO goal: a shard with a jittered start pose shows a large
    frame-0 pixel delta across episodes (SIMDATA so101-reach-21: 20.64/255 mean pairwise absolute
    difference) but most of it is the ARM sitting elsewhere, which proprioception already supplies,
    while a shard with a fixed start shows only the cube moving (LE ~/svla-data/so101: 0.88/255).
    Those two numbers are not comparable as "goal signal".

    This isolates it: same reset pose every time, ONLY the cube moves, render `scene`. The result
    is an upper bound on what any vision policy could read about the goal from one frame of this
    scene, and it is a property of the SCENE (cube size, camera pose, resolution), not of a policy.
    """
    frames = []
    for _ in range(n):
        xy = S.sample_cube(e, rng, a.task)
        robot.reset()
        robot.move_object(S.CUBE, position=[xy[0], xy[1], e.cube_z])
        # Images come from the OBSERVATION dict (raw camera names), the same route
        # strands_vla/images.py reads, not from Robot.render (which returns an agent-tool text
        # payload, not pixels). Measured in it2: render()['content'] is [{'text': ...}].
        o = robot.get_observation(robot_name=e.name)
        v = o.get("scene", o.get("observation.images.scene"))
        if v is not None:
            frames.append(np.asarray(v).astype(np.float32))
    if len(frames) < 2:
        return {"error": f"rendered {len(frames)} frames"}
    A = np.stack(frames)
    d = [float(np.abs(A[i] - A[j]).mean()) for i in range(len(A)) for j in range(i + 1, len(A))]
    return {"n": len(A), "mean_pairwise_abs_diff": round(float(np.mean(d)), 4),
            "min_pairwise_abs_diff": round(float(np.min(d)), 4),
            "mean_per_pixel_std": round(float(A.std(0).mean()), 4),
            "identical_pairs": int(sum(1 for x in d if x == 0.0)), "pairs": len(d),
            "note": "arm held at reset pose; only the cube moves -> goal-only signal, 0-255 scale"}


def _as_array(content) -> "np.ndarray | None":
    """Best-effort decode of whatever Robot.render returns (png bytes, list, or ndarray)."""
    if content is None:
        return None
    if isinstance(content, np.ndarray):
        return content
    if isinstance(content, (list, tuple)) and content and isinstance(content[0], dict):
        for item in content:
            for k in ("image", "data", "png", "bytes"):
                if k in item:
                    return _as_array(item[k])
        return None
    if isinstance(content, dict):
        for k in ("image", "data", "png", "bytes", "path"):
            if k in content:
                return _as_array(content[k])
        return None
    if isinstance(content, (bytes, bytearray)):
        import cv2
        return cv2.imdecode(np.frombuffer(bytes(content), np.uint8), cv2.IMREAD_COLOR)
    if isinstance(content, str):
        import base64
        import cv2
        if os.path.exists(content):
            return cv2.imread(content)
        try:
            raw = base64.b64decode(content.split(",")[-1], validate=False)
            return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None
    return None


def _write(a, rec) -> None:
    out = a.report or (a.out / "GEN-REPORT.json")
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({"phase": "report", "path": str(out), "verdict": rec.get("verdict")}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
