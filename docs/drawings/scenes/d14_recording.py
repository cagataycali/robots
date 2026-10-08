"""D14: how a frame becomes a dataset. One DatasetRecorder under the simulation and the hardware script.

Left: the two ways in, a sim session (start_recording, run_policy, stop_recording) and a hardware loop
(add_frame per step, save_episode, finalize), both landing on the DatasetRecorder (the one accent
element). Right: the LeRobot v3 layout the recorder writes under resolve_dataset_dir, and
verify_dataset_episodes reading it back. Every name is on learn/data/record.md.
"""
from scene import Scene

L, LW = 60, 440
M, MW = 560, 220
R, RW = 840, 300


def scene() -> Scene:
    s = Scene(
        "d14_recording",
        "One recorder under every session",
        "A sim session or a hardware loop hands frames to DatasetRecorder; it writes the LeRobot v3 layout that verify_dataset_episodes reads back.",
        "Left column, two cards: a simulation session, start_recording declares the schema from the live model, "
        "run_policy records one (observation, action) frame per control step, stop_recording flushes and "
        "finalizes; and a hardware loop, DatasetRecorder.create with the joint names and camera keys, add_frame "
        "once per step, save_episode once per episode, finalize at the end. Both wires lead to the middle card, "
        "the one green element: DatasetRecorder, which refuses a frame missing a declared column and drops "
        "undeclared action keys. Right column: the directory resolve_dataset_dir(repo_id, root) answers with, "
        "under HF_LEROBOT_HOME: meta/info.json, meta/episodes parquet (the ground truth, one row per episode), "
        "data parquet (one row per frame), videos per camera as mp4, and the optional episode_labels.json the "
        "judge writes. Under it verify_dataset_episodes reads the layout back. Footnote: the task label "
        "describes the intent, not the motion; the mock's report says it never read the instruction.",
        h=600,
    )

    # ---------------------------------------------------------------- two ways in
    s.section(L, 122, "a sim session")
    s.box(L, 134, LW, 150, 'Robot("so101")', "one call opens the session; run_policy writes while it is open",
          size=14, subsize=11.5)
    s.chips(L + 14, 196, ["start_recording(repo_id, task, fps, cameras)"])
    s.chips(L + 14, 226, ["run_policy(..., n_episodes, reset_between)"])
    s.chips(L + 14, 256, ["save_episode", "stop_recording(push_to_hub)"])

    s.section(L, 336, "a hardware loop")
    s.box(L, 348, LW, 150, "your control loop", "the same class under any script that reads an arm",
          size=14, subsize=11.5)
    s.chips(L + 14, 410, ["DatasetRecorder.create(repo_id, fps, joint_names, ...)"])
    s.chips(L + 14, 440, ["add_frame(observation, action, task)"])
    s.chips(L + 14, 470, ["save_episode()", "finalize()", "resume(repo_id)"])

    # ---------------------------------------------------------------- the recorder (the one green element)
    s.arrow([(L + LW, 209), (M + MW / 2, 209), (M + MW / 2, 292)], id="from_sim")
    s.text(M + MW / 2 + 8, 250, "one frame per step", cls="mono muted", size=10.5)
    s.arrow([(L + LW, 423), (M + MW / 2, 423), (M + MW / 2, 340)], id="from_loop")
    s.box(M, 292, MW, 48, "DatasetRecorder", None, accent=True, size=14, id="recorder")
    s.para(M, 452, "refuses a frame missing a declared column; drops undeclared action keys",
           MW, size=11.5, cls="grot muted")

    # ---------------------------------------------------------------- what it writes
    s.arrow([(M + MW, 316), (R, 316)], id="writes")
    s.text((M + MW + R) / 2, 306, "writes", cls="mono muted", size=10.5, anchor="middle")
    s.section(R, 122, "resolve_dataset_dir(repo_id, root)")
    s.box(R, 134, RW, 250, "$HF_LEROBOT_HOME/<repo_id>", "the LeRobot v3 layout", size=13.5, subsize=12)
    rows = [
        ("meta/info.json", "fps, features, totals"),
        ("meta/episodes/**.parquet", "the ground truth, one row per episode"),
        ("data/**.parquet", "one row per frame"),
        ("videos/<camera>/**.mp4", "observation.images.<camera>"),
        ("episode_labels.json", "optional, written by the judge"),
    ]
    y = 204
    for name, what in rows:
        s.text(R + 14, y, name, cls="mono fg", size=11.5)
        s.text(R + 14, y + 15, what, cls="grot muted", size=11)
        y += 36
    s.down(R + RW / 2, 384, 420, label="read back", label_dx=10, label_dy=4, id="verify")
    s.box(R, 420, RW, 78, "verify_dataset_episodes(expected=5)",
          "counts episodes and frames, decodes a video, checks every stats vector", size=13.5, subsize=12)
    s.motion = [("recorder", "pulse"), ("from_sim", "flow"), ("from_loop", "flow"), ("writes", "flow"), ("verify", "flow")]

    s.footnote(566, "the task label describes the intent, not the motion; swap in a real provider or a teleoperator "
                    "before training.")
    return s
