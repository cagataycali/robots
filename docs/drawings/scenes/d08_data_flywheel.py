"""D8: the data flywheel. record -> verify -> label -> train -> deploy -> measure, back to record.

Sources: strands_robots/dataset_recorder.py, verify_dataset.py, episode_labels.py + tools/episode_judge.py,
training/factory.py create_trainer + training/base.py TrainSpec, policies/factory.py create_policy,
simulation/base.py run_policy + eval_policy, docs/hooks/data/checkpoints.json.
"""
from scene import Scene

BW, BH = 300, 108
XS = (60, 450, 840)
TOP, BOT = 150, 392


def scene() -> Scene:
    s = Scene(
        "d08_data_flywheel",
        "Record, train, deploy, measure, record again",
        "Six steps in a loop; one Robot object and one run_policy call carry every one of them, and a number leaves with its source.",
        "Six cards in a loop. Top row, left to right: record, DatasetRecorder and start_recording, episodes on "
        "disk; verify, verify_dataset, frames and videos check out; label, predicates and a judge agent, a "
        "verdict each. A wire down the right side, the episodes worth training on, to the bottom row read "
        "right to left: train, create_trainer from a TrainSpec, a checkpoint; deploy, the one green element, "
        "create_policy then run_policy, sim first and then the gate; measure, eval_policy and labels, a rate "
        "with its source. A wire up the left side, what to record next, closes the loop. Footnote: a number "
        "leaves this loop only with the dataset and the checkpoint that produced it.",
        h=580,
    )
    s.section(XS[0], 122, "data")
    s.section(XS[0], BOT - 28, "policy")
    steps = [
        (XS[0], TOP, "record", "DatasetRecorder, start_recording on the robot; teleop or a policy drives", "episodes on disk", False),
        (XS[1], TOP, "verify", "verify_dataset reads the parquet and the videos back", "frames and videos check out", False),
        (XS[2], TOP, "label", "predicates over the frames, then a judge agent", "a verdict per episode", False),
        (XS[2], BOT, "train", "create_trainer(TrainSpec) on the labelled episodes", "a checkpoint", False),
        (XS[1], BOT, "deploy", "create_policy, then run_policy: the simulator first, then the gate and the arm", "sim first, then the gate", True),
        (XS[0], BOT, "measure", "eval_policy in the simulator, labels on the arm", "a rate with its source", False),
    ]
    for x, y, title, sub, out, accent in steps:
        s.box(x, y, BW, BH, title, sub, accent=accent, size=14, subsize=12)
        s.chips(x + 14, y + BH - 34, [out])
    # top row, left to right
    s.arrow([(XS[0] + BW, TOP + 54), (XS[1], TOP + 54)])
    s.arrow([(XS[1] + BW, TOP + 54), (XS[2], TOP + 54)])
    # down the right
    s.arrow([(XS[2] + 150, TOP + BH), (XS[2] + 150, BOT)], label="the episodes worth training on",
            label_dx=-10, label_dy=4, label_anchor="end")
    # bottom row, right to left
    s.arrow([(XS[2], BOT + 54), (XS[1] + BW, BOT + 54)])
    s.arrow([(XS[1], BOT + 54), (XS[0] + BW, BOT + 54)])
    # up the left
    s.arrow([(XS[0] + 150, BOT), (XS[0] + 150, TOP + BH)], label="what to record next", label_dx=10, label_dy=4)
    s.text(600, 296, "one Robot object and one run_policy call carry every step", cls="mono muted", size=10.5,
           anchor="middle")

    s.footnote(552, "a number leaves this loop only with the dataset and the checkpoint that produced it.")
    return s
