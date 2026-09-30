"""D8: the data flywheel. record -> verify -> label -> train -> deploy -> measure, back to record.

Sources: strands_robots/dataset_recorder.py, verify_dataset.py, episode_labels.py + tools/episode_judge.py,
training/factory.py create_trainer + training/base.py TrainSpec/Trainer.validate, policies/factory.py
create_policy, simulation/base.py run_policy + eval_policy, docs/hooks/data/checkpoints.json.
"""

from excal import MUTED, Drawing

d = Drawing(
    "d08_data_flywheel",
    "Six steps in a loop: record episodes with DatasetRecorder, verify the parquet with verify_dataset, "
    "label episodes with predicates and a judge, train a checkpoint with create_trainer from a TrainSpec, "
    "deploy it with create_policy and run_policy in the simulator and then behind the gate on the arm, "
    "measure with eval_policy or labels; the measured number returns as the next recording target.",
)

steps = [
    (60, 60, "record", "DatasetRecorder, start_recording", "episodes on disk"),
    (430, 60, "verify", "verify_dataset", "frames and videos check out"),
    (800, 60, "label", "predicates + judge agent", "a verdict each"),
    (800, 300, "train", "create_trainer(TrainSpec)", "a checkpoint"),
    (430, 300, "deploy", "create_policy -> run_policy", "sim first, then the gate"),
    (60, 300, "measure", "eval_policy, labels", "a rate with its source"),
]
boxes = []
for x, y, title, sub, out in steps:
    kind = "accent" if title == "deploy" else "plain"
    b = d.box(x, y, 300, 72, title, kind=kind, sub=sub, size=18)
    d.caption(x, y + 82, out)
    boxes.append(b)
rec, ver, lab, tr, dep, mea = boxes
d.arrow(rec, "r", ver, "l")
d.arrow(ver, "r", lab, "l")
d.arrow(lab, "b", tr, "t", "the episodes worth training on", off_a=0.4, off_b=0.4, label_dy=-4)
d.arrow(tr, "l", dep, "r")
d.arrow(dep, "l", mea, "r")
d.arrow(mea, "t", rec, "b", "what to record next", off_a=0.4, off_b=0.4, label_dy=-4)

d.text(430, 200, "one Robot object and one run_policy call carry every step", size=14, color=MUTED)
d.caption(60, 440, "a number leaves this loop only with the dataset and the checkpoint that produced it.")
d.save()
