"""D13: the trainer lifecycle. One TrainSpec, five calls, a path create_policy can load.

Top: TrainSpec, the one dataclass every supervised backend reads. A row of five cards, visited in
order: validate (pure preflight), prepare (optional), train (TrainResult), status (the RUNNING != learning
verdict), export (the one accent element: the artifact that leaves training). Under the row the
TrainResult fields and the metrics that tell a live process from a learning one. Bottom: create_policy
loads the exported path, and the same trainer names come back through train_policy for an agent.
Every name is on learn/training/index.md.
"""
from scene import Scene

X, W = 60, 1080
N = 5
GAP = 14
BW = (W - GAP * (N - 1)) / N
ROW = 282


def scene() -> Scene:
    s = Scene(
        "d13_trainer_lifecycle",
        "One TrainSpec, validate to export",
        "Every trainer takes the same spec and answers the same five calls; the last one hands a path to create_policy.",
        "Top card: TrainSpec, one dataclass for every supervised backend, with dataset_root or dataset_repo_id, "
        "base_model, output_dir, embodiment, steps, global_batch_size, learning_rate, method and extra. A row of "
        "five cards the spec passes through in order: validate, a pure preflight that launches nothing and "
        "returns an empty list when launchable; prepare, optional, where cosmos3 converts its base checkpoint; "
        "train, which returns a TrainResult; status, which reads the run's logs for the verdict that RUNNING is "
        "not learning; export, the one green card, which writes a path create_policy can load. Under the row: "
        "TrainResult carries status (success, running, error), job_id, checkpoint_dir, exported_model, metrics "
        "and message; the metrics latest_loss, latest_step, learning and liveness_ok tell a live process from a "
        "learning one. Bottom: create_policy loads the exported path on the simulator and the arm, and "
        "train_policy offers the same trainers to an agent. Footnote: the trainer that fits is picked by name "
        "from list_trainers; mock runs the whole lifecycle on a laptop.",
        h=666,
    )

    # ---------------------------------------------------------------- the spec
    s.section(X, 122, "the spec")
    s.box(X, 134, W, 96, "TrainSpec",
          "one dataclass for every supervised backend; RLTrainSpec adds the reward-driven fields",
          size=14, subsize=12)
    s.chips(X + 14, 192, ["dataset_root", "dataset_repo_id", "base_model", "output_dir", "embodiment", "steps",
                          "global_batch_size", "learning_rate", 'method="full"', "extra"])
    s.down(600, 230, ROW - 18, id="spec")

    # ---------------------------------------------------------------- the five calls
    s.section(X, ROW - 32, "the lifecycle, in order")
    s.arrow([(X + BW / 2, ROW - 18), (X + W - BW / 2, ROW - 18)], head=False)
    steps = [
        ("validate", "pure preflight; launches nothing; an empty list means launchable", "problems: list", False),
        ("prepare", "optional; cosmos3 converts the base checkpoint here", "once per base model", False),
        ("train", "launches the run and returns a TrainResult", "TrainResult", False),
        ("status", "reads the run's logs: RUNNING is not learning", "status(job_id)", False),
        ("export", "a path create_policy can load, sim or arm", "artifact", True),
    ]
    for i, (name, sub, out, accent) in enumerate(steps):
        x = X + i * (BW + GAP)
        s.down(x + BW / 2, ROW - 18, ROW, id=f"to_{name}")
        s.box(x, ROW, BW, 124, f"trainer.{name}", sub, accent=accent, size=13.5, subsize=11.5, id=name)
        s.chips(x + 14, ROW + 90, [out])
        s.motion.append((name, "visit"))

    # ---------------------------------------------------------------- the result
    s.box(X, 428, W, 90, None, None, dashed=True)
    s.text(X + 14, 450, "TRAINRESULT, AND THE METRICS THAT TELL A LIVE PROCESS FROM A LEARNING ONE",
           cls="mono muted", size=10.5, spacing="0.05em")
    s.chips(X + 14, 464, ["status: success | running | error", "job_id", "checkpoint_dir", "exported_model", "message"])
    s.chips(X + 14, 492, ["latest_loss", "latest_step", "learning", "liveness_ok"])

    # ---------------------------------------------------------------- what leaves
    s.down(X + W - BW / 2, 406, 428, id="to_result")
    s.down(600, 518, 544, id="to_policy")
    s.box(X, 544, W, 64, "create_policy(..., checkpoint_dir=...)",
          "the exported artifact runs on the simulator first, then behind the gate on the arm; train_policy offers "
          "the same trainers to an agent", size=14, subsize=12)
    s.motion += [("spec", "flow"), ("to_result", "flow"), ("to_policy", "flow")]

    s.footnote(646, "pick the trainer by name from list_trainers; mock runs the whole lifecycle on a laptop, "
                    "hardware_floor says what the others need.")
    return s
