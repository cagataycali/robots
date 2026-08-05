"""Measure how each training backend answers an unusable reproducibility seed."""

from __future__ import annotations

import json
import math
import pathlib
import sys
import tempfile

import strands_robots.training.base as _base

TREE = str(pathlib.Path(_base.__file__).parents[2])

from strands_robots.training.base import TrainSpec  # noqa: E402
from strands_robots.training.cosmos3 import Cosmos3Trainer  # noqa: E402
from strands_robots.training.lerobot import LerobotTrainer  # noqa: E402
from strands_robots.training.rl.base_algo import RLTrainSpec  # noqa: E402
from strands_robots.training.rl.fast_sac import FastSacTrainer  # noqa: E402
from strands_robots.training.rl.ppo import PpoTrainer  # noqa: E402

NAMES_SEED = ": seed "
PROBES = [
    ("-1", -1),
    ("-5", -5),
    ("True", True),
    ("2.7", 2.7),
    ("3.0", 3.0),
    ("nan", float("nan")),
    ("inf", float("inf")),
    ("'42'", "42"),
    ("[7]", [7]),
    ("0", 0),
    ("42", 42),
    ("None", None),
]


def verdict(trainer, spec, value):  # noqa: ANN001, ANN201
    """refused / accepted / raised - classified by the question being asked."""
    spec.seed = value
    try:
        problems = trainer.validate(spec)
    except BaseException as exc:  # noqa: BLE001 - an escape past a -> list contract IS the finding
        return f"raised {type(exc).__name__}"
    return "refused" if any(NAMES_SEED in p for p in problems) else "accepted"


def main() -> None:  # noqa: D103
    tmp = pathlib.Path(tempfile.mkdtemp())
    rows: dict[str, dict[str, str]] = {}
    for label, value in PROBES:
        rows[label] = {}
        for name, trainer, spec in (
            ("lerobot", LerobotTrainer(), TrainSpec(dataset_root=str(tmp / "ds"), output_dir=str(tmp / "o"))),
            ("cosmos3", Cosmos3Trainer(), TrainSpec(dataset_root=str(tmp / "ds"), output_dir=str(tmp / "o"))),
            ("fast_sac", FastSacTrainer(), RLTrainSpec(output_dir=str(tmp / "o"), env_factory=lambda: None)),
            ("ppo", PpoTrainer(), RLTrainSpec(output_dir=str(tmp / "o"), env_factory=lambda: None)),
        ):
            rows[label][name] = verdict(trainer, spec, value)

    # What actually reaches the wire for a usable seed - the no-regression claim.
    good = TrainSpec(dataset_root=str(tmp / "ds"), output_dir=str(tmp / "o"), base_model="x", steps=10, seed=42)
    good.extra["sft_toml"] = str(tmp / "r.toml")
    control = {
        "cosmos_override": [o for o in Cosmos3Trainer().build_overrides(good) if "seed" in o],
        "lerobot_argv": [t for t in LerobotTrainer().build_command(good) if "seed" in t],
    }

    # The consequence torch's modulo produces, independent of the tree.
    import torch

    collide = {}
    for supplied, actual in ((-1, 2**64 - 1), (-5, 2**64 - 5), (True, 1), (2.7, 2)):
        torch.manual_seed(supplied)
        a = [round(x, 6) for x in torch.rand(4).tolist()]
        torch.manual_seed(actual)
        b = [round(x, 6) for x in torch.rand(4).tolist()]
        collide[str(supplied)] = {"actual": actual, "draws": a, "identical": a == b}

    out = {"tree": TREE, "rows": rows, "control": control, "collide": collide}
    pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print("TREE:", TREE)
    for label in rows:
        print(f"  seed={label:6} " + "  ".join(f"{k}={v}" for k, v in rows[label].items()))
    print("control:", control)
    print("collide:", {k: v["identical"] for k, v in collide.items()})
    assert not math.isnan(0.0)


main()
