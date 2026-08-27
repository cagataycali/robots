#!/usr/bin/env python3
"""Fetch a microduck_rl HF-Jobs export and evaluate it here - the full loop.

`microduck_rl` submitted to HF Jobs trains on a managed GPU and, on exit, its
container uploads the exported policy to ``exported/policy.onnx`` in the run's
model repo (see scripts/hf/hf_jobs.py). This script downloads that ONNX and
hands it to :mod:`examples.microduck.eval_rl_policy`, so a training run that
finished in the cloud is evaluated in sim on this machine with one command::

    export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
    python examples/microduck/eval_from_hf.py \
        --repo cagataydev/microduck-walk-e2e-20260827-1849 --record

That closes the train -> export -> eval loop end to end: mjlab on HF Jobs ->
self-describing ONNX -> MicroduckPolicy rollout + LeRobot dataset here.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from huggingface_hub import hf_hub_download


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="HF model repo id of the run")
    ap.add_argument("--filename", default="exported/policy.onnx", help="ONNX path in the repo")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--vx", type=float, default=0.25)
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    print(f"downloading {args.filename} from {args.repo} ...")
    try:
        onnx = hf_hub_download(repo_id=args.repo, filename=args.filename)
    except Exception as exc:  # noqa: BLE001 - the run may not have exported yet
        raise SystemExit(
            f"could not fetch {args.filename} from {args.repo}: {exc}\n"
            "The training job may still be running - it uploads the export on exit."
        )
    print(f"got {onnx}")

    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable, os.path.join(here, "eval_rl_policy.py"),
        "--onnx", onnx, "--steps", str(args.steps), "--vx", str(args.vx),
    ]
    if args.record:
        cmd.append("--record")
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
