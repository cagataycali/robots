"""VERA rollout critic — Cosmos3-Nano reasons about a rollout video to find bugs.

Usage:
    python vera_critique.py <video.mp4> [more.mp4 ...]

Requires a Cosmos3-Nano reasoner serving on :8000 (see strands-cosmos
`c3-serve-reason`). Videos must live under an allowed workspace
(/tmp/vera-critique or COSMOS_VIDEO_WORKSPACE, or set COSMOS_WORKSPACE).
"""

import os
import shutil
import sys

from strands import Agent
from strands_cosmos import Cosmos3ReasonerModel

BASE_URL = os.environ.get("COSMOS_BASE_URL", "http://localhost:8000/v1")
WORKSPACE = os.environ.get("COSMOS_VIDEO_WORKSPACE", "/tmp/vera-critique")

PROMPT = (
    "You are a robotics QA reviewer grading a VERA video-to-action policy rollout "
    "in MuJoCo. <video>{path}</video>\n\n"
    "Be specific and terse:\n"
    "1. MOTION: does the arm/pusher actually move, and is it smooth or jittery?\n"
    "2. GOAL: is the motion purposeful toward the manipulation goal?\n"
    "3. BUGS: list any of — frozen frames, teleporting, clipping, physics blow-up, "
    "arm static, gripper never actuating, object never contacted.\n"
    "4. FIX: one concrete code/param change to improve the next rollout.\n"
    "End with a verdict line: 'VERDICT: PASS' or 'VERDICT: NEEDS-WORK'."
)


def critique(path: str) -> str:
    os.makedirs(WORKSPACE, exist_ok=True)
    local = os.path.join(WORKSPACE, os.path.basename(path))
    if os.path.abspath(path) != os.path.abspath(local):
        shutil.copy(path, local)
    agent = Agent(model=Cosmos3ReasonerModel(base_url=BASE_URL, model_id="nvidia/Cosmos3-Nano"))
    return str(agent(PROMPT.format(path=local)))


def main() -> int:
    videos = sys.argv[1:] or [
        "docs/assets/vera/mimicgen_panda.mp4",
    ]
    for v in videos:
        if not os.path.exists(v):
            print(f"!! missing: {v}")
            continue
        print("\n" + "=" * 70 + f"\n### {os.path.basename(v)}\n" + "=" * 70)
        try:
            print(critique(v))
        except Exception as e:
            print(f"[error] {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
