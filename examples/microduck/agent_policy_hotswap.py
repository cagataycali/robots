#!/usr/bin/env python3
"""A Strands agent hot-swaps the Microduck's locomotion policies mid-session.

The richer end-to-end test: one :class:`strands.Agent`, given ``Robot("microduck")``
as a tool, is asked to run the duck through a *sequence* of behaviours - walk
forward, then hold a stand - by swapping which ONNX policy drives the same
``run_policy`` seam between phases. Nothing about the robot changes; only the
policy behind ``policy_provider="microduck"`` does, which is exactly how a
locomotion stack is exercised in practice.

Both policies share the 61-D locomotion observation, so the swap is a pure
policy change. Pollen's *skill* policies (sit/stand toggle, kicks, ground-pick,
roulade) are a different, narrower interface - on real hardware they are driven
as ``robot.do`` skills through the native ``mode="real"`` driver, not as a
locomotion policy - so this sim demo stays on the two locomotion policies it can
honestly run.

Run::

    export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
    python examples/microduck/agent_policy_hotswap.py \
        --policy-dir "$PWD/../microduck/policies"

Dependencies:
  pip install "strands-robots[sim-mujoco,microduck]" strands-agents
  A strands-agents model provider (Bedrock by default).
"""

from __future__ import annotations

import argparse
import os

from strands import Agent

from strands_robots import Robot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--policy-dir",
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../microduck/policies")
        ),
        help="Directory holding alpha_walking.onnx and alpha_stand.onnx",
    )
    ap.add_argument("--steps", type=int, default=120, help="control steps per phase")
    args = ap.parse_args()

    walk = os.path.join(args.policy_dir, "alpha_walking.onnx")
    stand = os.path.join(args.policy_dir, "alpha_stand.onnx")

    sim = Robot("microduck", mesh=False)
    agent = Agent(tools=[sim])

    prompt = f"""You operate a Pollen Microduck biped in a MuJoCo simulation via the microduck_sim tool.
Run it through a two-phase routine by HOT-SWAPPING the driving policy between phases. Both phases use
the same robot and the same run_policy call - only the onnx_path changes.

Phase 1 - WALK FORWARD:
  run a policy with policy_provider="microduck",
  policy_config={{"onnx_path": "{walk}"}},
  control_frequency=50, n_steps={args.steps},
  policy_kwargs={{"target_velocity": [0.25, 0.0, 0.0]}}.

Phase 2 - HOLD A STAND (swap the policy, do NOT change the robot):
  run a policy with policy_provider="microduck",
  policy_config={{"onnx_path": "{stand}"}},
  control_frequency=50, n_steps={args.steps},
  policy_kwargs={{"target_velocity": [0.0, 0.0, 0.0]}}.

After each phase, report the status, steps used, and action_errors from that rollout. At the end,
state plainly whether BOTH phases completed with zero action errors, confirming the mid-session
policy hot-swap worked."""

    result = agent(prompt)
    print("\n===== AGENT RESULT =====")
    print(result)


if __name__ == "__main__":
    main()
