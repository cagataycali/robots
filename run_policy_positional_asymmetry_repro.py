"""Repro: docs/reference/api/robot.md:3 promises 'same agent-facing surface' for
run_policy across sim/real, but the two signatures have different first
positional parameters and this fails silently on the sim path.

Upstream cite:
  - docs/reference/api/robot.md:3
    "Both expose the same agent-facing surface: `act`, `observe`, `run_policy`, `cleanup`."
  - strands_robots/simulation/base.py:2946-2949
    SimEngine.run_policy(robot_name=None, policy_provider='mock', policy_config=None, ...)
  - strands_robots/hardware_robot.py:3208-3213
    HardwareRobot.run_policy(policy_object: Policy, instruction='', duration=30.0, ...)
  - docs/index.md:113
    result = robot.run_policy(policy, instruction="pick up the cube", duration=10.0)  # real
  - docs/index.md:99
    result = robot.run_policy(robot_name="so101", policy_provider="lerobot_local", ...)  # sim

Symptom: a user reading docs/index.md sees the two fences side-by-side under
'The same checkpoint, sim or real' and reasonably concludes that a portable
call surface exists. Reusing the real-fence call shape on a sim (or copy-paste
around a mode switch) produces:

  status=error, "Robot '<strands_robots.policies.mock.MockPolicy object at 0x...>' not found.
   Available robots: ['so101']."

The sim binds the Policy instance to `robot_name` (the sim's first positional)
without noticing it received a Policy object, so the error message names the
Python repr of the policy as if it were a robot name. There is no 'Did you
mean policy_object=?' hint even though the machine has the type information.

Run:
    cd /path/to/robots && python run_policy_positional_asymmetry_repro.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot
from strands_robots.policies import create_policy


def main() -> None:
    # Build a policy the way docs/index.md 'The same checkpoint, sim or real' shows on the right (mode='real') fence.
    policy = create_policy("mock")

    # Now invoke run_policy on a SIM engine with the SAME positional shape that
    # the mode='real' fence uses. Docs claim 'same agent-facing surface'.
    sim = Robot("so101", mode="sim")
    try:
        result = sim.run_policy(policy, instruction="pick up the cube", duration=1.0)
    finally:
        try:
            sim.cleanup()
        except Exception:
            pass

    status = result.get("status")
    text = str(result.get("content", ""))
    print(f"status = {status}")
    print(f"content = {text[:400]}")

    assert status == "error", f"expected error, got {status}"
    # The failure is a Policy instance leaking into a robot-name refusal
    # message, without any hint that policy_object= was probably intended.
    assert "MockPolicy object" in text, (
        "expected the sim to name the repr of the Policy in its 'robot not found' message"
    )
    assert "did you mean" not in text.lower(), (
        "regression: a helpful 'did you mean policy_object=' hint has appeared"
    )
    print("\nrepro OK: sim binds a Policy to `robot_name` and reports it as a missing robot.")


if __name__ == "__main__":
    main()
