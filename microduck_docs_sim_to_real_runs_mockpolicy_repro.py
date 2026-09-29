"""Repro: docs/learn/hardware/microduck.md "Sim to real" sketch runs MockPolicy.

The docs sketch on line 49 reads::

    sim = Robot("microduck")
    sim.run_policy("microduck", instruction="walk forward")   # the provider self-configures from the ONNX metadata

`Robot("microduck")` returns a MuJoCoSimEngine. Its `run_policy` signature is:

    run_policy(robot_name=None, policy_provider="mock", policy_config=None,
               instruction="", duration=10.0, ...)

so the docs' positional `"microduck"` binds to `robot_name` and
`policy_provider` keeps its default value `"mock"`. The sketch therefore
runs `MockPolicy`, not `MicroduckPolicy`, in the same file whose next
paragraph advertises that "the on-robot policy is the same
alpha_walking.onnx the [microduck] extra runs in MuJoCo (byte-compatible,
difference 0.0), so a sim rollout with equal observations predicts the
hardware". The rollout still reports `status="success"` and the runtime
notes MockPolicy in the trailing text (`MockPolicy | walk forward`), but a
reader who copies the sketch has verified no sim-to-real parity.

Every other doc in the tree spells `run_policy` with keyword arguments
(`robot_name="..."`, `policy_provider="..."`, `policy_config={...}`):
docs/learn/policies/microduck.md uses that form twice on the same page
(lines 58, 91). This hardware sketch is the sole positional-only holdout.

Expected:
    Either
    * spell the sketch as the sibling doc does, naming
      `policy_provider="microduck"` and `policy_config={"onnx_path":
      "alpha_walking.onnx"}` so the "same ONNX" claim the paragraph makes
      is what the sketch runs; or
    * remove the sketch and cite `docs/learn/policies/microduck.md`
      instead, which already documents the correct call.

Run:
    python microduck_docs_sim_to_real_runs_mockpolicy_repro.py
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot


def main() -> None:
    sim = Robot("microduck")

    result = sim.run_policy("microduck", instruction="walk forward")
    assert result["status"] == "success", result

    content = result["content"][0]["text"]
    print("=== sketch output ===")
    print(content)
    print()
    assert "MockPolicy" in content, content
    print(
        "REPRO: the sketch that follows a paragraph titled 'Sim to real' and "
        "claims 'the on-robot policy is the same alpha_walking.onnx' has just "
        "run MockPolicy (see the 'MockPolicy | walk forward' line above). "
        "MicroduckPolicy was never constructed."
    )
    print()
    print("Fix (matches docs/learn/policies/microduck.md line 91):")
    print(
        "  sim.run_policy(\n"
        "      robot_name='microduck',\n"
        "      policy_provider='microduck',\n"
        "      policy_config={'onnx_path': 'alpha_walking.onnx'},\n"
        "      instruction='walk forward',\n"
        "      duration=3.0,\n"
        "      control_frequency=50.0,\n"
        "  )"
    )


if __name__ == "__main__":
    main()
