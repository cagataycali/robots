"""
Microduck ONNX weight 404 has no 'Did you mean' hint — asymmetric with sibling
create_policy() refusal in the same package.

Repro: a one-letter typo in `onnx_path` falls through the Hub lookup and raises
a bare 404 that re-dumps the HF URL. The valid file list (nine shipped weights)
is documented in docs/learn/policies/microduck.md line 15 and is also
discoverable at runtime via huggingface_hub.list_repo_files, but the refusal
cites neither — the user has to pattern-match the typo themselves.

Compare: strands_robots/policies/factory.py:399-400 does the exact difflib
.get_close_matches dance for a wrong provider name; same package, same import.

Expected: refusal names the nearest shipped weight(s), e.g.
    "Microduck ONNX policy 'alpha_walkinng.onnx' not found on
     pollen-robotics/microduck-policies. Did you mean: 'alpha_walking.onnx'?"

Actual: a raw hf_hub 404 is wrapped and re-raised, no hint, no list.
"""
import os
import sys

# devduck shim: do not forward the host SYSTEM_PROMPT into the subprocess env.
os.environ.pop("SYSTEM_PROMPT", None)

from strands_robots.simulation import create_simulation


def main() -> int:
    sim = create_simulation("mujoco")
    sim.create_world()
    sim.add_robot("microduck")

    try:
        result = sim.run_policy(
            robot_name="microduck",
            policy_provider="microduck",
            policy_config={"onnx_path": "alpha_walkinng.onnx"},  # one-letter typo
            duration=0.5,
            control_frequency=50.0,
        )
    finally:
        sim.cleanup()

    text = result["content"][0]["text"]
    assert result["status"] == "error", f"expected error, got {result['status']}"
    assert "alpha_walkinng.onnx" in text, "the typo should at least appear"
    # The paper-cut: no mention of the correct name, no 'Did you mean' hint.
    assert "did you mean" not in text.lower(), "if this starts failing the fix landed"
    assert "alpha_walking" not in text, (
        "the valid neighbour is never named, even though difflib ships in stdlib "
        "and sibling factory.py:399 uses it on the same import"
    )
    print("REPRODUCED — no hint in refusal:")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
