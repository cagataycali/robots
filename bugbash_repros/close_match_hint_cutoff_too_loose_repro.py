"""Repro: close_match_hint's cutoff=0.4 produces misleading "Did you mean"
suggestions for semantically-unrelated inputs.

The MuJoCo sim tool is the agent-facing surface. Every unknown-entity refusal
(action/object/camera/robot/parameter) routes through
``simulation/base.py:close_match_hint``, which calls
``difflib.get_close_matches(..., cutoff=0.4)``.

The Python stdlib default is 0.6. Every OTHER difflib call in strands_robots
uses 0.5-0.8 (see ``_MISSPELLING_RATIO = 0.8`` ten lines above the hint helper
in the same module). The 0.4 cutoff is a lone outlier.

Consequence: short/semantically-unrelated inputs - exactly what an LLM reading
the README's hero line "pick up the red cube" produces as its first guess -
get authoritative-looking but wrong suggestions:

    robot(action="pick")   -> "Did you mean: run_policy, stop_policy, eval_policy?"
    robot(action="grab")   -> "Did you mean: set_gravity?"
    robot(action="grasp")  -> "Did you mean: raycast, step, reset?"
    robot(action="help")   -> "Did you mean: step, eval_policy?"
    robot(action="place")  -> "Did you mean: load_scene, apply_force, replace_scene_mjcf?"

None of these are what the caller meant. The LLM/user takes the first
suggestion and heads down a garden path, often into actively destructive
sibling actions (``stop_policy`` kills the running controller).

Expected: no suggestion for a name that has no close match, OR a tight-cutoff
suggestion only. The pointer at the published enum ("77 actions in the
'action' enum of its schema; see tool_spec") already carries recovery for
no-match inputs - it is the second half of the message and is unaffected by
the cutoff.

Verified against upstream c5d2a5c2c (strands-labs/robots main @ 2026-10-05).
"""

from __future__ import annotations

import os
import sys

# Keep MuJoCo headless so this reproduces without a display server.
os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot  # noqa: E402


MISLEADING_PROBES = [
    # (input, verdict about the suggestion that is actually returned)
    ("pick", "the README hero example 'pick up the red cube' - run_policy is "
             "the closest an LLM would recognise as 'picking', but stop_policy "
             "and eval_policy are both offered with equal weight; stop_policy "
             "kills a running controller"),
    ("grab", "returns 'set_gravity'; grab and gravity share two letters but no "
             "semantic overlap - taking the suggestion flips gravity to zero"),
    ("grasp", "returns 'raycast, step, reset' - none manipulate the scene; "
              "reset wipes any progress the caller has made"),
    ("help", "returns 'step, eval_policy' - neither prints help; step advances "
             "physics with no observation returned, eval_policy requires "
             "a prepared episode"),
    ("place", "returns 'load_scene, apply_force, replace_scene_mjcf' - all "
              "destructive; the caller probably meant add_object / move_object"),
    ("spawn", "returns 'step' - unrelated"),
]


def main() -> int:
    """Run the probes and report the misleading suggestions."""
    robot = Robot("so100")
    print(
        "close_match_hint cutoff=0.4 produces misleading 'Did you mean' "
        f"suggestions in {Robot.__module__}.Robot('so100'):\n"
    )
    any_misleading = False
    for probe, note in MISLEADING_PROBES:
        result = robot(action=probe)
        text = result["content"][0]["text"]
        print(f"  robot(action={probe!r})")
        # Extract the 'Did you mean: ...?' fragment.
        import re
        match = re.search(r"Did you mean: ([^?]+)\?", text)
        suggestion = match.group(1) if match else "<none>"
        print(f"    -> Did you mean: {suggestion}")
        print(f"    note: {note}")
        print()
        if match is not None:
            # Any 'Did you mean' at all on these inputs is a misfire.
            any_misleading = True
    robot.cleanup()
    if any_misleading:
        print(
            "DEFECT CONFIRMED: at least one probe above received a suggestion "
            "that does not match the caller's intent.\n"
            "\n"
            "Fix: raise close_match_hint's cutoff from 0.4 to the stdlib "
            "default (0.6), matching every other difflib site in the codebase "
            "(see tests/simulation/mujoco/test_unknown_action_message_suggests_a_published_action.py "
            "for the one-edit-typo pin that keeps passing)."
        )
        return 1
    print("NO DEFECT: every probe refused without a suggestion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
