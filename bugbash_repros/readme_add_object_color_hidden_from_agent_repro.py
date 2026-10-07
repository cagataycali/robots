"""
README quickstart (strands-labs/robots:README.md L50-57) passes
`color=[1.0, 0.0, 0.0]` to `add_object`, but the colour is silently dropped
from every agent-visible discovery surface:

  * `tool_spec.description` (LLM hot path) names only the object identifier.
  * `add_object` success text names `shape at pos, mass` — no colour.
  * `list_objects` enumerates `name: shape at pos, mass` — no colour.
  * `get_body_state(body_name=...)` returns pose/vel/mass — no colour.

A new user reading the hero snippet reasonably believes passing `color` lets
the agent resolve "the red cube". Reality: the agent only sees the NAME
`red_cube`; the colour kwarg is a rendering-only knob. Rename the object to
`cube` or add two colored boxes with neutral names and the agent is blind.

This repro uses NEUTRAL object names (`box_a`, `box_b`, `box_c`) so the
colour signal cannot leak via the identifier string, isolating the kwarg's
observable effect from the name's.

Expected: at least one agent-visible surface emits each object's colour so
`color=` has observable effect on grounding ("pick up the red one").

Actual: `color` lives only on `SimObject.color`
(strands_robots/simulation/models.py:148) and reaches the renderer; no text
surface emits it. RC=1 pre-fix, RC=0 once a surface names it.

Run:
  MUJOCO_GL=egl python bugbash_repros/readme_add_object_color_hidden_from_agent_repro.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot

# Neutral names: the colour has to come from the `color` kwarg, not the name.
BOXES = [
    ("box_a", [1.0, 0.0, 0.0]),  # red
    ("box_b", [0.0, 1.0, 0.0]),  # green
    ("box_c", [0.0, 0.0, 1.0]),  # blue
]

COLOUR_WORDS = ("red", "green", "blue", "colour", "color", "rgb", "rgba")


def _any_colour_signal(text: str) -> bool:
    """True if the surface emits any colour signal — words or RGB floats."""
    low = text.lower()
    if any(w in low for w in COLOUR_WORDS):
        return True
    # Numeric colour vector in text form (e.g. "[1.0, 0.0, 0.0]").
    for name, rgb in BOXES:
        if f"[{rgb[0]}, {rgb[1]}, {rgb[2]}]" in text:
            return True
    return False


def main() -> int:
    r = Robot("so100")
    for name, rgb in BOXES:
        res = r.add_object(
            name=name, shape="box", size=[0.05, 0.05, 0.05],
            position=[0.0, -0.2, 0.025], color=rgb,
        )
        if res.get("status") != "success":
            print(f"SETUP FAIL: add_object({name=}) -> {res}")
            return 2

    failures: list[str] = []

    # 1) add_object success text — the first thing an agent reads after it
    #    places the object. User passed colour; the response should name it.
    one = r.add_object(
        name="box_probe", shape="box", size=[0.05]*3,
        position=[0.2, -0.2, 0.025], color=[1.0, 0.0, 0.0],
    )
    text = one["content"][0]["text"]
    if not _any_colour_signal(text):
        failures.append(f"add_object success text omits colour: {text!r}")

    # 2) list_objects — the agent-callable catalogue of what's in the scene.
    lo = r.list_objects()
    text = lo["content"][0]["text"]
    if not _any_colour_signal(text):
        failures.append(f"list_objects omits colour on every object:\n{text}")

    # 3) tool_spec.description — the hot path (read before the first call).
    desc = r.tool_spec["description"]
    scene_start = desc.find("scene also holds")
    scene_slice = desc[scene_start:scene_start + 400] if scene_start >= 0 else desc
    if not _any_colour_signal(scene_slice):
        failures.append(
            "tool_spec.description (LLM hot path) names object identifiers "
            "but not their colours. Agent asked 'pick up the red one' with "
            "neutral names sees no colour anywhere.\n"
            f"Scene sentence: {scene_slice!r}"
        )

    # 4) get_body_state — the next natural call after list_objects for pose/size.
    gs = r(action="get_body_state", body_name="box_a")
    text = str(gs)
    if not _any_colour_signal(text):
        failures.append(
            "get_body_state('box_a') payload omits colour (pose/vel/mass only).\n"
            f"Payload: {text[:300]!r}"
        )

    if failures:
        print("DEFECT CONFIRMED — the `color` kwarg has NO observable effect on")
        print("agent-visible discovery. Four surfaces checked, each silent on colour:")
        print()
        for i, f in enumerate(failures, 1):
            print(f"  [{i}] {f}")
            print()
        print("A user who names boxes neutrally and prompts 'pick up the red one'")
        print("has given the agent NOTHING to ground 'red' against.")
        return 1

    print("OK — at least one surface emits colour. Fix landed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
