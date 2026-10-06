"""docs/learn/agents.md:27 names 6 sim actions + 'the world API'; the sim's
own tool_spec (strands_robots/simulation/mujoco/simulation.py:7013-7035,
cached in ``_TOOL_SPEC_SCHEMA``) publishes 77. An LLM that treats the agents
table as the API contract never emits ``set_gripper`` / ``rotate_wrist`` /
``start_recording`` / ``apply_force`` etc. — including two of the three
motion primitives that the v0.5.0 release notes
(``docs/reference/changelog.md:25``) list by name as the headline features
for that version.

Prints the gap and exits 1 if more than ``world API``'s generous reading
cannot cover the missing actions. The ``world API`` hand-wave appears
nowhere else in docs/ (grep is clean) so this script holds it to its most
liberal reading: ``create_world`` plus every ``add_*``/``remove_*``/
``list_*``/``reset``/``get_state`` on world/robots/objects/cameras.

Pins:
- strands_robots/simulation/mujoco/simulation.py:7013 (``[Motion primitives]``
  line in the description — names ``set_gripper`` and ``rotate_wrist``).
- strands_robots/simulation/mujoco/simulation.py:7006 ("Actions (77 total):"
  — the integer the description leads with).
- docs/learn/agents.md:27 (the table row this script targets).
- docs/reference/changelog.md:25 (the v0.5.0 release-note sentence that
  names ``set_gripper`` and ``rotate_wrist`` as marquee features).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# -------------------------------------------------------------------------
# 1. Source of truth: the sim's tool_spec action enum.
# -------------------------------------------------------------------------
sys.path.insert(0, str(ROOT))
from strands_robots.simulation.mujoco.simulation import _TOOL_SPEC_SCHEMA

enum_actions: list[str] = _TOOL_SPEC_SCHEMA["properties"]["action"]["enum"]
assert enum_actions, "sim _TOOL_SPEC_SCHEMA.properties.action.enum is empty"

# -------------------------------------------------------------------------
# 2. What docs/learn/agents.md names on line 27.
# -------------------------------------------------------------------------
docs_path = ROOT / "docs" / "learn" / "agents.md"
line = docs_path.read_text(encoding="utf-8").splitlines()[26]  # line 27, 0-indexed
assert "mode=\"sim\"" in line, f"docs/learn/agents.md line 27 shape changed: {line!r}"

# Pull back-ticked identifiers from the Actions cell only.
# The row looks like: | `mode="sim"` ... | `<name>_sim` | `a`, `b`, ... and the world API |
cells = [c.strip() for c in line.strip().strip("|").split("|")]
assert len(cells) == 3, f"expected 3 cells on line 27, got {cells!r}"
actions_cell = cells[2]
docs_named = re.findall(r"`([a-z_]+)`", actions_cell)

# -------------------------------------------------------------------------
# 3. The ``world API`` hand-wave: give it its most liberal reading. Anything
#    that create_world/add_*/remove_*/list_*/reset/get_state could cover.
# -------------------------------------------------------------------------
WORLD_API_GENEROUS = {
    "create_world", "load_scene", "reset", "get_state", "destroy", "export_xml",
    "add_robot", "remove_robot", "list_robots", "list_bodies",
    "add_object", "remove_object", "move_object", "list_objects",
    "add_camera", "remove_camera", "list_cameras",
}
docs_covered = set(docs_named) | WORLD_API_GENEROUS

# -------------------------------------------------------------------------
# 4. v0.5.0 release-note marquee primitives from the changelog.
# -------------------------------------------------------------------------
changelog = (ROOT / "docs" / "reference" / "changelog.md").read_text(encoding="utf-8")
marquee_match = re.search(
    r"analytic motion primitives \(`([^`]+)`, `([^`]+)`, `([^`]+)`\)",
    changelog,
)
assert marquee_match, "v0.5.0 marquee-primitive sentence has moved in changelog.md"
marquee = list(marquee_match.groups())

# -------------------------------------------------------------------------
# 5. Report.
# -------------------------------------------------------------------------
missing = sorted(a for a in enum_actions if a not in docs_covered)
marquee_missing_from_docs = [m for m in marquee if m not in docs_covered and m in enum_actions]

print("=" * 70)
print("docs/learn/agents.md:27 vs sim _TOOL_SPEC_SCHEMA action enum")
print("=" * 70)
print(f"Published sim actions:           {len(enum_actions)}")
print(f"Docs row names (back-ticked):    {len(docs_named)} -> {docs_named}")
print(f"'world API' generous reading:    {len(WORLD_API_GENEROUS)} actions")
print(f"Together, docs cover:            {len(docs_covered & set(enum_actions))}/{len(enum_actions)}")
print()
print(f"v0.5.0 changelog marquee primitives: {marquee}")
print(f"Marquee NOT covered by docs row:     {marquee_missing_from_docs}")
print()
print(f"Actions NOT named or world-API-covered: {len(missing)}")
for a in missing:
    print(f"  - {a}")
print("=" * 70)

# -------------------------------------------------------------------------
# 6. Fail loud: at minimum the marquee gap is a shipped bug.
# -------------------------------------------------------------------------
if marquee_missing_from_docs:
    print(
        f"FAIL - v0.5.0 release-note marquee primitives "
        f"{marquee_missing_from_docs} are headline features in the changelog "
        f"but are absent from the ONLY docs page describing the agent-tool "
        f"action surface. An LLM reading agents.md will not emit them."
    )
    sys.exit(1)

if len(missing) > 20:
    print(
        f"NOTE - {len(missing)} published sim actions still unnamed in the "
        f"docs row (even generously). The row directs the reader to "
        f"`robot.tool_spec['description']` for the full enum, which is "
        f"acceptable as long as the row's category grouping does not drift "
        f"out of parity with the tool_spec's own grouping. Not failing on "
        f"this (counts move every release); marquee gap is the hard check."
    )

print("OK - docs row is within tolerance of the published action surface.")
sys.exit(0)
