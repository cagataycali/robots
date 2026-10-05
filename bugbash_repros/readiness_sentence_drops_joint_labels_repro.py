"""Repro: _world_readiness_sentence drops joint_labels on the LLM hot path.

Context — a user follows README.md's quickstart but types so101 (the
"upgraded SO-100" per registry) instead of so100. On v0.5.3 HEAD, the
registry carries joint_labels for both arms (post-harness#520), but the
TOOL DESCRIPTION the LLM sees still prints only the asset joint names.

Upstream pin: strands_robots/simulation/mujoco/simulation.py:6909-6945
             (_world_readiness_sentence builds `6 joints: {shown}` from
             robot.joint_names only).

Compare with the SIBLING get_robot_state output on the SAME object, which
#520's fix did reach: it prints "1 (shoulder_pan): pos=…". Two renderings
of the same registry fact in the same engine disagree on the LLM hot path.

Run:
    MUJOCO_GL=egl python readiness_sentence_drops_joint_labels_repro.py

Exit code: 0 if the asymmetry reproduces (defect present), 1 if it is fixed.
"""
import os
import re
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot

r = Robot("so101")

# A) LLM hot path: the tool_spec description.
tool_desc = r.tool_spec["description"]
# Capture the whole "N joints: ...)" clause by matching to the robot-entry
# closing paren rather than any `)` inside a label like "(shoulder_pan)".
m = re.search(r"\(\d+ joints: ([^\n]+?)\)(?= - do not)", tool_desc)
tool_joints_clause = m.group(1) if m else "<not found>"

# B) Output path: get_robot_state, which #520 fixed.
state_text = r.get_robot_state()["content"][0]["text"]
state_lines = [ln for ln in state_text.splitlines() if ln.startswith(("1 ", "6 "))]

print("TOOL_SPEC (LLM hot path):")
print(f"  {tool_joints_clause!r}")
print()
print("GET_ROBOT_STATE (output path):")
for ln in state_lines[:2]:
    print(f"  {ln.strip()}")

# Defect fingerprint — tool_spec names "1".."6" with no "(label)" annotation,
# while get_robot_state names them.
tool_shows_labels = "(shoulder_pan)" in tool_joints_clause or "(gripper)" in tool_joints_clause
state_shows_labels = any("(shoulder_pan)" in ln or "(gripper)" in ln for ln in state_lines)

print()
print(f"tool_spec includes labels? {tool_shows_labels}")
print(f"get_robot_state includes labels? {state_shows_labels}")

if (not tool_shows_labels) and state_shows_labels:
    print()
    print("DEFECT REPRODUCED: labels in state (post-#520), NOT in tool_spec sentence.")
    print("Fix site: strands_robots/simulation/mujoco/simulation.py:6945")
    sys.exit(0)
sys.exit(1)
