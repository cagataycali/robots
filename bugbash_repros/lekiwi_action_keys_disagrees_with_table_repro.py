#!/usr/bin/env python3
"""Repro: docs/robots/lekiwi.md's sim quickstart prints robot_action_keys("lekiwi"),
whose output disagrees with the "send_action label" column of the table two
lines below on the same page.

A new user reading docs/robots/lekiwi.md top-to-bottom runs:

    robot = Robot("lekiwi", position=[0.0, 0.0, 0.0346])
    print(robot.robot_action_keys("lekiwi"))

and sees:

    ['base_back_wheel', 'base_right_wheel', 'base_left_wheel',
     'Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']

Six lines below, the "Observation key / send_action label" table on the same
page tells them the send_action labels are:

    base_back_wheel, base_right_wheel, base_left_wheel,
    shoulder_pan,    shoulder_lift,    elbow_flex,
    wrist_flex,      wrist_roll,       gripper

These are two different vocabularies for the SAME robot in the SAME doc page.

Why it matters:
  - The method's own docstring (strands_robots/simulation/base.py:1521-1549)
    says the output is "the names a policy should emit as its action-dict
    keys" and "orders the observation.state vector a policy reads".
  - Recordings keyed by robot_action_keys() produce action columns named
    'action.Rotation'/'action.Pitch' etc; a dataset keyed by the docs table
    labels produces 'action.shoulder_pan'/'action.shoulder_lift' etc. Both
    describe the SAME joint on the SAME robot in two incompatible schemas.
  - send_action accepts BOTH forms (registry joint_labels on `lekiwi`
    bridges Rotation→shoulder_pan since harness#755 landed), so a user
    following the table and a user following the quickstart print both
    "work" - but produce incompatible datasets they then can't merge.

Related priors (none duplicate):
  - harness#712: get_observation() ignores joint_labels on so101 (READ-side
    asymmetry on so101). Different method, different robot named in title.
  - harness#755: lekiwi registry omits joint_labels (WRITE-side resolver).
    FIXED - the registry now has the labels, so send_action({shoulder_pan})
    succeeds. This issue is that robot_action_keys() introspection does not
    surface what the write-side resolver already consumes.
  - harness#683: docs/learn/hardware/feetech-arms.md robot_ip typo (docs).
  - harness#696: lekiwi category classification (registry).
  - harness#741: lekiwi docs quickstart position= warning (docs generator).

Run: python3 bugbash_repros/lekiwi_action_keys_disagrees_with_table_repro.py
"""
import os
import subprocess
import sys


def run_sim(code: str) -> str:
    env = {k: v for k, v in os.environ.items() if k != "SYSTEM_PROMPT"}
    r = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, timeout=90,
    )
    return r.stdout + ("\n--STDERR--\n" + r.stderr if r.returncode else "")


DOCS_TABLE_LABELS = [
    "base_back_wheel", "base_right_wheel", "base_left_wheel",
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

print("=" * 72)
print("1. docs/robots/lekiwi.md:14-19 — the printed quickstart")
print("=" * 72)
out = run_sim("""
import warnings; warnings.filterwarnings('ignore')
from strands_robots import Robot
robot = Robot('lekiwi', position=[0.0, 0.0, 0.0346])
print(robot.robot_action_keys('lekiwi'))
robot.cleanup()
""").strip()
print(f"OUTPUT: {out}")

print()
print("=" * 72)
print("2. docs/robots/lekiwi.md:22-32 — the 'send_action label' column")
print("=" * 72)
print(f"LABELS: {DOCS_TABLE_LABELS}")

print()
print("=" * 72)
print("3. Comparison")
print("=" * 72)
printed = out.strip().strip("'[]").replace("'", "").split(", ")
print(f"quickstart print match docs table?  {printed == DOCS_TABLE_LABELS}")
print(f"  wheels agree:  {printed[:3] == DOCS_TABLE_LABELS[:3]}")
print(f"  arm  differs:  print={printed[3:]}")
print(f"                 docs ={DOCS_TABLE_LABELS[3:]}")

print()
print("=" * 72)
print("4. Both vocabularies ARE accepted by send_action — but introspection"
      " is one-sided")
print("=" * 72)
dual = run_sim("""
import warnings; warnings.filterwarnings('ignore')
from strands_robots import Robot
r = Robot('lekiwi', position=[0.0, 0.0, 0.0346])
# docs quickstart form
r1 = r.send_action({'Rotation': 0.1, 'Pitch': 0.1})
print('quickstart-form:', r1['status'], '-', r1['content'][0]['text'])
# docs table form
r2 = r.send_action({'shoulder_pan': 0.1, 'shoulder_lift': 0.1})
print('docs-table-form:', r2['status'], '-', r2['content'][0]['text'])
r.cleanup()
""").strip()
print(dual)

print()
print("=" * 72)
print("5. The sibling so100 (same arm family, has joint_labels in registry)")
print("=" * 72)
so100 = run_sim("""
import warnings; warnings.filterwarnings('ignore')
from strands_robots import Robot
r = Robot('so100')
print('so100 action_keys:', r.robot_action_keys('so100'))
r.cleanup()
""").strip()
print(so100)
print("docs/robots/so100.md teaches the SAME shoulder_pan..gripper label set.")

print()
print("Papercut class: Docs mismatch + Introspection asymmetry (write path "
      "is label-aware since harness#755; read-side robot_action_keys is not).")
