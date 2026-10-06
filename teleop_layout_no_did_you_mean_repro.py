#!/usr/bin/env python3
"""Repro: teleop_layout() refusal lists 4 names but omits the 'Did you mean' hint
every sibling refusal in the project carries.

Fire anchored at strands_robots/teleop/layouts.py:162 (upstream main, commit
1bf94772e).  _LAYOUTS currently names 4 keys:

    g1_joint_29, blog_31_66, blog_31_31, lerobot_token_64

A near-miss typo (user forgets the plural; types the dim count wrong; uses
hyphens) prints the enumeration but nothing points at the actual spelling.
The project's own convention for name-lookup refusals is difflib.get_close_matches
+ 'Did you mean: ...?'; it is applied at 20+ sites (see
strands_robots/policies/factory.py:450, strands_robots/robot.py:_unknown_robot_msg,
strands_robots/hardware_robot.py:252, and the sibling hardness notes at
cagataycali/robots-harness#708, #693, #686, #679).

Run:
    MUJOCO_GL=egl python teleop_layout_no_did_you_mean_repro.py
Expected (after fix):
    'g1_joint_28' -> "Did you mean: 'g1_joint_29'?"
Observed on main:
    'g1_joint_28' -> only "Choose from: [...]" — no hint.
"""
from __future__ import annotations

import sys

from strands_robots.teleop.layouts import teleop_layout


_NEAR_MISS_CASES: list[tuple[str, str]] = [
    # (typo, canonical the user actually wanted)
    ("g1_joint_28", "g1_joint_29"),
    ("lerobot_tokens_64", "lerobot_token_64"),  # plural slip
    ("blog_31_6",   "blog_31_66"),               # dropped digit
    ("blog_66_31",  "blog_31_66"),               # dim swap
    ("G1_JOINT_29", "g1_joint_29"),              # case
    ("blog-31-66",  "blog_31_66"),               # dash-vs-underscore
]


def _refusal(name: str) -> str:
    try:
        teleop_layout(name)
    except ValueError as e:
        return str(e)
    raise AssertionError(f"teleop_layout({name!r}) did not refuse; the registry grew")


def main() -> int:
    bad = 0
    for typo, canonical in _NEAR_MISS_CASES:
        msg = _refusal(typo)
        has_hint = "Did you mean" in msg
        tag = "OK  " if has_hint else "MISS"
        if not has_hint:
            bad += 1
        print(f"{tag}  {typo!r:>22}  (wanted {canonical!r:>20})\n       {msg}")
    print()
    if bad:
        print(f"FAIL  {bad}/{len(_NEAR_MISS_CASES)} near-miss typos received NO 'Did you mean' hint.")
        print("      Sibling refusals (robot.py, hardware_robot.py, factory.py x2) all do.")
        return 1
    print("PASS  teleop_layout() names the closest spelling on every near-miss.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
