#!/usr/bin/env python3
"""Repro for: ``annotate_episode`` / ``filter_episodes`` / ``measure_agreement``
vocabulary refusals dump the full ``QUALITY_GRADES`` / ``FAILURE_MODES`` tuple
without a ``Did you mean 'X'?`` hint, while 17+ sibling refusals in the same
codebase do - the user who follows the documented ``docs/learn/data/label-and-judge.md``
recipe lands on realistic typos (``'hi'``, ``'Medium'``, ``'near-miss'``,
``'occlusion'``, ``'collide'``) and gets a stone wall instead of the
one-token nudge that every other refusal in the project ships.

The pattern:
- ``annotate_episode(root, 0, quality="hi")`` currently emits
  ``quality must be one of ('low', 'medium', 'high'), got 'hi'.`` - the hint
  is one ``difflib.get_close_matches(..., cutoff=0.6, n=1)`` call away, and
  the exact same convention is already used at:
    - ``strands_robots/robot.py:211``       (Unknown robot)
    - ``strands_robots/policies/factory.py:398,649``  (Unknown provider / kwarg)
    - ``strands_robots/hardware_robot.py:253``        (Unknown camera type)
    - ``strands_robots/simulation/base.py:205``       (Unknown kwarg)
    - ``strands_robots/simulation/predicates.py:2312`` (Unknown predicate)
    - ``strands_robots/training/factory.py:167``      (Unknown provider)
    - ... 11 more sites, all cutoff=0.6
- The three raise sites touched are the only ``QUALITY_GRADES`` /
  ``FAILURE_MODES`` refusals that reach a user through documented API:
    - ``episode_labels.py:546`` annotate_episode quality
    - ``episode_labels.py:549`` annotate_episode failure_mode
    - ``episode_labels.py:624`` filter_episodes min_quality
    - ``episode_labels.py:717`` measure_agreement human_labels[i].quality
    - ``episode_labels.py:731`` measure_agreement human_labels[i].failure_mode

Run on an **UNPATCHED** tree (``git checkout origin/main``) and the five
realistic typos below all show the full tuple with no arrow. Run on the
patch branch and the five land on ``'high'`` / ``'medium'`` / ``'near_miss'``
/ ``'camera_occlusion'`` / ``'collision'`` while the test-pinned non-matches
(``'excellent'``, ``'sloppy'``, ``'jerky'``) keep their byte-exact historical
message - the suffix is additive, not disruptive.

Expected (patched)::

    annotate quality='hi'        -> ... got 'hi'. Did you mean 'high'?
    annotate quality='Medium'    -> ... got 'Medium'. Did you mean 'medium'?
    annotate fm='near-miss'      -> ... got 'near-miss'. Did you mean 'near_miss'?
    annotate fm='occlusion'      -> ... got 'occlusion'. Did you mean 'camera_occlusion'?
    annotate fm='collide'        -> ... got 'collide'. Did you mean 'collision'?
    annotate quality='excellent' -> ... got 'excellent'.          (BYTE-EXACT)
    annotate fm='sloppy'         -> ... got 'sloppy'.             (BYTE-EXACT)
    annotate fm='jerky'          -> ... got 'jerky'.              (BYTE-EXACT - cutoff=0.6)
    filter min_quality='Medium'  -> ... got 'Medium'. Did you mean 'medium'?
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo-relative import so the repro is self-contained on a fresh clone.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from strands_robots.episode_labels import (  # noqa: E402
    FAILURE_MODES,
    QUALITY_GRADES,
    annotate_episode,
    filter_episodes,
)


def _show(label: str, call) -> None:
    try:
        call()
    except ValueError as exc:
        print(f"  {label}: {exc}")
    except Exception as exc:
        # Only the vocabulary guard should fire; anything else means the test
        # was wired wrong (e.g. the function reached the filesystem).
        print(f"  {label}: ({type(exc).__name__}) {exc}")


def main() -> int:
    print("QUALITY_GRADES =", QUALITY_GRADES)
    print("FAILURE_MODES =", FAILURE_MODES)
    print()

    print("=== quality typos (one word off) ===")
    for bad in ("hi", "Medium", "excellent", "good", "HIGH"):
        _show(f"annotate quality={bad!r:>12}", lambda b=bad: annotate_episode("/tmp/__nxyz", 0, quality=b))

    print()
    print("=== failure_mode typos (dash vs underscore, short form) ===")
    for bad in ("near-miss", "occlusion", "collide", "jerky", "sloppy", "camera-occlusion"):
        _show(
            f"annotate fm={bad!r:>18}",
            lambda b=bad: annotate_episode("/tmp/__nxyz", 0, quality="high", failure_mode=b),
        )

    print()
    print("=== filter_episodes typos (same vocabulary) ===")
    for bad in ("Medium", "excellent"):
        _show(f"filter min_quality={bad!r:>12}", lambda b=bad: filter_episodes("/tmp/__nxyz", min_quality=b))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
