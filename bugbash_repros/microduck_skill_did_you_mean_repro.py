"""Repro: Microduck ``send_action({"skill": X})`` refusal has no 'Did you mean'
hint and does not name the cross-surface vocabulary collision.

Target: v0.5.3
Upstream file: strands_robots/drivers/microduck.py:375 (action_to_wire),
               strands_robots/drivers/microduck.py:1849 (do() wrapper)
Harness: cagataycali/robots-harness#<N>
Branch:  cagataycali/robots@bugbash/microduck-skill-did-you-mean

Context
-------
Two separate "skill" vocabularies ship together:

1. ``strands_robots.drivers.microduck.SKILLS`` - the five ``robot.do`` names
   the on-robot daemon answers to:
     ('ground_pick', 'kick_left', 'kick_right', 'sit_toggle', 'roulade')

2. ``strands_robots.policies.microduck`` - the nine ONNX actors listed in
   ``docs/learn/policies/microduck.md:16``:
     alpha_walking, alpha_stand, alpha_sitstand, roulade,
     ball_kick_{left,right}, roller, roller_crouch, alpha_ground_pick

Four of (2) are also in (1) with different names
(``ball_kick_left`` <-> ``kick_left``; ``alpha_ground_pick`` <-> ``ground_pick``;
``ball_kick_right`` <-> ``kick_right``). The others are *policies* that run
through ``create_policy("microduck", onnx_path=...)`` and
``run_policy(policy_provider="microduck", ...)``, not ``robot.do`` skills.

A user who reads the policies page first and then tries
``send_action({"skill": "alpha_walking"})`` on ``Robot("microduck", mode="real")``
lands on a stone-wall refusal: no 'Did you mean', no cross-reference to the
policy surface, even though five sibling refusals in the same project already
import ``difflib`` for exactly this (harness#708 lists them).

Pre-fix output
--------------
    unknown skill 'alpha_walking'; expected one of
    ['ground_pick','kick_left','kick_right','sit_toggle','roulade']

Post-fix output
---------------
    unknown skill 'alpha_walking'; expected one of
    ['ground_pick','kick_left','kick_right','sit_toggle','roulade'];
    'alpha_walking' is an ONNX actor, not a ``robot.do`` skill - run it with
    ``create_policy("microduck", onnx_path="alpha_walking.onnx")`` or
    ``sim.run_policy(policy_provider="microduck",
    policy_config={"onnx_path": "alpha_walking.onnx"})``

    unknown skill 'ball_kick_left'; expected one of [...]; did you mean
    'kick_left'?

Run
---
    python bugbash_repros/microduck_skill_did_you_mean_repro.py
"""

from __future__ import annotations

import threading
from typing import Any


def _connected_driver():
    """Build a connected MicroduckDriver stand-in that never dials a socket."""
    from strands_robots.drivers import microduck as m

    drv = m.MicroduckDriver.__new__(m.MicroduckDriver)
    drv._tool_name = "microduck"
    drv._connected = True
    drv._stopped = False
    drv._skills = None
    drv._cache_lock = threading.Lock()

    class _FakeClient:
        alive = True

        def notify(self, *a: Any, **k: Any) -> None:
            return None

        def call(self, *a: Any, **k: Any) -> dict[str, Any]:
            return {"ok": True}

    drv._client = _FakeClient()
    return drv


def main() -> int:
    drv = _connected_driver()

    cases: list[tuple[str, str]] = [
        # ONNX-actor name (not a robot.do skill)
        ("alpha_walking", "onnx-actor"),
        ("alpha_stand", "onnx-actor"),
        # Policy-side names that map 1:1 to a robot.do skill with a different name
        ("ball_kick_left", "near:kick_left"),
        ("ball_kick_right", "near:kick_right"),
        ("alpha_ground_pick", "near:ground_pick"),
        # Natural-English casual
        ("kick", "near:kick_left/kick_right"),
        # True unknown - fallback path (preserves pin: 'unknown skill' in text)
        ("backflip", "fallback"),
        # Valid - must still succeed
        ("kick_left", "valid"),
        ("roulade", "valid (overlap)"),
    ]

    print("case".ljust(22), "status".ljust(10), "text")
    print("-" * 90)
    for skill, label in cases:
        out = drv.send_action({"skill": skill})
        text = (out.get("content") or [{}])[0].get("text", "")
        print(f"{skill!r:22s} {out['status']:10s} {label}")
        if text:
            print(f"  -> {text}")
        print()

    # Hard assertion: the four near-matches get a 'did you mean', and the
    # ONNX actor gets a cross-surface pointer.
    def _text(skill: str) -> str:
        return (
            drv.send_action({"skill": skill})["content"][0]["text"]
        )

    assert "did you mean 'kick_left'" in _text("ball_kick_left")
    assert "did you mean 'kick_right'" in _text("ball_kick_right")
    assert "did you mean 'ground_pick'" in _text("alpha_ground_pick")
    assert "ONNX actor" in _text("alpha_walking")
    assert 'policy_provider="microduck"' in _text("alpha_walking")

    # Pin: the fallback still begins with ``unknown skill`` so
    # tests/drivers/microduck/test_microduck_driver_over_socket.py:139
    # continues to pass.
    assert "unknown skill" in _text("backflip")
    assert "unknown skill" in _text("alpha_walking")

    print("OK - every assertion passed on the patched tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
