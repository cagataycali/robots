"""
Repro for: docs/learn/policies/microduck.md prose vs sketch key mismatch
         `bundle.switch("alpha_stand")` named in prose line 45, but the
         sketch 6 lines below (:51-58) keys the bundle by {"walk", "stand"}.

A reader copy-pasting the shipped sketch and following the prose hits
`ValueError: MicroduckPolicyBundle: unknown skill 'alpha_stand';
have ['walk', 'stand'].`

Target: v0.5.3
Upstream: docs/learn/policies/microduck.md:45 vs :51-58
         strands_robots/policies/microduck/composite.py:181 (switch validator)

No network, no weights, no onnxruntime: a bare MicroduckSession stub is
all we need - the bundle only looks up its own keys.
"""

from __future__ import annotations

import sys


def main() -> int:
    from strands_robots.policies.microduck.composite import MicroduckPolicyBundle
    from strands_robots.policies.microduck.policy import MicroduckPolicy
    from strands_robots.policies.microduck._session import MicroduckSession

    sess1 = MicroduckSession.__new__(MicroduckSession)
    sess2 = MicroduckSession.__new__(MicroduckSession)
    p_walk = MicroduckPolicy(session=sess1)
    p_stand = MicroduckPolicy(session=sess2)

    # Lines 51-58 of docs/learn/policies/microduck.md, byte-for-byte:
    bundle = MicroduckPolicyBundle(
        {"walk": p_walk, "stand": p_stand},
        active="stand",
        switch_on_velocity=0.05,
        move_key="walk",
        idle_key="stand",
    )
    assert list(bundle._policies.keys()) == ["walk", "stand"]

    # Line 45 of the same page: "`bundle.switch("alpha_stand")` swaps mid-rollout."
    try:
        bundle.switch("alpha_stand")
    except ValueError as e:
        print(f"REPRO: {e}")
        # Verifies the mismatch: the key the prose names is not held.
        assert "unknown skill 'alpha_stand'" in str(e)
        assert "['walk', 'stand']" in str(e)
        return 0
    print("UNEXPECTED SUCCESS: prose aligned with sketch, defect may be fixed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
