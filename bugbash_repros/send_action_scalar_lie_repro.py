"""Minimal repro: `send_action` refusal message says "scalar number" but
accepts non-scalars (str, bytes, list, tuple, 1-element array) and only
rejects the sibling non-scalar shapes (dict, >1-element list, 'hello').

Target: v0.5.3
Upstream: strands_robots/simulation/base.py:2259 (refusal text),
                                    :2237-2244 (deliberate unwrap).
"""

from __future__ import annotations

import sys

import numpy as np

from strands_robots import Robot


def main() -> int:
    r = Robot("so101", mesh=False)

    cases: list[tuple[str, object, str]] = [
        # (label, value, expected)
        ("'0.5' (str)", "0.5", "accepted silently"),
        ("b'0.5' (bytes)", b"0.5", "accepted silently"),
        ("[0.5] (list[1])", [0.5], "accepted silently"),
        ("(0.5,) (tuple[1])", (0.5,), "accepted silently"),
        ("np.array([0.5])", np.array([0.5]), "accepted silently"),
        # The refusal says "scalar number" - which is what a reader would
        # expect the ACCEPTED set above to violate. These, however, DO fail:
        ("[0.5, 0.6] (list[2])", [0.5, 0.6], "rejected 'scalar number'"),
        ("{'v':0.5} (dict)", {"v": 0.5}, "rejected 'scalar number'"),
    ]

    worst = 0
    for label, value, expected in cases:
        res = r.send_action({"1": value})
        status = res.get("status", "???")
        text = (res.get("content", [{}]) or [{}])[0].get("text", "")
        print(f"{label:28s} -> {status:7s}   expected: {expected}")
        if "scalar number" in text:
            print(f"    message: {text}")

        # The lie: refusal names a "scalar number" rule the acceptances violate.
        # Specifically: the length-1 unwrap (#1538) is a deliberate contract
        # that the message does NOT disclose, so a caller reading
        # 'must be a scalar number ... got list' for [0.5, 0.6] cannot
        # know that [0.5] is a different story.
        if status == "success" and not isinstance(value, (int, float, np.floating, np.integer)):
            worst = 1  # we accepted a non-scalar as a "scalar"

    print()
    print("DEFECT:" if worst else "clean.")
    if worst:
        print(
            "  send_action's refusal names a 'scalar number (one value per actuator/\n"
            "  joint)' rule, but the same code deliberately unwraps a length-1\n"
            "  str/bytes/list/tuple/ndarray into a scalar (see #1538 fix at\n"
            "  base.py:839 `_unwrap_single_element_action_value`). A caller reading\n"
            "  'must be a scalar number ... got list' for `[0.5, 0.6]` has no way\n"
            "  to see that `[0.5]` would have worked. The message tells only half\n"
            "  of its rule."
        )
    return worst


if __name__ == "__main__":
    sys.exit(main())
