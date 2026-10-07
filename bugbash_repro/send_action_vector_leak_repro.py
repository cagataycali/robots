"""
Repro: send_action list-path refusal leaks the raw float() exception text,
       while the dict-path refusal is type-aware and names the offending key.

Both paths route through strands_robots/simulation/base.py::SimEngine._normalize_action
but refuse non-numeric input with asymmetric messages:

  Dict path  (base.py:2334-2358):
      "send_action: action value for key 'shoulder_pan' must be a scalar
       number (one value per actuator/joint), got str."
      --> names the key, names the type, has the project's refusal voice.

  List path (base.py:2383-2392):
      "send_action: action vector has a non-numeric entry: could not
       convert string to float: 'banana'."
      --> leaks the raw CPython float() ValueError message.
      --> no index of the offender ("entry 2 ('elbow_flex')")
      --> no action-key context
      --> no robot name
      --> no mention of the per-entry type (str vs bool vs bytes)

User-visible impact:
- The dict refusal teaches the shape of the valid input ("one value per
  actuator/joint"). The list refusal doesn't, so the next guess is often
  the same wrong shape.
- When the typo is in the middle of a long vector (14-DOF microduck,
  21-DOF g1), the leaked message doesn't say *which* entry is bad, only
  that one is - the dict message always names its key.
- Agent envelopes (status=error) are consumed by LLMs; a message that
  reads "could not convert string to float: 'banana'" is a stack-trace
  leak, not a next-step suggestion. The dict-path message is actionable.

Fix sketch (base.py:2383-2392, ~8 LOC):
    for idx, raw in enumerate(raw_entries):
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            action_key = action_keys[idx] if idx < len(action_keys) else "?"
            return None, {
                "status": "error",
                "content": [{"text": (
                    f"send_action: action vector entry {idx} "
                    f"('{action_key}') must be a scalar number "
                    f"(one value per actuator/joint), got "
                    f"{type(raw).__name__}."
                )}],
            }
    values = [float(v) for v in raw_entries]

This mirrors the dict-path language verbatim; the leaked float() text
goes away and the index + action-key appear.

Run:  python send_action_vector_leak_repro.py
Expected exit: 0
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import strands_robots  # noqa: E402


def main() -> int:
    r = strands_robots.Robot("so101", mesh=False)

    # ------------------------------------------------------------------
    # Part A: dict-path non-numeric string refusal (reference voice)
    # ------------------------------------------------------------------
    res_dict = r.send_action({"shoulder_pan": "banana"})
    assert res_dict.get("status") == "error", res_dict
    msg_dict = res_dict["content"][0]["text"]
    print("[A] dict path:", msg_dict)
    assert "must be a scalar number" in msg_dict
    assert "got str" in msg_dict
    assert "shoulder_pan" in msg_dict          # key named
    assert "could not convert" not in msg_dict  # no float() leak

    # ------------------------------------------------------------------
    # Part B: list-path non-numeric string refusal (leaky voice)
    # ------------------------------------------------------------------
    r2 = strands_robots.Robot("so101", mesh=False)
    res_list = r2.send_action(
        ["banana", "0.0", "0.0", "0.0", "0.0", "0.0"]
    )
    assert res_list.get("status") == "error", res_list
    msg_list = res_list["content"][0]["text"]
    print("[B] list path:", msg_list)

    # Three concrete gaps vs the dict message:
    assert "could not convert string to float" in msg_list, "raw float() leak"
    assert "entry 0" not in msg_list, "no index pointer in the message"
    assert "shoulder_pan" not in msg_list, "no action-key name in the message"
    assert "so101" not in msg_list, "no robot name in the message"
    assert "scalar number" not in msg_list, (
        "the project's refusal voice is on the dict path, not here"
    )

    # ------------------------------------------------------------------
    # Part C: middle-of-vector failure proves the index gap matters
    # ------------------------------------------------------------------
    r3 = strands_robots.Robot("so101", mesh=False)
    res_mid = r3.send_action(["0.0", "0.0", "0.0", "banana", "0.0", "0.0"])
    assert res_mid.get("status") == "error"
    msg_mid = res_mid["content"][0]["text"]
    print("[C] list path with bad entry at idx=3:", msg_mid)
    # The leaked message DOES carry the bad token verbatim ("'banana'") -
    # the only mercy - but still no index and no action-key.
    assert "'banana'" in msg_mid
    assert "3 (" not in msg_mid, "fourth entry maps to action-key 'wrist_flex'"
    assert "entry 3" not in msg_mid

    # ------------------------------------------------------------------
    # Part D: non-str leaky types - sequences inside a vector
    # ------------------------------------------------------------------
    r4 = strands_robots.Robot("so101", mesh=False)
    res_nested = r4.send_action([[0.1], 0.0, 0.0, 0.0, 0.0, 0.0])
    # float([0.1]) raises TypeError, caught by the same except.
    assert res_nested.get("status") == "error"
    msg_nested = res_nested["content"][0]["text"]
    print("[D] list with nested list entry:", msg_nested)
    assert "could not convert" in msg_nested or "float()" in msg_nested or \
           "argument must be" in msg_nested, (
        "a TypeError also leaks the raw CPython message verbatim"
    )

    for robot in (r, r2, r3, r4):
        robot.cleanup()

    print()
    print("Asymmetry confirmed:")
    print("  dict({'shoulder_pan': 'banana'}) -> type-aware, names key, named type")
    print("  list(['banana', ...])            -> raw float() ValueError leak")
    print()
    print("Fix: base.py:2383 - mirror the dict path's isinstance+type guard,")
    print("     naming index + action-key before the float() coercion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
