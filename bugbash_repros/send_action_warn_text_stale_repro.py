"""Minimal repro: send_action's warning text contradicts its return envelope.

Since commit b0fb474 (#4486, "fix(sim): send_action refuses a batch with an
unknown key whole, before anything moves"), send_action with any invalid key
in the batch refuses the WHOLE batch: no ctrl is written, the world does not
advance. The return envelope says so correctly:

    "Nothing was applied and the world did not advance."

But the WARNING log emitted from the same code path still says:

    "[sim] action key 'nonsense_joint' (prefix='so101/') could not be applied:
     no actuator or joint. The value was dropped. Valid keys for this robot: ..."

"The value was dropped" is factually wrong twice:
  * There was never a value written anywhere — the resolver runs BEFORE any
    write (simulation.py:1337-1350).
  * Nothing *else* in the batch was written either — the whole batch is refused.

A log reader believes the valid keys in the batch landed and only the typo'd
key was dropped. The return envelope says the opposite. The two contradict
each other on every mixed-batch call.

The same `_warn_unresolved_action_key` is also called from the action-controller
fallback path (rendering.py:1028) where keys ARE partially applied — so the
message is misleading in both call sites but in opposite directions.

Expected: warning text matches current whole-batch semantics (or carries a
`batch_refused` signal so the two call sites can phrase their outcome).

Repro:
    MUJOCO_GL=egl python bugbash_repros/send_action_warn_text_stale_repro.py

Observe the WARNING line vs the final error text. They disagree.
"""
import logging
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

# Capture WARNING logs
log_records = []
class Capture(logging.Handler):
    def emit(self, record):
        log_records.append(record.getMessage())

logging.getLogger("strands_robots.simulation.mujoco.rendering").addHandler(Capture())
logging.getLogger("strands_robots.simulation.mujoco.rendering").setLevel(logging.WARNING)

from strands_robots import Robot

robot = Robot("so101")
# Mixed batch: one valid key ("1"), one invalid key ("nonsense_joint")
result = robot.send_action({"1": 0.5, "nonsense_joint": 1.0})

print("=== WARNING log said ===")
for msg in log_records:
    print(" ", msg)

print()
print("=== Return envelope said ===")
print(" ", result["content"][0]["text"])

print()
# Verify the valid key "1" did NOT land (batch refused whole):
state = robot.get_robot_state()["content"][1]["json"]["state"]
pos_1 = state["1"]["position"]
print(f"=== Joint 1 position after mixed-batch send_action: {pos_1} (expect ~0.0, NOT 0.5) ===")

robot.cleanup()

# The contradiction:
#   log says: "The value was dropped." (singular key — implies other keys applied)
#   envelope says: "Nothing was applied and the world did not advance."
#   joint 1 did NOT move despite being named with a valid value.
#
# They cannot both be true. If a user trusts the WARNING ("just the typo was
# dropped, the valid keys landed"), they skip the retry and the arm is stuck
# at its last pose while they debug why their controller "sent" a command
# that never arrived.
if abs(pos_1 - 0.5) < 0.1:
    print("UNEXPECTED: joint 1 moved — pre-batch refusal did not actually take effect")
    sys.exit(2)
else:
    print("CONFIRMED defect: batch was refused whole (joint 1 at zero), but the")
    print("                  WARNING log says 'The value was dropped' — misleading.")
    sys.exit(1)  # non-zero: defect reproduces
