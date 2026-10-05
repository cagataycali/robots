"""
Shortest failing example for defect:
  lekiwi registry entry at strands_robots/registry/robots.json:1046 omits the
  `joint_labels` block that so100 (:244) and so101 (:290) both ship, breaking
  the canonical label spelling (shoulder_pan..gripper) for the SAME so-arm
  family on the SAME MJCF joint names (Rotation/Pitch/Elbow/Wrist_Pitch/
  Wrist_Roll/Jaw).

Resolver chain (unchanged, works when joint_labels is present):
  simulation/mujoco/simulation.py:1355   send_action -> _action_key_actuator
  simulation/mujoco/rendering.py:1065    actuator-miss -> _resolve_joint_label
  simulation/mujoco/physics.py:1162      label -> joint via registry.joint_labels()
  registry/robots.py:117                 reads robots.json "joint_labels"

Expected:
  Any of {so100, so101, lekiwi} should accept the SAME canonical label dict
  {"shoulder_pan": ..., ...} because they share the so-arm family (lekiwi =
  so100 arm + 3 omniwheels).

Actual (on v0.5.3 HEAD, each as a fresh process to avoid other confounders):
  so100    -> success
  so101    -> success
  lekiwi   -> error  Keys ['shoulder_pan', 'shoulder_lift', ...] could not
                     be resolved to actuators or joints on 'lekiwi'.

Fix (verified on this branch): 8-line additive `joint_labels` block on
registry/robots.json's lekiwi entry, mirroring so100's exactly.
"""
import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

ARM_LABELS = {
    "shoulder_pan":  0.0,
    "shoulder_lift": 0.0,
    "elbow_flex":    0.0,
    "wrist_flex":    0.0,
    "wrist_roll":    0.0,
    "gripper":       0.0,
}
WHEELS = {
    "base_back_wheel":  0.0,
    "base_left_wheel":  0.0,
    "base_right_wheel": 0.0,
}

# Each robot in a FRESH subprocess so no shared sim state can mask the result.
TEMPLATE = """
import warnings; warnings.filterwarnings('ignore')
from strands_robots import Robot
r = Robot({name!r})
action = {action!r}
res = r.send_action(action)
print({name!r}, res.get('status'), '-', res['content'][0]['text'][:160])
r.cleanup()
"""

env = {k: v for k, v in os.environ.items() if k != "SYSTEM_PROMPT"}
cases = [
    ("so100",  dict(ARM_LABELS)),
    ("so101",  dict(ARM_LABELS)),
    ("lekiwi", {**ARM_LABELS, **WHEELS}),
]
for name, action in cases:
    code = TEMPLATE.format(name=name, action=action)
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=90)
    print((r.stdout or r.stderr).strip())
