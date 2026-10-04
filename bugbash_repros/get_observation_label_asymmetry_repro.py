"""
Repro: get_observation() violates its own documented schema on so101.

Upstream contract (strands_robots/simulation/base.py:1894):

    Schema:
        - "<joint_name>" (float): One entry per joint on the robot,
          keyed by the *short* joint name (e.g. "shoulder_pan").

docs/robots/so101.md names "shoulder_pan"..."gripper" the *Action keys*
and nothing in the per-robot page or base.py's schema warns the reader
that write-side (send_action) and read-side (get_observation) use two
different vocabularies. send_action({"shoulder_pan": v}) returns
status=success; obs["shoulder_pan"] raises KeyError. The obs dict carries
'1'..'6' instead - the MuJoCo asset joint names - because
_get_sim_observation (strands_robots/simulation/mujoco/rendering.py:719)
writes obs[jnt_name] with the raw asset name, never consulting
_robot_joint_labels (physics.py:1129) which the write path uses to
resolve 'shoulder_pan' -> '1'.

Impact: a feedback loop
    tgt = {"shoulder_pan": 0.1}
    robot.send_action(tgt)
    obs = robot.get_observation()
    err = tgt["shoulder_pan"] - obs["shoulder_pan"]  # KeyError
breaks on the first iteration. The per-robot page's own table is the
primary teaching surface for these names; a user writing a controller
has no reason to suspect that '1' is the read key for the same joint
they just wrote as 'shoulder_pan'.

Run:
    MUJOCO_GL=egl python bugbash_repros/get_observation_label_asymmetry_repro.py
Expected (schema): obs["shoulder_pan"] returns the position after the write.
Actual: KeyError on 'shoulder_pan'; obs carries '1'..'6' only.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot


def main() -> int:
    r = Robot("so101")
    try:
        # The write-side vocabulary docs/robots/so101.md teaches.
        write_result = r.send_action({"shoulder_pan": 0.1})
        write_status = write_result.get("status") if isinstance(write_result, dict) else write_result
        print(f"send_action({{'shoulder_pan': 0.1}}): status={write_status}")
        assert write_status == "success", "send_action label write unexpectedly refused"

        # The read-side the base.py schema promises.
        obs = r.get_observation(skip_images=True)
        joint_keys = sorted(k for k in obs.keys() if "." not in k and not k.startswith("base_") and not k.startswith("body."))
        print(f"obs joint keys: {joint_keys}")

        # Does the label used on the write side appear on the read side?
        if "shoulder_pan" in obs:
            print(f"PASS: obs['shoulder_pan'] = {obs['shoulder_pan']}")
            return 0

        # The schema violation.
        try:
            _ = obs["shoulder_pan"]
        except KeyError as e:
            print(f"FAIL: KeyError({e}) - schema says keys are 'short joint name "
                  f"(e.g. \"shoulder_pan\")', but so101 obs carries only {joint_keys}.")
            print("Write-side resolved 'shoulder_pan' via _robot_joint_labels; "
                  "read-side _get_sim_observation never consults the same map.")
            return 1
    finally:
        r.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
