"""Repro: Robot("lekiwi", mode="real") picks FeetechDriver whose MOTORS map
is SO_ARM_MOTORS (6 arm servos). The 3 omniwheels of the mobile base have no
motor entries, so every wheel action the docs promote fails at the bus-write
layer with "unknown motor 'base_back_wheel'".

Source-of-truth:
- docs/robots/lekiwi.md:19 sketches
      Robot("lekiwi", mode="real", port="/dev/ttyACM0")  # FeetechDriver
  as if the native driver were a valid default for lekiwi.
- strands_robots/drivers/feetech/driver.py:102-108  lists lekiwi in
  SUPPORTED_ROBOTS.
- strands_robots/drivers/feetech/driver.py:75,158  sets
      MOTORS: dict = SO_ARM_MOTORS
  with no per-robot override.
- strands_robots/drivers/feetech/bus.py:161-167    SO_ARM_MOTORS has 6 arm
  joints, zero wheels.
- strands_robots/drivers/feetech/bus.py:449-453    unknown motor rejection.

Expected (either):
  (a) Robot("lekiwi", mode="real", driver="strands") refuses at construction
      with: "FeetechDriver drives the SO-arm half; the lekiwi base needs
      driver='lerobot'", OR
  (b) MOTORS for lekiwi includes 3 wheel entries (feature-sized, not covered
      here).

Actual: construction succeeds; the arm-six works; every wheel action fails
with the bus's "unknown motor" message, which does not point at the
driver/registry mismatch the user is actually hitting.

Repro uses transport='twin' so no serial device is needed.
"""

from __future__ import annotations

import sys


def main() -> int:
    from strands_robots import Robot

    sim = Robot("lekiwi")  # MuJoCoSimEngine for the twin bus

    # After the fix, this construction raises at the driver seam with a
    # message that names the lerobot driver as the whole-robot path. Before
    # the fix, construction succeeds and the wheel action fails downstream.
    try:
        arm = Robot("lekiwi", mode="real", driver="strands", transport="twin", sim=sim)
    except Exception as e:
        text = str(e)
        print(f"CONSTRUCTION REFUSED (post-fix): {type(e).__name__}: {text}")
        assert "lekiwi" in text and "omniwheels" in text and "driver='lerobot'" in text, (
            f"expected the fixed refusal to name the base and the lerobot driver, got {text!r}"
        )
        print("\nFIX VERIFIED: Robot('lekiwi', mode='real', driver='strands')")
        print("refuses up front with the driver/registry mismatch named, pointing")
        print("the caller at driver='lerobot' which covers the whole robot.")
        return 0

    # ---- The pre-fix path (kept so the repro still demonstrates the defect
    # if the gating is dropped from this driver). ----
    wheel_action = {"base_back_wheel": 0.1, "base_right_wheel": 0.1, "base_left_wheel": 0.1}
    reply = arm.send_action(wheel_action)
    status = reply.get("status")
    text = ((reply.get("content") or [{}])[0].get("text", ""))

    arm_action = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0}
    arm_reply = arm.send_action(arm_action)
    arm_status = arm_reply.get("status")

    print(f"wheel send_action status: {status!r}")
    print(f"wheel send_action text:   {text!r}")
    print(f"arm   send_action status: {arm_status!r}")
    print(f"driver MOTORS keys:       {sorted(arm.MOTORS.keys())}")
    print(f"driver SUPPORTED_ROBOTS:  {arm.SUPPORTED_ROBOTS}")

    assert status == "error", "expected the current (defective) refusal"
    assert "unknown motor 'base_back_wheel'" in text, (
        f"expected bus-level 'unknown motor' refusal, got {text!r}"
    )
    assert arm_status == "success", "arm half is reachable; only the base is dead"

    print("\nDEFECT CONFIRMED (pre-fix path): FeetechDriver (the docs' lekiwi")
    print("default) cannot drive the 3 omniwheels; its MOTORS map is")
    print("SO_ARM_MOTORS. The refusal names the bus's inventory rather than")
    print("the driver/registry mismatch the user is actually hitting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
