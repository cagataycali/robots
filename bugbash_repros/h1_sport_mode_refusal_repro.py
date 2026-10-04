"""
Repro: `Robot("h1", mode="real").send_action(...)` refuses with Go2-only vocabulary.

Upstream: https://github.com/strands-labs/robots PR #4502, released at 86b2ef8
"feat(drivers): Robot("h1", mode="real") drives the H1 over the Go2's unitree_go wire"

Symptom:
  The new native H1 path reuses Go2Driver with an H1 WireProfile. The write-gate
  (`_check_motion_gates`, strands_robots/drivers/go2.py:1109-1141), the
  gate-refusal copy, the state envelope keys (`sport_mode_released`,
  `sport_mode_name`), the DDS subscription plan (`rt/sportmodestate`,
  `_subscription_plan`:L785-787) and the recommended unlock (`release_sport_mode()`,
  L1004-1075) are all Go2 vocabulary -- "sport mode" is the Go2's onboard
  locomotion service. The H1's SDK example writes `rt/lowcmd` directly without
  calling MotionSwitcherClient.ReleaseMode(), and the H1 does not publish
  `rt/sportmodestate`.

  An H1 user who follows docs/robots/unitree_h1.md ("Robot('unitree_h1',
  mode='real', port=..., network_interface=...)") then calls `send_action`
  is told to call release_sport_mode() first -- on a robot that has no sport
  mode. The sibling G1Driver tells its callers to clear "FSM id in
  HANDSHAKE_FSMS", using the right vocabulary for the robot.

Repro: no DDS, no SDK, no robot -- the refusal text is deterministic.

  $ python3 h1_sport_mode_refusal_repro.py
  profile: unitree_h1    # H1 WireProfile selected by Robot("h1", mode="real")
  refusal: send_action refused: sport mode is not released - call release_sport_mode()
           first, or the onboard controller and this driver fight over the same
           motors

Expected:
  Either (A) the refusal names the H1's own unlock (whatever the H1 firmware
  actually gates on), or (B) the refusal is removed for the H1 (its example
  shows no equivalent service to release), or (C) the WireProfile declares
  which gates apply and `_check_motion_gates` reads that.

Actual: the Go2's sport-mode gate is force-applied to the H1, and the refusal
  advertises a service name the H1 has never heard of.

Related: the state envelope keys from `state` (:L762-768) also read
  `sport_mode_released`/`sport_mode_name` unconditionally; `docs/learn/hardware/
  unitree.md` "Two drivers, two gates" table (:L44-52) does not have an H1
  column, so the user has no doc source of truth that would warn them.

Harness target: v0.5.3.
"""
from strands_robots import Robot
from strands_robots.drivers.go2 import Go2Driver, WIRE_PROFILES


def main() -> None:
    # 1. Robot("h1", mode="real") lands on Go2Driver with the H1 WireProfile.
    #    This mirrors docs/robots/unitree_h1.md:25 (the sketch block added by
    #    #4502 itself).
    r = Robot("h1", mode="real", port="192.168.123.161", network_interface="eth0")
    assert isinstance(r, Go2Driver), type(r).__name__
    assert r._profile is WIRE_PROFILES["unitree_h1"], r._profile.model
    print("profile:", r._profile.model)

    # 2. Pretend connect_eagerly() succeeded. On the real H1 this would come
    #    through DDS; the gate under test is independent of DDS state.
    r._connected = True
    r._pubs = object()  # the gate fires before publisher is touched

    # 3. Send a bona-fide H1 joint. H1_JOINT_INDEX includes "left_hip_yaw" at
    #    slot 7 (strands_robots/drivers/go2.py:216-235).
    result = r.send_action({"left_hip_yaw": 0.0})

    # 4. The refusal is Go2-only vocabulary applied to an H1.
    assert result["status"] == "error", result
    text = result["content"][0]["text"]
    print("refusal:", text)

    # These assertions are what pins the defect:
    assert "sport mode is not released" in text, text
    assert "release_sport_mode()" in text, text
    # ... and nothing about the H1's own motor-release concept, because the
    # driver has none.
    assert "H1" not in text.upper().replace("UNITREE_H1", "")
    assert "FSM" not in text  # the G1 vocabulary leaks through sibling driver
    #                           but not this one; neither does the H1's.

    print()
    print("Verdict: Go2's sport-mode refusal is applied to an H1 verbatim.")


if __name__ == "__main__":
    main()
