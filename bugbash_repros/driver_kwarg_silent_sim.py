"""
Repro: Robot("so101", driver="lerobot") silently builds a MuJoCo simulator.

Context:
--------
`docs/concepts/robots.md:18` promises
    "A hardware keyword on a sim robot (`cameras=`) is refused rather than ignored."

`strands_robots/robot.py:838-846` implements that promise -- but only for
`cameras=` and the kwargs in `_hardware_only_kwargs()` (= `_FORWARDABLE_KWARGS`
+ `_ADDRESS_FIELDS` + native-driver ctor kwargs).

`driver=` itself -- the Robot kwarg whose own docstring says
    "The value is checked in every mode, but only ``mode='real'`` acts on it;
     ``mode='sim'`` reports it as ignored at debug level."
-- is a HARDWARE-ONLY word (its values are 'strands' / 'lerobot' / 'auto', all
hardware-driver dispatchers). On `mode="sim"` it is semantically meaningless.
Yet it is NOT refused: the factory emits a DEBUG log and builds a simulator
under `status=success`.

That is the same silent-sim failure harness#551 and harness#710 extended the
guard to catch -- missed on `driver=` because `driver=` is in the factory's
`own` set, so `_hardware_only_kwargs()` excludes it on purpose.

User impact
-----------
`docs/robots/so101.md:44` teaches:
    Robot("so101", mode="real", driver="lerobot")
A user who copies that line and forgets `mode="real"` gets a simulator while
asking for the lerobot driver. The arm on the desk stays still, `status`
reports success, and the DEBUG log line is invisible on default logging.

Run:  python bugbash_repros/driver_kwarg_silent_sim.py
"""

from __future__ import annotations

from strands_robots import Robot


def main() -> int:
    # 1) The sibling kwarg `cameras=` is refused -- the guard that
    #    `docs/concepts/robots.md:18` promises.
    try:
        Robot("so101", cameras={"front": {"type": "opencv", "index_or_path": 0}})
        print("UNEXPECTED: cameras= did not refuse on sim")
        return 1
    except ValueError as e:
        print("cameras= on sim REFUSED (good):", str(e)[:120])

    # 2) The sibling kwarg `port=` is refused -- harness#551's follow-through.
    try:
        Robot("so101", port="/dev/ttyACM0")
        print("UNEXPECTED: port= did not refuse on sim")
        return 1
    except TypeError as e:
        print("port= on sim REFUSED (good):", str(e)[:120])

    # 3) `driver="lerobot"` on sim is refused (post-fix) with the same shape
    #    as `cameras=` and `port=`.
    try:
        robot = Robot("so101", driver="lerobot")
        print(f"UNEXPECTED: driver= did not refuse on sim; got {type(robot).__name__}")
        robot.cleanup()
        return 1
    except TypeError as e:
        msg = str(e)
        print("driver='lerobot' on sim REFUSED (fixed):", msg[:200])
        for needle in ("driver=", "mode='real'", "'so101'"):
            assert needle in msg, f"missing in refusal: {needle!r}"
        print("  message names driver=, mode='real', and 'so101' -- parity with cameras=/port=")

    # 4) Regression guard: `driver='auto'` (the default) is NOT refused.
    robot = Robot("so101")
    assert type(robot).__name__ == "MuJoCoSimEngine"
    robot.cleanup()
    print("regression guard: driver='auto' (default) still builds sim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
