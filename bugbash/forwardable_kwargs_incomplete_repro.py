"""Minimal repro: _FORWARDABLE_KWARGS is INCOMPLETE, so the sim-mode
guard added by harness#551 (strands-labs/robots#3740) silently misses
every network/discovery kwarg that is not literally spelled 'port' or
'robot_ip' — including kwargs the project's own hardware docs sketch
tells users to pass.

Target: strands-labs/robots main @ 2558aee
Defect class: Silent-wrong + Error UX (asymmetric guard across sibling
hardware kwargs; incomplete follow-through on harness#551).

HOW TO RUN:
    cd /path/to/strands-labs/robots-fork  # editable install from this tree
    pip install -e .
    python bugbash/forwardable_kwargs_incomplete_repro.py

EXPECTED (what the design in harness#551 promises):
    Every row below should refuse with
        TypeError: Robot('X') is a simulation ... and would ignore <kwarg>=:
                   ... Add mode='real' to drive the hardware, or drop them to simulate.

ACTUAL:
    Only rows passing 'port' or 'robot_ip' refuse. The others
    (network_interface, ip_address, remote_ip, host, motion_switcher_client_factory)
    all build a MuJoCo sim under status=success and swallow the kwarg — the
    operator's hardware intent is silently dropped.
"""

from __future__ import annotations

import sys
from strands_robots import Robot
from strands_robots.hardware_robot import _FORWARDABLE_KWARGS, _ADDRESS_FIELDS


# ----- (robot_name, kwarg_name, value, is_hardware_only?) ---------------
# Every kwarg listed below is one `strands_robots` itself declares as a
# driver-only concept. See
#   - strands_robots/hardware_robot.py:220 (_ADDRESS_FIELDS)
#   - strands_robots.drivers.unitree.g1.G1Driver signature (network_interface,
#     motion_switcher_client_factory)
#   - strands_robots.drivers.unitree.go2.Go2Driver signature (network_interface)
CASES = [
    # IN _FORWARDABLE_KWARGS → should refuse (the control group)
    ("so101",       "port",                             "/dev/ttyACM0"),
    ("g1",          "port",                             "192.168.123.161"),
    ("g1",          "robot_ip",                         "192.168.123.161"),

    # NOT in _FORWARDABLE_KWARGS → *silently accepted* (the defect)
    ("g1",          "network_interface",                "eth0"),
    ("g1",          "motion_switcher_client_factory",   lambda: None),
    ("unitree_go2", "network_interface",                "eth0"),
    ("lekiwi",      "remote_ip",                        "192.168.1.100"),  # in _ADDRESS_FIELDS!
    ("reachy_mini", "ip_address",                       "192.168.1.100"),  # in _ADDRESS_FIELDS!
    ("franka",      "host",                             "franka.local"),   # in _ADDRESS_FIELDS!
]


def main() -> int:
    print("_FORWARDABLE_KWARGS has", len(_FORWARDABLE_KWARGS), "entries:")
    print(" ", _FORWARDABLE_KWARGS)
    print()
    print("_ADDRESS_FIELDS declares 4 address kwargs (hardware_robot.py:220):")
    print(" ", _ADDRESS_FIELDS)
    missing = tuple(k for k in _ADDRESS_FIELDS if k not in _FORWARDABLE_KWARGS)
    print(f"  of those, {len(missing)}/4 are MISSING from the guard: {missing}")
    print()

    bad_silent = 0
    for robot_name, kw_name, val in CASES:
        try:
            r = Robot(robot_name, **{kw_name: val})
            built = type(r).__name__
            try:
                r.cleanup()
            except Exception:
                pass
            in_guard = kw_name in _FORWARDABLE_KWARGS
            marker = "   OK (guard caught would be TypeError above)" if in_guard else "   ❌ SILENT (guard missed)"
            print(f"  Robot({robot_name!r}, {kw_name}=...) -> built {built}{marker}")
            if not in_guard:
                bad_silent += 1
        except TypeError as e:
            msg = str(e).replace("\n", " ")[:140]
            in_guard = kw_name in _FORWARDABLE_KWARGS
            marker = "   ✓ guard fired (as expected)" if in_guard else "   ⚠ guard fired (should not have — gap closed?)"
            print(f"  Robot({robot_name!r}, {kw_name}=...) -> TypeError: {msg}{marker}")
        except Exception as e:  # pragma: no cover
            print(f"  Robot({robot_name!r}, {kw_name}=...) -> {type(e).__name__}: {str(e)[:140]}")

    print()
    if bad_silent:
        print(f"FAIL: {bad_silent} hardware-only kwarg(s) silently built a sim. See defect notes.")
        return 1
    print("PASS: every hardware-only kwarg was refused.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
