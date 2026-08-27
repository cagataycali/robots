#!/usr/bin/env python3
"""Join the fleet mesh with a real Unitree G1 via the native CycloneDDS driver.

The G1 does not speak the lerobot serial bus; it speaks raw Unitree IDL over
CycloneDDS - a bus lerobot's driver cannot reach. This example wires the
native driver into the mesh: ``Robot("g1", mode="real", driver="strands", ...)``
returns :class:`~strands_robots.drivers.g1.G1Driver`, and ``Mesh(...)`` wraps
it so the fleet dashboard's presence card, IMU chip, battery chip and lidar
summary appear the moment the driver's DDS callbacks deliver their first
messages.

Dependencies (on the robot's control PC): ``pip install
"strands-robots[mesh]" cyclonedds unitree_sdk2py``. The SDK is lazy-imported by
the driver, so ``strands_robots`` remains importable on a machine without it -
that is what makes every headless test pass.

Runtime: keeps running until Ctrl-C. The DDS subscribers deliver at ~1 kHz
(low-state), 10 Hz (lidar) and 1-2 Hz (battery); the mesh pacer decides what
to publish and at what rate.

Hardware note: **this example only makes sense against a real G1**. On Thor,
in CI or on any laptop without CycloneDDS on the same LAN as the robot, the
DDS init fails with a named reason and the driver stays in the
"usable but not connected" state. That is deliberate - a mesh peer that never
connects is still a valid peer, so ``Mesh(...)`` publishes an "offline" card
instead of raising.
"""

from __future__ import annotations

import os
import signal
import sys
import time

# The driver never touches MuJoCo but the mesh transport bootstrap does, so
# stay consistent with the other mesh examples in this directory.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "1")

from strands_robots import Robot
from strands_robots.mesh import Mesh


def main() -> int:
    """Build the G1 driver, wrap in Mesh, run until Ctrl-C."""
    # 192.168.123.161 is the G1's default eth0 address for user-side CycloneDDS.
    # Adjust to match your unit; the driver only records it for logging, the
    # actual bus binding is by network interface below.
    robot_ip = os.environ.get("G1_IP", "192.168.123.161")
    network_interface = os.environ.get("G1_NIC", "eth0")
    peer_id = os.environ.get("G1_PEER_ID", "neon")

    driver = Robot(
        "g1",
        mode="real",
        driver="strands",  # the seam from issue #353 chooses this over lerobot
        port=robot_ip,
        network_interface=network_interface,
    )

    err = driver.connect_eagerly()
    if err is not None:
        # A caller who wants a hard failure raises here. This example prefers
        # to keep going so the dashboard card still appears - the "offline"
        # state is what the operator needs to see, and hiding it would leave
        # a real production console blank on a real production outage.
        print(f"[neon] driver did not connect: {err}", file=sys.stderr)
        print("[neon] continuing so the mesh peer still appears as offline.", file=sys.stderr)

    mesh = Mesh(driver, peer_id=peer_id)
    mesh.start()
    print(f"[neon] joined mesh as peer_id={peer_id!r}; robot ip={robot_ip!r}")
    print("[neon] the fleet dashboard should now show a G1 card with IMU/battery/lidar chips.")
    print("[neon] press Ctrl-C to leave the mesh cleanly.")

    stop = {"pressed": False}

    def _handler(_signum: int, _frame: object) -> None:
        stop["pressed"] = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    try:
        while not stop["pressed"]:
            time.sleep(0.5)
    finally:
        mesh.stop()
        driver.cleanup()
        print("[neon] mesh peer left; DDS subscribers released.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
