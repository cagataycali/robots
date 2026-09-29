#!/usr/bin/env python3
"""One command to one robot over AWS IoT Core Direct Messaging, and what an offline robot costs.

Goal: Run this file twice, as the robot and as the operator, and watch a Mesh.send
travel point to point: delivered to that one client with a PUBACK, answered the
same way, and reported offline in one round trip once the robot is gone.

Dependencies: pip install "strands-robots[mesh-iot]"; AWS credentials; identities from
              `strands-robots iot provision-robot so101-arm-01` and `... provision-operator ops-1`.
Usage:  python examples/mesh/iot_direct_rpc.py --role robot                        # terminal 1
        python examples/mesh/iot_direct_rpc.py --role operator --target so101-arm-01  # terminal 2
Expected output: three replies with their delivery verdict and round trip, then the
                 offline verdict after the robot exits, in well under a second.
Runtime: the robot answers for 20 s; the operator exits after the offline verdict.
"""

import argparse
import os
import time

os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "1")

from strands_robots.mesh.core import Mesh


class Reachable:
    """The smallest peer that can be asked how it is doing."""

    tool_name_str = "iot-direct-example"

    def get_task_status(self) -> dict[str, str]:
        return {"status": "idle"}


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--role", choices=("robot", "operator"), required=True)
parser.add_argument("--target", default="", help="the robot's peer id (operator role)")
args = parser.parse_args()

peer_id = os.environ["STRANDS_IOT_THING_NAME"]
mesh = Mesh(Reachable(), peer_id=peer_id, peer_type=args.role)
mesh.start()
if not mesh.alive:
    raise SystemExit("mesh did not start: check STRANDS_MESH_BACKEND=iot and the exported certificate variables")
try:
    if args.role == "robot":
        print(f"{peer_id}: answering commands for 20 s")
        time.sleep(20)
    else:
        time.sleep(2)  # one heartbeat, so the robot's presence names its IoT client id
        for _ in range(3):
            t0 = time.monotonic()
            reply = mesh.send(args.target, {"action": "status"}, timeout=5.0)
            print(f"{reply.get('result')} via={reply['delivery']['via']} rtt={(time.monotonic() - t0) * 1000:.0f} ms")
        print("waiting for the robot to leave...")
        time.sleep(22)
        t0 = time.monotonic()
        reply = mesh.send(args.target, {"action": "status"}, timeout=30.0)
        print(f"{reply.get('error') or reply.get('status')} in {(time.monotonic() - t0) * 1000:.0f} ms (budget 30 s)")
finally:
    mesh.stop()  # the session runs on non-daemon threads; release it or the script never exits
