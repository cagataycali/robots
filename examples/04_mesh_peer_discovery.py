#!/usr/bin/env python3
"""Join the robot mesh and discover peers on the local network.

Goal: Show that Robot() with mesh=True gives automatic peer discovery via
Zenoh. Every robot that joins the mesh is visible to every other - no DHCP,
no config server, no manual IP lists.

Dependencies: pip install "strands-robots[sim-mujoco,mesh]"
Expected output: Prints local robot info and discovered peer list.
Runtime: ~3 seconds (waits for one round of peer heartbeats, then exits and
         releases the mesh session).

Note: Set STRANDS_MESH_LOCAL_DEV=1 to skip TLS for local development.
      Set STRANDS_MESH=0 to disable mesh entirely (the example still runs
      but reports empty peer lists).
"""

import os
import sys
import time

os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "1")
os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

from strands_robots import Robot
from strands_robots.mesh import get_local_robots, get_peers
from strands_robots.mesh.session import HEARTBEAT_HZ

# Robot with mesh=True (default) auto-joins the mesh on creation. The factory
# also builds the world and adds the "so100" robot, so no create_world/add_robot
# is needed. Use mesh=False in CI or when Zenoh is unavailable.
use_mesh = os.environ.get("STRANDS_MESH", "true").lower() != "0"
sim = Robot("so100", mesh=use_mesh, peer_id="example-arm-01")

# Peers announce themselves by heartbeat, not on demand: a robot in another
# process shows up here only after its next beat lands (~1/HEARTBEAT_HZ s).
# Reading the registry straight after Robot() returns sees nobody but us.
time.sleep(3 / HEARTBEAT_HZ)

# Query the mesh - see who is online.
local = get_local_robots()
peers = get_peers()

print(f"Local robots in this process: {list(local.keys())}")

# Each peer row is PeerInfo.to_dict(): peer_id, type ("robot" | "sim" |
# "agent"), hostname, age (seconds since its last heartbeat), reachable, plus
# whatever the peer put in its presence payload. This process's own robots
# heartbeat over the same session, so they show up here too - split them out,
# otherwise a single process reports itself as a discovered fleet.
own = [p for p in peers if p.get("peer_id") in local]
others = [p for p in peers if p.get("peer_id") not in local]
print(f"Discovered mesh peers: {len(others)} (plus {len(own)} of our own)")
for peer in others:
    print(f"  {peer.get('peer_id', '?')}: type={peer.get('type', '?')} age={peer.get('age', '?')}s")
if not others:
    print("  (no other robot on this LAN - start this script in a second terminal to see one)")

# Cleanup. Release the Zenoh session through the attribute the Robot factory
# assigns (`sim.mesh`) - the same one strands_robots.robot's own teardown reads.
# The session runs on non-daemon threads, so a cleanup that silently reads a
# name the SDK never sets leaves this script running forever.
mesh = getattr(sim, "mesh", None)
if mesh:
    mesh.stop()
