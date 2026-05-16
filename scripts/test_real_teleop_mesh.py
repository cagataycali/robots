#!/usr/bin/env python3
"""Real hardware test: left arm (leader) publishes over mesh, right arm (follower) receives."""
import sys
import time

sys.path.insert(0, ".")

from strands_robots.mesh import Mesh, InputPublisher, InputReceiver, init_mesh

print("=" * 60)
print("REAL HARDWARE: Mesh Teleoperation (left -> right over zenoh)")
print("=" * 60)

# Connect leader teleoperator
print("\n[1/4] Connecting leader arm (ACM0)...")
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderConfig

leader_config = SOLeaderConfig(port="/dev/ttyACM0")
leader_config.id = "left_arm"
leader_config.calibration_dir = None
leader = SOLeader(leader_config)
leader.connect()
print(f"  Connected: {leader}")

# Connect follower robot
print("\n[2/4] Connecting follower arm (ACM1)...")
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

follower_config = SOFollowerConfig(port="/dev/ttyACM1")
follower_config.id = "right_arm"
follower_config.calibration_dir = None
follower = SOFollower(follower_config)
follower.connect(calibrate=False)
print(f"  Connected: {follower}")

# Create mesh peers
print("\n[3/4] Starting mesh peers...")

class LeaderWrapper:
    tool_name_str = "so101_leader"

leader_mesh = init_mesh(LeaderWrapper(), peer_id="left-arm-leader", peer_type="robot")
follower_mesh = init_mesh(follower, peer_id="right-arm-follower", peer_type="robot")

if not leader_mesh or not follower_mesh:
    print("FAIL: zenoh not available")
    leader.disconnect()
    follower.disconnect()
    sys.exit(1)

print(f"  Leader mesh: {leader_mesh.peer_id}")
print(f"  Follower mesh: {follower_mesh.peer_id}")

# Start input streaming
print("\n[4/4] Starting mesh teleoperation...")
publisher = InputPublisher(
    mesh=leader_mesh,
    teleoperator=leader,
    device_name="arm",
    method="arm",
    hz=50.0,
)
publisher.start()

receiver = InputReceiver(
    mesh=follower_mesh,
    robot=follower,
    source_peer_id="left-arm-leader",
    device_name="arm",
)
receiver.start()

print(f"\n  Publisher: {publisher.topic}")
print(f"  Receiver: {receiver.topic}")
print(f"\n  >>> MOVE THE LEFT ARM! Right arm will follow over zenoh <<<")
print(f"  Running for 5 seconds...")

time.sleep(5.0)

# Stop and report
pub_stats = publisher.stop()
rcv_stats = receiver.stop()

print(f"\n--- Results ---")
print(f"  Published: {pub_stats['frames']} frames @ {pub_stats['hz_actual']:.1f}Hz")
print(f"  Received:  {rcv_stats['frames_received']} frames @ {rcv_stats['hz_actual']:.1f}Hz")
print(f"  Drops: {rcv_stats['drops']}")
print(f"  Errors: pub={pub_stats['errors']}, rcv={rcv_stats['errors']}")

delivery_pct = (rcv_stats["frames_received"] / max(pub_stats["frames"], 1)) * 100
print(f"  Delivery: {delivery_pct:.1f}%")

# Cleanup
leader_mesh.stop()
follower_mesh.stop()
leader.disconnect()
follower.disconnect()

print(f"\n{'=' * 60}")
if pub_stats["frames"] > 100 and rcv_stats["frames_received"] > 50:
    print("PASSED! Real hardware teleoperation over mesh works!")
else:
    print("PARTIAL: Streaming worked but frame count low")
print(f"{'=' * 60}")
