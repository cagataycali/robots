#!/usr/bin/env python3
"""Test mesh input streaming — leader publishes, follower receives over zenoh."""
import sys
import time
import threading

sys.path.insert(0, ".")

from strands_robots.mesh import Mesh, InputPublisher, InputReceiver, init_mesh

# --- Mock teleoperator (simulates SOLeader reading joint positions) ---
class MockLeader:
    """Simulates a leader arm reading positions at variable angles."""
    def __init__(self):
        self._step = 0
    
    def get_action(self):
        self._step += 1
        # Simulate 6 joints with slowly changing positions
        import math
        t = self._step * 0.02  # 50Hz steps
        return {
            "shoulder_pan.pos": 2048 + int(500 * math.sin(t)),
            "shoulder_lift.pos": 2048 + int(300 * math.cos(t * 0.7)),
            "elbow_flex.pos": 2048 + int(400 * math.sin(t * 1.3)),
            "wrist_flex.pos": 2048 + int(200 * math.cos(t * 0.5)),
            "wrist_roll.pos": 2048 + int(600 * math.sin(t * 2.0)),
            "gripper.pos": 2048 + int(100 * math.sin(t * 0.3)),
        }

# --- Mock follower robot (records received actions) ---
class MockFollower:
    def __init__(self):
        self.received_actions = []
        self.tool_name_str = "mock_follower"
    
    def send_action(self, action):
        self.received_actions.append(action)

# --- Mock robot for mesh init ---
class MockRobotForMesh:
    def __init__(self, name):
        self.tool_name_str = name


print("=" * 60)
print("TEST: Mesh Input Streaming (InputPublisher -> InputReceiver)")
print("=" * 60)

# Create two mesh peers (leader + follower)
leader_robot = MockRobotForMesh("leader_arm")
follower_robot = MockFollower()

leader_mesh = init_mesh(leader_robot, peer_id="leader-test-001", peer_type="robot")
follower_mesh = init_mesh(follower_robot, peer_id="follower-test-001", peer_type="robot")

if not leader_mesh or not follower_mesh:
    print("SKIP: zenoh not available")
    sys.exit(0)

assert leader_mesh.alive, "Leader mesh not alive"
assert follower_mesh.alive, "Follower mesh not alive"
print("[OK] Both meshes started")

# Create teleoperator (mock leader arm)
teleop = MockLeader()

# Start publishing from leader
publisher = InputPublisher(
    mesh=leader_mesh,
    teleoperator=teleop,
    device_name="arm_leader",
    method="arm",
    hz=50.0,
)
publisher.start()
print(f"[OK] Publisher started: {publisher.topic}")

# Start receiving on follower
receiver = InputReceiver(
    mesh=follower_mesh,
    robot=follower_robot,
    source_peer_id="leader-test-001",
    device_name="arm_leader",
)
receiver.start()
print(f"[OK] Receiver started: {receiver.topic}")

# Let it run for 3 seconds
print("\n[...] Streaming for 3 seconds...")
time.sleep(3.0)

# Stop both
pub_stats = publisher.stop()
rcv_stats = receiver.stop()

# Check results
print(f"\n--- Publisher Stats ---")
print(f"  Frames sent: {pub_stats['frames']}")
print(f"  Hz actual: {pub_stats['hz_actual']:.1f}")
print(f"  Errors: {pub_stats['errors']}")

print(f"\n--- Receiver Stats ---")
print(f"  Frames received: {rcv_stats['frames_received']}")
print(f"  Hz actual: {rcv_stats['hz_actual']:.1f}")
print(f"  Drops: {rcv_stats['drops']}")
print(f"  Errors: {rcv_stats['errors']}")

# Verify data integrity
print(f"\n--- Data Integrity ---")
print(f"  Actions recorded by follower: {len(follower_robot.received_actions)}")
if follower_robot.received_actions:
    sample = follower_robot.received_actions[-1]
    print(f"  Last action keys: {list(sample.keys())}")
    print(f"  Last action sample: shoulder_pan={sample.get('shoulder_pan.pos', '?')}")

# Assertions
assert pub_stats["frames"] > 100, f"Publisher too slow: {pub_stats['frames']} frames in 3s"
assert rcv_stats["frames_received"] > 50, f"Receiver too few frames: {rcv_stats['frames_received']}"
assert pub_stats["errors"] == 0, f"Publisher errors: {pub_stats['errors']}"
assert rcv_stats["errors"] == 0, f"Receiver errors: {rcv_stats['errors']}"
assert len(follower_robot.received_actions) > 50, f"Follower got too few actions"

# Check that actions have correct keys
sample_action = follower_robot.received_actions[0]
assert "shoulder_pan.pos" in sample_action, f"Missing key in action: {sample_action.keys()}"
assert "gripper.pos" in sample_action, f"Missing gripper key in action: {sample_action.keys()}"

# Calculate delivery ratio
delivery_pct = (rcv_stats["frames_received"] / pub_stats["frames"]) * 100
print(f"\n  Delivery ratio: {delivery_pct:.1f}%")

# Cleanup
leader_mesh.stop()
follower_mesh.stop()

print("\n" + "=" * 60)
print(f"PASSED! {pub_stats['frames']} frames published @ {pub_stats['hz_actual']:.1f}Hz")
print(f"        {rcv_stats['frames_received']} frames received @ {rcv_stats['hz_actual']:.1f}Hz")
print(f"        Delivery: {delivery_pct:.1f}%, Drops: {rcv_stats['drops']}")
print("=" * 60)
