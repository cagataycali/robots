#!/usr/bin/env bash
# Leader -> Follower physical teleop for two SO-101 arms on the mesh.
#
# Usage:
#   ./start_leader_follower.sh [LEADER_PORT] [FOLLOWER_PORT] [LEADER_ID] [FOLLOWER_ID]
# Defaults: leader=/dev/ttyACM1 follower=/dev/ttyACM3
#
# Both arms MUST be calibrated first (via the dashboard Calibrate panel, or
# lerobot-calibrate). The leader publishes joint positions to the mesh; the
# follower subscribes and mirrors them on its own hardware.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
export STRANDS_MESH_AUTH_MODE=none
export STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1
export STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1
LEADER_PORT="${1:-/dev/ttyACM1}"
FOLLOWER_PORT="${2:-/dev/ttyACM3}"
LEADER_ID="${3:-leader}"
FOLLOWER_ID="${4:-follower}"
exec .venv/bin/python - "$LEADER_PORT" "$FOLLOWER_PORT" "$LEADER_ID" "$FOLLOWER_ID" <<'PY'
import sys, time
from strands_robots import Robot
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

lead_port, foll_port, lead_id, foll_id = sys.argv[1:5]
print(f"Leader  : {lead_port} (id={lead_id})")
print(f"Follower: {foll_port} (id={foll_id})")

# Follower is a full strands Robot on the mesh (so the dashboard sees it
# and it can receive teleop frames).
follower = Robot("so101", mode="real", port=foll_port, id=foll_id)
print("follower peer:", follower.peer_id)
if not follower.robot.is_calibrated:
    print("\n*** FOLLOWER NOT CALIBRATED -- calibrate via dashboard first ***")

# Leader is a lerobot teleoperator (read-only joint source).
lead = SO101Leader(SO101LeaderConfig(port=lead_port, id=lead_id))
lead.connect(calibrate=False)
if not lead.is_calibrated:
    print("*** LEADER NOT CALIBRATED -- calibrate via dashboard first ***")
print("leader connected:", lead.is_connected)

# Publish leader -> mesh; follower receives + applies to its hardware.
follower.start_teleop_receive(source_peer_id=follower.peer_id, device_name="leader")
follower.start_teleop_publish(teleoperator=lead, device_name="leader", method="arm", hz=50.0)
print("\nTELEOP LIVE: move the LEADER arm; the FOLLOWER mirrors it.")
print("Watch in the dashboard (http://<host>:7860). Ctrl-C to stop.\n")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nstopping teleop...")
    follower.stop_teleop()
    lead.disconnect()
PY
