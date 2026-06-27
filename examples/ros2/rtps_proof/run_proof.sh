#!/usr/bin/env bash
# Orchestrate the cross-process proof inside the container:
#   1. start a real turtlesim node (CycloneDDS RMW) in the background,
#   2. run drive_and_verify.py (our use_rtps publisher + pose assertion).
# Source ROS *before* enabling nounset - ROS setup scripts reference unbound vars.
source /opt/ros/jazzy/setup.bash
set -euo pipefail

echo "== starting turtlesim (headless, CycloneDDS RMW) =="
ros2 run turtlesim turtlesim_node >/tmp/turtle.log 2>&1 &
TURTLE_PID=$!
trap 'kill ${TURTLE_PID} 2>/dev/null || true' EXIT

# Wait for the node to spawn turtle1.
for _ in $(seq 1 20); do
    if grep -q "Spawning turtle" /tmp/turtle.log 2>/dev/null; then break; fi
    sleep 0.5
done
grep -q "Spawning turtle" /tmp/turtle.log || { echo "turtlesim failed to start:"; cat /tmp/turtle.log; exit 2; }
echo "   turtlesim up: $(grep 'Spawning turtle' /tmp/turtle.log | head -1)"

python3 examples/ros2/rtps_proof/drive_and_verify.py
