#!/usr/bin/env bash
# Start a real turtlesim node, then let a Strands Agent drive a square via use_ros.
source /opt/ros/jazzy/setup.bash
set -euo pipefail

echo "== starting turtlesim (headless) =="
ros2 run turtlesim turtlesim_node >/tmp/turtle.log 2>&1 &
TURTLE_PID=$!
trap 'kill ${TURTLE_PID} 2>/dev/null || true' EXIT

for _ in $(seq 1 20); do
    grep -q "Spawning turtle" /tmp/turtle.log 2>/dev/null && break
    sleep 0.5
done
grep -q "Spawning turtle" /tmp/turtle.log || { echo "turtlesim failed:"; cat /tmp/turtle.log; exit 2; }

python3 examples/ros2/use_ros/agent_drive.py
