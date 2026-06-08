#!/usr/bin/env bash
# Spawn a sim robot on the mesh for dashboard testing.
export STRANDS_MESH_AUTH_MODE=none
export STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1
export STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1
export STRANDS_MESH_CAMERA_HZ="${STRANDS_MESH_CAMERA_HZ:-10}"
ROBOT="${1:-so100}"
exec .venv/bin/python -c "
import time
from strands_robots import Robot
r = Robot('$ROBOT')
print('spawned $ROBOT on mesh; ctrl-c to stop')
while True: time.sleep(1)
"
