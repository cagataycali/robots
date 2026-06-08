#!/usr/bin/env bash
# Local dev launcher for the Strands Robots dashboard.
# Dev-only mesh posture: wire security OFF, permissive ACL acknowledged.
# DO NOT use these env vars in production.
export STRANDS_MESH_AUTH_MODE=none
export STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1
export STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1
export STRANDS_MESH_CAMERA_HZ="${STRANDS_MESH_CAMERA_HZ:-10}"
exec .venv/bin/python -m strands_robots.dashboard --port "${1:-7860}"
