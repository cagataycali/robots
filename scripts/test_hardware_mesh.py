#!/usr/bin/env python3
"""End-to-end hardware mesh validation with 2x SO-101 + 2x cameras.

Tests the zenoh mesh layer with real servo hardware. Cameras are tested
separately via direct OpenCV (LeRobot's strict validation doesn't work
with all V4L2 drivers).

Usage:
    STRANDS_MESH_CAMERA_HZ=2 .testvenv/bin/python scripts/test_hardware_mesh.py
"""

from __future__ import annotations

import sys
import time
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_hw_mesh")


def make_so_follower(port: str, robot_id: str = "default"):
    """Create and connect an SOFollower (no cameras — tested separately)."""
    from lerobot.robots.so_follower import SOFollowerConfig, SOFollower

    config = SOFollowerConfig(port=port, cameras={})
    config.id = robot_id
    config.calibration_dir = None

    robot = SOFollower(config)
    robot.connect(calibrate=False)
    return robot


def test_cameras_direct():
    """Test cameras directly via OpenCV (bypasses LeRobot strict validation)."""
    import cv2
    results = {}
    for path in ["/dev/video0", "/dev/video2"]:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                results[path] = {"ok": True, "shape": list(frame.shape)}
            else:
                results[path] = {"ok": False, "error": "read failed"}
            cap.release()
        else:
            results[path] = {"ok": False, "error": "cannot open"}
    return results


def main() -> int:
    logger.info("=" * 60)
    logger.info("HARDWARE MESH VALIDATION — 2x SO-101 + 2x Camera")
    logger.info("=" * 60)

    # --- Step 0: Check imports ---
    logger.info("[0/7] Checking imports...")
    try:
        from strands_robots.mesh import Mesh, init_mesh
        from strands_robots.mesh_session import get_peers, get_session
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return 1

    session = get_session()
    if session is None:
        logger.error("Zenoh session failed — eclipse-zenoh installed?")
        return 1
    logger.info("[0/7] OK — all imports + zenoh session ready")

    # --- Step 1: Test cameras directly ---
    logger.info("[1/7] Testing cameras via OpenCV...")
    cam_results = test_cameras_direct()
    for path, res in cam_results.items():
        if res["ok"]:
            logger.info(f"       {path}: OK shape={res['shape']}")
        else:
            logger.warning(f"       {path}: FAIL — {res['error']}")

    working_cams = {p: r for p, r in cam_results.items() if r["ok"]}
    logger.info(f"[1/7] {len(working_cams)}/2 cameras working")

    # --- Step 2: Connect arm A ---
    logger.info("[2/7] Connecting SO-101 arm A (/dev/ttyACM0)...")
    try:
        arm_a = make_so_follower(port="/dev/ttyACM0", robot_id="arm_a")
        logger.info(f"[2/7] OK — arm_a connected: {arm_a.is_connected}")
    except Exception as e:
        logger.error(f"[2/7] FAILED — arm A: {e}")
        import traceback; traceback.print_exc()
        return 1

    # --- Step 3: Connect arm B ---
    logger.info("[3/7] Connecting SO-101 arm B (/dev/ttyACM1)...")
    try:
        arm_b = make_so_follower(port="/dev/ttyACM1", robot_id="arm_b")
        logger.info(f"[3/7] OK — arm_b connected: {arm_b.is_connected}")
    except Exception as e:
        logger.error(f"[3/7] FAILED — arm B: {e}")
        import traceback; traceback.print_exc()
        arm_a.disconnect()
        return 1

    # --- Step 4: Read joint positions ---
    logger.info("[4/7] Reading joint observations...")
    try:
        obs_a = arm_a.get_observation()
        obs_b = arm_b.get_observation()
        logger.info(f"       arm_a joints: {list(obs_a.keys())}")
        logger.info(f"       arm_b joints: {list(obs_b.keys())}")
        for k, v in obs_a.items():
            logger.info(f"         arm_a.{k} = {v}")
    except Exception as e:
        logger.warning(f"[4/7] observation error: {e}")

    # --- Step 5: Attach mesh to both arms ---
    logger.info("[5/7] Attaching mesh peers...")

    class MeshableRobot:  # noqa
        """Adapter for init_mesh."""
        def __init__(self, robot, tool_name: str, camera_names: list):
            self.robot = robot
            self.tool_name_str = tool_name
            self.is_connected = robot.is_connected
            self.name = "so101"
            # Fake camera config for mesh advertising
            self._camera_names = camera_names

        @property
        def config(self):
            """Config-like object with cameras dict for heartbeat advertisement."""
            class _Cfg:
                pass
            cfg = _Cfg()
            cfg.cameras = {n: True for n in self._camera_names}
            return cfg

        def get_observation(self):
            return self.robot.get_observation()

    wrapper_a = MeshableRobot(arm_a, "so101_arm_a", ["cam_front"])
    wrapper_b = MeshableRobot(arm_b, "so101_arm_b", ["cam_wrist"])

    mesh_a = init_mesh(wrapper_a, peer_id="so101-arm-a", peer_type="robot")
    mesh_b = init_mesh(wrapper_b, peer_id="so101-arm-b", peer_type="robot")

    if mesh_a is None or mesh_b is None:
        logger.error("[5/7] FAILED — mesh init returned None")
        arm_a.disconnect(); arm_b.disconnect()
        return 1

    logger.info(f"[5/7] OK — mesh_a={mesh_a.peer_id}, mesh_b={mesh_b.peer_id}")

    # --- Step 6: Wait for presence discovery ---
    logger.info("[6/7] Waiting for presence heartbeats (up to 12s)...")
    time.sleep(3)

    deadline = time.time() + 12
    all_peers = []
    hw_peers = []
    while time.time() < deadline:
        all_peers = get_peers()
        hw_peers = [p for p in all_peers if p.get("hw")]
        if len(hw_peers) >= 2:
            break
        time.sleep(0.5)

    logger.info(f"[6/7] Total peers discovered: {len(all_peers)}")
    for p in all_peers:
        caps = p
        logger.info(
            f"       - {p.get('peer_id')}: hw={caps.get('hw')}, "
            f"connected={caps.get('connected')}, "
            f"cameras={caps.get('cameras', [])}"
        )

    if len(hw_peers) < 2:
        logger.error(f"[6/7] FAILED — expected >=2 hw peers, found {len(hw_peers)}")
        _cleanup(mesh_a, mesh_b, arm_a, arm_b)
        return 1
    logger.info("[6/7] OK — both arms visible on mesh")

    # --- Step 7: Cross-visibility ---
    logger.info("[7/7] Bidirectional peer visibility...")
    time.sleep(2)

    a_sees_b = mesh_b.peer_id in mesh_a.peers
    b_sees_a = mesh_a.peer_id in mesh_b.peers
    logger.info(f"[7/7] arm_a sees arm_b: {a_sees_b}, arm_b sees arm_a: {b_sees_a}")

    # Camera advertisement check
    cameras_seen = {}
    for p in hw_peers:
        cams = p.get("cameras", [])
        cameras_seen[p.get("peer_id", "?")] = cams
    total_cams = sum(len(v) for v in cameras_seen.values())
    logger.info(f"[7/7] Cameras advertised: {json.dumps(cameras_seen)}")

    # --- Summary ---
    passed = len(hw_peers) >= 2
    logger.info("=" * 60)
    logger.info(f"RESULT: {'PASSED' if passed else 'FAILED'}")
    logger.info(f"  Hardware cameras (OpenCV):  {len(working_cams)}/2")
    logger.info(f"  Arms on mesh:              {len(hw_peers)}/2")
    logger.info(f"  Cameras advertised:        {total_cams}")
    logger.info(f"  Bidirectional:             a->b={a_sees_b}, b->a={b_sees_a}")
    logger.info(f"  Peer IDs:                  {[p.get('peer_id') for p in hw_peers]}")
    logger.info("=" * 60)

    _cleanup(mesh_a, mesh_b, arm_a, arm_b)
    return 0 if passed else 1


def _cleanup(*items):
    """Stop meshes and disconnect robots."""
    for item in items:
        try:
            if hasattr(item, "stop"):
                item.stop()
            elif hasattr(item, "disconnect"):
                item.disconnect()
        except Exception as e:
            logger.debug(f"Cleanup: {e}")


if __name__ == "__main__":
    sys.exit(main())
