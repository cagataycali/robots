"""Agent tool for controlling a Pollen Robotics Reachy Mini.

Communicates with the Reachy Mini daemon via its REST API (http://<ip>:8000).
The daemon also streams telemetry at 50 Hz via Zenoh (reachy_mini/**).

Actions:
    status           - Get robot state (head pose, joints, motors)
    get_joints       - Read current joint positions and head pose
    get_imu          - Read IMU data via Zenoh subscription
    goto             - Smooth interpolated move (head pose + antennas + duration)
    set_target       - Set instantaneous target (head pose, no interpolation)
    wake_up          - Wake up (enable motors, move to neutral pose)
    sleep            - Go to sleep (rest position, then disable motors)
    set_motor_mode   - Set mode: enabled, disabled, gravity_compensation
    stop_move        - Stop any running goto/move
    play_move        - Play a recorded move dataset
    list_moves       - List available recorded move datasets
    apps             - List/start/stop installed apps

Joint layout (head, 7 DOF):
    Body yaw | Neck (roll, pitch, yaw) | Eyes (L pitch, L yaw, R pitch, R yaw)
    Note: head_pose is expressed as x, y, z, roll, pitch, yaw

Antenna positions: [right_antenna, left_antenna] in radians

Example:
    use_reachy_mini(action="status")
    use_reachy_mini(action="wake_up")
    use_reachy_mini(action="goto", head_pose='{"roll": 0, "pitch": 0.3, "yaw": 0}', duration=1.0)
    use_reachy_mini(action="set_target", antennas="[0.5, -0.5]")
    use_reachy_mini(action="sleep")
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests
from strands import tool

logger = logging.getLogger(__name__)

# Default robot address
_DEFAULT_HOST = "192.168.1.2"
_DEFAULT_PORT = 8000


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": text}]}


def _ok(text: str) -> dict[str, Any]:
    return {"status": "success", "content": [{"text": text}]}


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _get(host: str, port: int, path: str, timeout: float = 5.0) -> dict:
    """GET request to the Reachy Mini API."""
    url = f"{_base_url(host, port)}{path}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _post(host: str, port: int, path: str, body: dict | None = None, timeout: float = 5.0) -> dict | None:
    """POST request to the Reachy Mini API."""
    url = f"{_base_url(host, port)}{path}"
    if body is not None:
        resp = requests.post(url, json=body, timeout=timeout)
    else:
        resp = requests.post(url, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return None


@tool
def use_reachy_mini(
    action: str,
    host: str = "192.168.1.2",
    port: int = 8000,
    head_pose: str = "",
    antennas: str = "",
    body_yaw: str = "",
    duration: float = 0.5,
    mode: str = "",
    app_name: str = "",
    dataset_name: str = "",
    move_name: str = "",
) -> dict[str, Any]:
    """Control a Pollen Robotics Reachy Mini robot via its REST API.

    The Reachy Mini is a desktop robot with a 7-DOF head and 2 antennas.
    Commands go through the HTTP API at http://<host>:8000.

    Args:
        action: One of: status, get_joints, get_imu, goto, set_target,
                wake_up, sleep, set_motor_mode, stop_move, play_move,
                list_moves, apps, start_app, stop_app
        host: Robot IP address (default: 192.168.1.2)
        port: API port (default: 8000)
        head_pose: JSON object with target pose {roll, pitch, yaw} or
                   flattened 4x4 matrix (16 floats)
        antennas: JSON array [right, left] antenna angles in radians
        body_yaw: Body yaw angle in radians (as string)
        duration: Goto duration in seconds (default: 0.5)
        mode: Motor mode: 'enabled', 'disabled', or 'gravity_compensation'
        app_name: App name for start_app/stop_app actions
        dataset_name: Dataset name for play_move
        move_name: Move name within dataset for play_move

    Returns:
        Dict with status and content
    """
    try:
        if action == "status":
            state = _get(host, port, "/api/state/full")
            daemon = _get(host, port, "/api/daemon/status")

            hp = state.get("head_pose", {})
            ant = state.get("antennas_position", [])
            by = state.get("body_yaw", 0)
            ctrl = state.get("control_mode", "?")

            text = (
                f"Reachy Mini ({host}:{port})\n"
                f"  State: {daemon.get('state', '?')}\n"
                f"  Version: {daemon.get('version', '?')}\n"
                f"  Motors: {ctrl}\n"
                f"  Wireless: {daemon.get('wireless_version', '?')}\n"
                f"  Head pose: roll={hp.get('roll', 0):.3f} pitch={hp.get('pitch', 0):.3f} yaw={hp.get('yaw', 0):.3f}\n"
                f"  Head xyz: x={hp.get('x', 0):.4f} y={hp.get('y', 0):.4f} z={hp.get('z', 0):.4f}\n"
                f"  Body yaw: {by:.4f} rad\n"
                f"  Antennas [R, L]: {[round(a, 3) for a in ant]}\n"
                f"  Control loop: ~{daemon.get('backend_status', {}).get('control_loop_stats', {}).get('mean_control_loop_frequency', 0):.1f} Hz"
            )
            return _ok(text)

        elif action == "get_joints":
            state = _get(host, port, "/api/state/full")
            hp = state.get("head_pose", {})
            ant = state.get("antennas_position", [])
            by = state.get("body_yaw", 0)
            ts = state.get("timestamp", "?")
            return _ok(
                f"Joint State (at {ts}):\n"
                f"  Head pose: roll={hp.get('roll', 0):.4f} pitch={hp.get('pitch', 0):.4f} yaw={hp.get('yaw', 0):.4f}\n"
                f"  Head position: x={hp.get('x', 0):.4f} y={hp.get('y', 0):.4f} z={hp.get('z', 0):.4f}\n"
                f"  Body yaw: {by:.4f}\n"
                f"  Antennas [R, L]: {[round(a, 4) for a in ant]}"
            )

        elif action == "get_imu":
            # IMU is only available via Zenoh subscription
            try:
                import time

                import zenoh

                session = zenoh.open(zenoh.Config())
                samples = []

                def cb(s):
                    samples.append(json.loads(s.payload.to_bytes().decode("utf-8")))

                sub = session.declare_subscriber("reachy_mini/imu_data", cb)
                time.sleep(0.2)
                sub.undeclare()
                session.close()

                if samples:
                    imu = samples[-1]
                    return _ok(
                        f"IMU Data:\n"
                        f"  Accelerometer: {[round(x, 3) for x in imu.get('accelerometer', [])]}\n"
                        f"  Gyroscope: {[round(x, 5) for x in imu.get('gyroscope', [])]}\n"
                        f"  Quaternion: {[round(x, 4) for x in imu.get('quaternion', [])]}\n"
                        f"  Temperature: {imu.get('temperature')} C"
                    )
                return _ok("No IMU data received (zenoh timeout)")
            except ImportError:
                return _err("IMU requires eclipse-zenoh. pip install eclipse-zenoh")

        elif action == "goto":
            body: dict[str, Any] = {"duration": duration}
            if head_pose:
                pose = json.loads(head_pose)
                if isinstance(pose, dict):
                    # RPY format -> convert to flattened 4x4 for the API
                    body["head"] = pose
                elif isinstance(pose, list):
                    body["head"] = pose
            if antennas:
                body["antennas"] = json.loads(antennas)
            if body_yaw:
                body["body_yaw"] = float(body_yaw)

            if len(body) == 1:  # only duration
                return _err("Provide at least one of: head_pose, antennas, body_yaw")

            _post(host, port, "/api/move/goto", body)
            return _ok(f"Goto started (duration={duration}s): {json.dumps(body)[:200]}")

        elif action == "set_target":
            target_body: dict[str, Any] = {}
            if head_pose:
                target_body["head"] = json.loads(head_pose)
            if antennas:
                target_body["antennas"] = json.loads(antennas)
            if body_yaw:
                target_body["body_yaw"] = float(body_yaw)

            if not target_body:
                return _err("Provide at least one of: head_pose, antennas, body_yaw")

            _post(host, port, "/api/move/set_target", target_body)
            return _ok(f"Target set: {json.dumps(target_body)[:200]}")

        elif action == "wake_up":
            _post(host, port, "/api/move/play/wake_up")
            return _ok("Wake up command sent")

        elif action == "sleep":
            _post(host, port, "/api/move/play/goto_sleep")
            return _ok("Sleep command sent")

        elif action == "set_motor_mode":
            if not mode:
                return _err("'mode' required: 'enabled', 'disabled', 'gravity_compensation'")
            valid_modes = ("enabled", "disabled", "gravity_compensation")
            if mode not in valid_modes:
                return _err(f"Invalid mode '{mode}'. Use: {valid_modes}")
            _post(host, port, f"/api/motors/set_mode/{mode}")
            return _ok(f"Motor mode set to: {mode}")

        elif action == "stop_move":
            _post(host, port, "/api/move/stop")
            return _ok("All moves stopped")

        elif action == "play_move":
            if not dataset_name or not move_name:
                return _err("'dataset_name' and 'move_name' required for play_move")
            _post(host, port, f"/api/move/play/recorded-move-dataset/{dataset_name}/{move_name}")
            return _ok(f"Playing move: {dataset_name}/{move_name}")

        elif action == "list_moves":
            ds = dataset_name or "default"
            try:
                moves = _get(host, port, f"/api/move/recorded-move-datasets/list/{ds}")
                return _ok(f"Moves in '{ds}':\n{json.dumps(moves, indent=2)}")
            except requests.HTTPError as e:
                return _err(f"Dataset '{ds}' not found: {e}")

        elif action == "apps":
            apps = _get(host, port, "/api/apps/list-available")
            current = _get(host, port, "/api/apps/current-app-status")
            text = f"Current app: {json.dumps(current)}\n\nAvailable apps:\n"
            if isinstance(apps, list):
                for app in apps:
                    name = app.get("name", "?") if isinstance(app, dict) else str(app)
                    text += f"  - {name}\n"
            else:
                text += json.dumps(apps, indent=2)[:500]
            return _ok(text)

        elif action == "start_app":
            if not app_name:
                return _err("'app_name' required")
            _post(host, port, f"/api/apps/start-app/{app_name}")
            return _ok(f"App '{app_name}' started")

        elif action == "stop_app":
            _post(host, port, "/api/apps/stop-current-app")
            return _ok("Current app stopped")

        else:
            return _err(
                f"Unknown action: '{action}'. Valid: status, get_joints, get_imu, "
                "goto, set_target, wake_up, sleep, set_motor_mode, stop_move, "
                "play_move, list_moves, apps, start_app, stop_app"
            )

    except requests.ConnectionError:
        return _err(f"Cannot connect to Reachy Mini at {host}:{port}. Is it on?")
    except requests.HTTPError as e:
        resp = e.response
        return _err(f"HTTP error: {resp.status_code} {resp.text[:200]}" if resp is not None else f"HTTP error: {e}")
    except json.JSONDecodeError as e:
        return _err(f"Invalid JSON input: {e}")
    except Exception as e:
        logger.error(f"use_reachy_mini error: {e}", exc_info=True)
        return _err(f"Error: {e}")
