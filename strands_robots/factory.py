"""Unified Robot Factory — convenience layer over Simulation and HardwareRobot.

Provides:
    - ``Robot("so100")`` → auto-detects sim/real, returns the right backend
    - ``list_robots()``  → what's available

Examples::

    # Auto-detect (sim if no hardware found)
    sim = Robot("so100")

    # Explicit sim
    sim = Robot("so100", mode="sim")

    # With custom URDF/MJCF path
    sim = Robot("my_arm", mode="sim", urdf_path="/path/to/robot.xml")

    # Real hardware
    hw = Robot("so100", mode="real", cameras={...})

Future (not yet implemented)::

    sim = Robot("unitree_go2", backend="isaac", num_envs=4096)
    sim = Robot("so100", backend="newton", num_envs=4096)
"""

import logging
import os
from typing import Any

from strands_robots.registry import (
    get_hardware_type,
    has_hardware,
    resolve_name,
)
from strands_robots.registry import list_robots as _registry_list_robots

logger = logging.getLogger(__name__)


def _auto_detect_mode(canonical: str) -> str:
    """Auto-detect sim vs real mode.

    Priority:
        1. ``STRANDS_ROBOT_MODE`` env var (explicit override)
        2. Robot-specific USB detection (Feetech/Dynamixel servo controllers)
        3. Default to sim (safest — never accidentally send commands to hardware)
    """
    env_mode = os.getenv("STRANDS_ROBOT_MODE", "").lower()
    if env_mode in ("sim", "real"):
        return env_mode

    # Only probe USB if the robot actually has hardware support
    if has_hardware(canonical):
        try:
            import serial.tools.list_ports

            ports = list(serial.tools.list_ports.comports())
            servo_keywords = ["feetech", "dynamixel", "sts3215", "xl430", "xl330"]
            exclude = ["bluetooth", "internal", "debug", "apple", "modem"]
            robot_ports = [
                p
                for p in ports
                if any(kw in (p.description + getattr(p, "manufacturer", "")).lower() for kw in servo_keywords)
                and not any(s in p.description.lower() for s in exclude)
            ]
            if robot_ports:
                logger.info(
                    "Auto-detected robot hardware: %s",
                    [p.device for p in robot_ports],
                )
                return "real"
        except (ImportError, Exception):
            pass

    return "sim"


def Robot(
    name: str,
    mode: str = "auto",
    backend: str = "mujoco",
    urdf_path: str = None,
    cameras: dict[str, dict[str, Any]] | None = None,
    position: list[float] | None = None,
    **kwargs,
):
    """Create a robot — returns a Simulation or HardwareRobot instance.

    This is a convenience factory, NOT a wrapper class.  You get the real
    backend instance back — with full access to all its methods.

    Args:
        name: Robot name ("so100", "aloha", "unitree_g1", "panda", ...)
              Accepts any alias defined in ``registry/robots.json``.
        mode: "auto" (detect hardware), "sim", or "real".
        backend: Simulation backend — currently only "mujoco" (CPU).
                 Future: "isaac" (GPU), "newton" (GPU).
        urdf_path: Explicit path to URDF/MJCF file. If not provided,
                   resolved via the model registry (asset manager or
                   STRANDS_URDF_DIR search paths).
        cameras: Camera config for real hardware. Example::

            {"wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30}}

        position: Robot position in sim world [x, y, z].
        **kwargs: Forwarded to the underlying backend constructor.

    Returns:
        ``Simulation`` (MuJoCo sim) or ``Robot`` (real hardware).

    Raises:
        RuntimeError: If the sim world or robot fails to initialize.
        NotImplementedError: If an unimplemented backend is requested.

    Examples::

        # MuJoCo sim (auto — no hardware detected)
        sim = Robot("so100")
        sim = Robot("so100", mode="sim")

        # Explicit MJCF model path
        sim = Robot("my_arm", mode="sim", urdf_path="path/to/robot.xml")

        # Real hardware
        hw = Robot("so100", mode="real", cameras={...})

        # The 5-line promise
        from strands_robots import Robot
        from strands import Agent
        robot = Robot("so100")
        agent = Agent(tools=[robot])
        agent("Pick up the red cube")
    """
    canonical = resolve_name(name)

    if mode == "auto":
        mode = _auto_detect_mode(canonical)

    # ── Simulation ──
    if mode == "sim":
        if backend != "mujoco":
            raise NotImplementedError(
                f"Backend {backend!r} is not yet implemented. "
                f"Currently supported: 'mujoco'. "
                f"Isaac and Newton backends are on the roadmap."
            )

        from strands_robots.simulation import Simulation

        sim = Simulation(
            tool_name=f"{canonical}_sim",
            **kwargs,
        )
        sim._dispatch_action("create_world", {})

        # Build add_robot params — pass urdf_path if the user provided one
        add_robot_params: dict[str, Any] = {
            "robot_name": canonical,
            "data_config": canonical,
            "position": position or [0.0, 0.0, 0.0],
        }
        if urdf_path:
            add_robot_params["urdf_path"] = urdf_path

        result = sim._dispatch_action("add_robot", add_robot_params)
        if result.get("status") == "error":
            # Extract human-readable message from content
            content = result.get("content", [])
            msg = content[0].get("text", str(result)) if content else str(result)
            raise RuntimeError(f"Failed to create sim robot '{canonical}': {msg}")
        return sim

    # ── Real hardware ──
    else:
        from strands_robots.robot import Robot as HardwareRobot

        real_type = get_hardware_type(canonical) or canonical
        return HardwareRobot(
            tool_name=canonical,
            robot=real_type,
            cameras=cameras,
            **kwargs,
        )


def list_robots(mode: str = "all") -> list[dict[str, Any]]:
    """List available robots.

    Args:
        mode: "all", "sim", "real", or "both" (has both sim and real).

    Returns:
        List of dicts with name, description, has_sim, has_real.
    """
    return _registry_list_robots(mode)


__all__ = ["Robot", "list_robots"]
