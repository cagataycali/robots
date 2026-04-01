#!/usr/bin/env python3
"""
Strands Robotics - Universal Robot Control with Policy Abstraction

A unified Python interface for controlling diverse robot hardware through
any VLA provider with clean policy abstraction architecture.

Key features:
- Policy abstraction for any VLA provider (GR00T, ACT, SmolVLA, etc.)
- Universal robot support through LeRobot integration
- Clean separation between robot control and policy inference
- Direct policy injection for maximum flexibility
- Multi-camera support with rich configuration options
- MuJoCo simulation backend (no GPU required)

Lazy Loading:
    Heavy imports (Robot, tools, Gr00tPolicy, Simulation) are deferred until
    first access. Heavy imports are deferred so ``import strands_robots`` stays
    fast when lerobot/torch/mujoco are installed but not yet needed.

    Light-weight symbols (Policy, MockPolicy, create_policy) are available
    immediately since they don't pull in torch/lerobot.
"""

import importlib as _importlib
import warnings as _warnings
from typing import Any

# ------------------------------------------------------------------
# Light-weight imports — no torch / lerobot / mujoco dependency
# ------------------------------------------------------------------
from strands_robots.policies import MockPolicy, Policy, create_policy  # noqa: F401

# ------------------------------------------------------------------
# Lazy-loaded heavy symbols
# ------------------------------------------------------------------
# Maps public name -> (module_path, attribute_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Hardware robot
    "Robot": ("strands_robots.factory", "Robot"),
    "list_robots": ("strands_robots.factory", "list_robots"),
    # Policies
    "Gr00tPolicy": ("strands_robots.policies.groot", "Gr00tPolicy"),
    # Simulation (MuJoCo)
    "Simulation": ("strands_robots.simulation", "Simulation"),
    "create_simulation": ("strands_robots.simulation.factory", "create_simulation"),
    "list_backends": ("strands_robots.simulation.factory", "list_backends"),
    "register_backend": ("strands_robots.simulation.factory", "register_backend"),
    "SimWorld": ("strands_robots.simulation", "SimWorld"),
    "SimRobot": ("strands_robots.simulation", "SimRobot"),
    "SimObject": ("strands_robots.simulation", "SimObject"),
    "SimCamera": ("strands_robots.simulation", "SimCamera"),
    # Tools
    "gr00t_inference": ("strands_robots.tools.gr00t_inference", "gr00t_inference"),
    "lerobot_calibrate": ("strands_robots.tools.lerobot_calibrate", "lerobot_calibrate"),
    "lerobot_camera": ("strands_robots.tools.lerobot_camera", "lerobot_camera"),
    "lerobot_teleoperate": ("strands_robots.tools.lerobot_teleoperate", "lerobot_teleoperate"),
    "pose_tool": ("strands_robots.tools.pose_tool", "pose_tool"),
    "serial_tool": ("strands_robots.tools.serial_tool", "serial_tool"),
}

__all__ = [
    # Always available
    "Policy",
    "MockPolicy",
    "create_policy",
    # Lazy-loaded
    "Robot",
    "Gr00tPolicy",
    "Simulation",
    "SimWorld",
    "SimRobot",
    "SimObject",
    "SimCamera",
    "gr00t_inference",
    "lerobot_camera",
    "lerobot_teleoperate",
    "lerobot_calibrate",
    "serial_tool",
    "pose_tool",
]


# Auto-configure MuJoCo GL backend for headless environments BEFORE any
# module imports mujoco at the top level.  MuJoCo locks the OpenGL backend
# at import time, so MUJOCO_GL must be set first.
try:
    from strands_robots.simulation.mujoco.backend import _configure_gl_backend

    _configure_gl_backend()
except (ImportError, AttributeError, OSError):
    pass


def __getattr__(name: str) -> Any:  # noqa: N807
    """Lazy-load heavy modules on first attribute access.

    This avoids importing torch, lerobot, numpy, mujoco, pyserial, etc. at
    ``import strands_robots`` time.  The first access to e.g.
    ``strands_robots.Robot`` or ``strands_robots.Simulation`` triggers the
    real import.
    """
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            module = _importlib.import_module(module_path)
            value = getattr(module, attr_name)
            # Cache in module dict so __getattr__ is not called again
            globals()[name] = value
            return value
        except ImportError as exc:
            _warnings.warn(
                f"{name} not available (missing dependencies): {exc}",
                stacklevel=2,
            )
            raise AttributeError(name) from exc
    raise AttributeError(f"module 'strands_robots' has no attribute {name!r}")
