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

Lazy Loading:
    Heavy imports (Robot, tools, Gr00tPolicy) are deferred until first access.
    Heavy imports are deferred so ``import strands_robots`` stays fast when lerobot/torch
    are installed but not yet needed.

    Light-weight symbols (Policy, MockPolicy, create_policy) are available
    immediately since they don't pull in torch/lerobot.
"""

import importlib as _importlib
import warnings as _warnings
from typing import Any

# ------------------------------------------------------------------
# Light-weight imports — no torch / lerobot dependency
# ------------------------------------------------------------------
from strands_robots.policies import MockPolicy, Policy, create_policy  # noqa: F401

# ------------------------------------------------------------------
# Lazy-loaded heavy symbols
# ------------------------------------------------------------------
# Maps public name -> (module_path, attribute_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Robot": ("strands_robots.robot", "Robot"),
    "Gr00tPolicy": ("strands_robots.policies.groot", "Gr00tPolicy"),
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
    "gr00t_inference",
    "lerobot_camera",
    "lerobot_teleoperate",
    "lerobot_calibrate",
    "serial_tool",
    "pose_tool",
]


def __getattr__(name: str) -> Any:  # noqa: N807
    """Lazy-load heavy modules on first attribute access.

    This avoids importing torch, lerobot, numpy, pyserial, etc. at
    ``import strands_robots`` time.  The first access to e.g.
    ``strands_robots.Robot`` triggers the real import.
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
