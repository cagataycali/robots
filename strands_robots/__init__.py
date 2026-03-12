"""Strands Robots — Build robot agents in a few lines of code.

Quick Start::

    from strands_robots import Robot
    sim = Robot("so100")

    from strands import Agent
    agent = Agent(tools=[sim])
    agent("Pick up the red cube")

Core API (always available)::

    Robot           — Create a robot (auto-detects sim vs real)
    create_policy   — Create a policy by name, HF ID, or URL
    create_trainer  — Create a trainer (lerobot, groot, dreamgen, cosmos)
    list_robots     — List available robots
    list_providers  — List available policy providers
    register_policy — Register a custom policy provider

Tools (for Strands Agent)::

    from strands_robots import gr00t_inference, lerobot_camera, pose_tool
    agent = Agent(tools=[Robot("so100"), gr00t_inference, lerobot_camera])

Advanced — import from submodules::

    from strands_robots.policies import Policy           # ABC
    from strands_robots.mujoco import MujocoBackend      # Direct backend access
    from strands_robots.dreamgen import DreamGenPipeline  # DreamGen pipeline
    from strands_robots.training import Trainer           # Training internals
    from strands_robots.telemetry import TelemetryStream  # Observability

See Also:
    - https://github.com/strands-labs/robots
    - https://strands-labs.github.io/robots/
"""

import logging

logger = logging.getLogger(__name__)

__all__: list[str] = []

# ─────────────────────────────────────────────────────────────────────
# Tier 1: Core — the 3 things you need to remember
#
#   Robot("so100")                    → sim or real robot
#   create_policy("mock")            → policy instance
#   create_trainer("lerobot", ...)   → trainer instance
#
# Mirrors Strands SDK:  Agent, tool, models
# ─────────────────────────────────────────────────────────────────────
try:
    from strands_robots.factory import Robot, list_robots
    from strands_robots.policies import (
        Policy,
        MockPolicy,
        create_policy,
        list_providers,
        register_policy,
    )
    from strands_robots.policy_resolver import resolve_policy

    __all__.extend([
        "Robot",
        "create_policy",
        "list_robots",
        "list_providers",
        "register_policy",
        "resolve_policy",
        "Policy",
        "MockPolicy",
    ])
except ImportError as e:
    logger.debug("Could not import core components: %s", e)

# Training factory (same pattern as create_policy)
try:
    from strands_robots.training import create_trainer, evaluate

    __all__.extend(["create_trainer", "evaluate"])
except (ImportError, AttributeError, OSError):
    pass

# ─────────────────────────────────────────────────────────────────────
# Tier 2: Tools — pick what your agent needs
#
#   agent = Agent(tools=[Robot("so100"), gr00t_inference, lerobot_camera])
#
# Each tool is imported individually so one failure doesn't cascade.
# ─────────────────────────────────────────────────────────────────────
_TOOL_IMPORTS = {
    "gr00t_inference": ("strands_robots.tools.gr00t_inference", "gr00t_inference"),
    "lerobot_camera": ("strands_robots.tools.lerobot_camera", "lerobot_camera"),
    "pose_tool": ("strands_robots.tools.pose_tool", "pose_tool"),
    "serial_tool": ("strands_robots.tools.serial_tool", "serial_tool"),
    "teleoperator": ("strands_robots.tools.teleoperator", "teleoperator"),
    "lerobot_dataset": ("strands_robots.tools.lerobot_dataset", "lerobot_dataset"),
    "newton_sim": ("strands_robots.tools.newton_sim", "newton_sim"),
    "stream": ("strands_robots.tools.stream", "stream"),
    "stereo_depth": ("strands_robots.tools.stereo_depth", "stereo_depth"),
    "robot_mesh": ("strands_robots.tools.robot_mesh", "robot_mesh"),
    "reachy_mini_tool": ("strands_robots.tools.reachy_mini_tool", "reachy_mini"),
    "download_assets": ("strands_robots.assets.download", "download_assets"),
}

for _export_name, (_module_path, _attr_name) in _TOOL_IMPORTS.items():
    try:
        import importlib as _importlib

        _mod = _importlib.import_module(_module_path)
        _obj = getattr(_mod, _attr_name)
        globals()[_export_name] = _obj
        __all__.append(_export_name)
    except ImportError:
        pass

# Clean up loop variables from module namespace
for _name in (
    "_export_name",
    "_module_path",
    "_attr_name",
    "_mod",
    "_obj",
    "_importlib",
):
    globals().pop(_name, None)
del _name