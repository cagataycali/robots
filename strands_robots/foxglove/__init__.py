"""Foxglove live view and MCAP export for any robot (``Robot(name, foxglove=True)``).

Importing this package does not import ``foxglove`` or ``mcap``; each function
imports them when called and names the ``[foxglove]`` extra when they are absent.
"""

from strands_robots.foxglove.bridge import FoxgloveBridge
from strands_robots.foxglove.export import export_episode, mcap_info
from strands_robots.foxglove.options import FOXGLOVE_ENV, FoxgloveOptions, resolve_foxglove_options
from strands_robots.foxglove.telemetry import TelemetryFanout

__all__ = [
    "FOXGLOVE_ENV",
    "FoxgloveBridge",
    "FoxgloveOptions",
    "TelemetryFanout",
    "export_episode",
    "mcap_info",
    "resolve_foxglove_options",
]
