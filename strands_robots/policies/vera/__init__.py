"""VERA policy provider — MIT's Video-to-Embodied Robot Action policy.

VERA (https://github.com/sizhe-li/VERA · https://vera.csail.mit.edu/) is a
**two-stage**, closed-loop video-to-action policy. It leaves a video
generative model **as-is** as an action-free world model that "dreams" the
future, and trains an embodiment-specific **inverse-dynamics model** built on
the robot's Jacobian to translate that dream into actions.

This module wraps the VERA WebSocket policy server as a strands-robots
:class:`~strands_robots.policies.base.Policy`. The transport is a
self-contained msgpack+NumPy WebSocket client (no ``vera`` / ``openpi-client``
Python dependency on the client side — the heavy GPU stack lives on the
server).

Quickstart::

    # 1. Start the policy server from a VERA checkout (holds the GPU):
    #    pip install -e ".[idm,video]"
    #    # PushT (smallest, loads in seconds)
    #    python -m vera.server.start_vera_server --embodiment pusht --port 8820
    #    # MimicGen Panda (WAN 1.3B planner, ~3.8 GB checkpoints)
    #    export VERA_WAN_CKPT_ROOT=/path/to/Wan2.1-T2V-1.3B
    #    export VERA_MIMICGEN_CKPT_DIR=./vera-ckpts/mimicgen-wan-1.3b
    #    python -m vera.server.start_vera_server --embodiment mimicgen --port 8800 \\
    #        --algo-config $VERA_MIMICGEN_CKPT_DIR/algo_config.yaml \\
    #        --text "A robot arm stacks one block on top of another block"
    #
    # 2. Client (this module) — just needs ``websockets`` + ``msgpack``:
    from strands_robots.policies import create_policy

    policy = create_policy("vera", embodiment="pusht", port=8820)
    chunk  = policy.get_actions_sync(observation, "push the T to the goal")

Available embodiments: ``pusht``, ``mimicgen``, ``droid``, ``allegro`` (see
``embodiments.py``).
"""

from .client import VeraWebsocketClient
from .embodiments import (
    ALLEGRO,
    DROID,
    MIMICGEN,
    PUSHT,
    ROBOT_ACTION_MAPPINGS,
    VeraEmbodiment,
    get_embodiment,
    get_robot_action_mapping,
    list_embodiments,
    list_robot_action_mappings,
)
from .policy import VeraPolicy

__all__ = [
    "VeraPolicy",
    "VeraWebsocketClient",
    "VeraEmbodiment",
    "PUSHT",
    "MIMICGEN",
    "DROID",
    "ALLEGRO",
    "ROBOT_ACTION_MAPPINGS",
    "get_embodiment",
    "get_robot_action_mapping",
    "list_embodiments",
    "list_robot_action_mappings",
]
