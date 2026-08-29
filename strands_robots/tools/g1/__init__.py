"""G1 hardware layer - vendored CycloneDDS engine for :class:`~strands_robots.drivers.g1.G1Driver`.

The Unitree G1 speaks raw Unitree IDL over CycloneDDS: ``rt/lowstate`` for IMU
and joints, ``rt/lf/bmsstate`` for battery, ``rt/utlidar/cloud_livox_mid360``
and ``rt/utlidar/lidar_state`` for the Livox Mid-360, ``rt/lowcmd`` and
``rt/armsdk`` for motion. Neither ROS 2 nor the lerobot serial bus can reach
those topics, so the driver owns its own subscriber layer.

The pieces in this package are the DDS layer only. The agent ``@tool``s that
sit *on top* of the same engine (``g1_arm``, ``g1_locomotion``, ``g1_speak``,
...) are a separate change (issue #358): they share this module's
:func:`ensure_dds` singleton and the
:data:`~strands_robots.tools.g1._g1_common._DDS_INIT_LOCK` it serialises on, so
the driver and the tools never subscribe the Livox cloud twice. That lock is
private to ``_g1_common``; reach it there rather than through this package.

``unitree_sdk2py`` is lazy-imported: ``from strands_robots.tools.g1 import ...``
never imports it, so a machine without the SDK - every headless CI runner, and
Thor before an office bring-up - can build the driver, list it in the registry
and run every test with a mocked bus. The SDK only loads when the driver is
:meth:`~strands_robots.drivers.g1.G1Driver.connect_eagerly`-ed against a real
robot.
"""

from strands_robots.tools.g1._g1_common import (
    ERR_CODES,
    HANDSHAKE_FSMS,
    WALK_FSMS,
    decode_code,
    ensure_dds,
    reset_dds_state,
)

# Verb modules under this package (``g1_joints``, ``g1_state``, ...) are
# imported directly by callers rather than re-exported here. Two reasons:
#
# * The driver at :mod:`strands_robots.drivers.g1` imports from this package
#   to reach :data:`HANDSHAKE_FSMS` and :func:`decode_code`. Verb modules that
#   themselves import from :mod:`strands_robots.drivers.g1` (to read the
#   driver's constants) would close a circle through this ``__init__``, so
#   the package's public surface is the DDS-only layer above and the verbs
#   sit as leaves.
# * The verb modules follow the SDK-load-hygiene contract only for their own
#   import; re-exporting them here would pull every one of them at package
#   load, and a caller who only wanted :func:`ensure_dds` would pay for the
#   whole tree. Callers name the verb module they want:
#   ``from strands_robots.tools.g1.g1_joints import g1_joint_reference``.

__all__ = [
    "ERR_CODES",
    "HANDSHAKE_FSMS",
    "WALK_FSMS",
    "decode_code",
    "ensure_dds",
    "reset_dds_state",
]
