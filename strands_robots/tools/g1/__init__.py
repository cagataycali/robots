"""G1 hardware layer - vendored CycloneDDS engine for :class:`~strands_robots.drivers.g1.G1Driver`.

The Unitree G1 speaks raw Unitree IDL over CycloneDDS: ``rt/lowstate`` for IMU
and joints, ``rt/lf/bmsstate`` for battery, ``rt/utlidar/cloud_livox_mid360``
and ``rt/utlidar/lidar_state`` for the Livox Mid-360, ``rt/lowcmd`` and
``rt/armsdk`` for motion. Neither ROS 2 nor the lerobot serial bus can reach
those topics, so the driver owns its own subscriber layer.

The pieces in this package are the DDS layer only. The agent ``@tool``s that
sit *on top* of the same engine (``g1_arm``, ``g1_locomotion``, ``g1_speak``,
...) are a separate change (issue #358): they share this module's
:data:`_DDS_INIT_LOCK` and :func:`ensure_dds` singleton, so the driver and the
tools never subscribe the Livox cloud twice.

``unitree_sdk2py`` is lazy-imported: ``from strands_robots.tools.g1 import ...``
never imports it, so a machine without the SDK - every headless CI runner, and
Thor before an office bring-up - can build the driver, list it in the registry
and run every test with a mocked bus. The SDK only loads when the driver is
:meth:`~strands_robots.drivers.g1.G1Driver.connect_eagerly`-ed against a real
robot.
"""

from strands_robots.tools.g1._g1_common import (
    _DDS_INIT_LOCK,
    ERR_CODES,
    HANDSHAKE_FSMS,
    WALK_FSMS,
    decode_code,
    ensure_dds,
    reset_dds_state,
)

__all__ = [
    "ERR_CODES",
    "HANDSHAKE_FSMS",
    "WALK_FSMS",
    "_DDS_INIT_LOCK",
    "decode_code",
    "ensure_dds",
    "reset_dds_state",
]
