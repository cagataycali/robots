"""Reachy Mini hardware layer - shared pieces for the native driver.

The Reachy Mini speaks two protocols at once, and which one carries real-time
data depends on the hardware variant the daemon reports:

* REST on ``:8000`` (``/api/daemon/status``, ``/api/move/...``) for
  reachability, variant detection, recorded moves and the motion stop.
* A real-time link for joints and IMU - a WebSocket straight to the daemon on a
  **Lite** (no onboard computer), or Zenoh on a **Wireless** (onboard CM4).

Both live in :mod:`strands_robots.device_connect.reachy_transport`, which the
Device Connect driver already ships and which
:class:`~strands_robots.drivers.reachy.ReachyDriver` reuses rather than
re-implements. This package holds only what the *two* Reachy consumers must
agree on and neither owns: the motion envelope.

The agent ``@tool``s that will sit on the same daemon (``reachy_look``,
``reachy_antennas``, ``reachy_express``, ...) are a separate change, and they
import :func:`~strands_robots.tools.reachy._reachy_common.envelope_error` from
here so a limit is defined once for the robot rather than once per caller.

Nothing here imports a transport, a daemon client or the driver, so this package
is importable and fully testable on a machine with no Reachy attached.
"""

from strands_robots.tools.reachy._reachy_common import (
    HEAD_BODY_YAW_DELTA_LIMIT_DEG,
    MOTION_ENVELOPE_DEG,
    envelope_error,
)

__all__ = [
    "HEAD_BODY_YAW_DELTA_LIMIT_DEG",
    "MOTION_ENVELOPE_DEG",
    "envelope_error",
]
