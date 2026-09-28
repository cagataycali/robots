"""Deprecated location of :mod:`strands_robots.drivers.reachy_transport`, kept for one minor.

The Reachy Mini daemon link is bus code the native driver owns, so it lives
beside that driver. Import it from :mod:`strands_robots.drivers.reachy_transport`;
this module is removed in 0.7 with the rest of Device Connect.
"""

from __future__ import annotations

import sys
import warnings

from strands_robots.drivers import reachy_transport as _moved

warnings.warn(
    "strands_robots.device_connect.reachy_transport moved to strands_robots.drivers.reachy_transport "
    "and is removed in 0.7",
    DeprecationWarning,
    stacklevel=2,
)

# One module object under both names, so a patch applied through the old
# spelling reaches the functions the driver calls.
sys.modules[__name__] = _moved
