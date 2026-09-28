"""Deprecated location of :mod:`strands_robots._pacing`, kept for one minor.

Loop pacing is pure timing that drivers, the mesh and the rollout loops all
share, so it lives in the core layer. Import ``Ticker`` and ``sleep_penalty_s``
from :mod:`strands_robots._pacing`; this module is removed in 0.7.
"""

from __future__ import annotations

import warnings

from strands_robots._pacing import Ticker, sleep_penalty_s

__all__ = ["Ticker", "sleep_penalty_s"]

warnings.warn(
    "strands_robots.mesh.pacing moved to strands_robots._pacing and is removed in 0.7",
    DeprecationWarning,
    stacklevel=2,
)
