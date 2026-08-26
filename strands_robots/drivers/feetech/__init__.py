"""Feetech STS/SCS-series native driver package.

The public surface starts with the wire codec (:mod:`.protocol`); the bus and
driver skeleton land as separate PRs stacked on this one (see :issue:`360`
scope 1). Nothing imported from this module opens a serial port.
"""

from __future__ import annotations

from strands_robots.drivers.feetech.protocol import (
    BROADCAST_ID,
    HEADER,
    MAX_UNICAST_ID,
    Instruction,
    build_packet,
    parse_status_packet,
    ping_packet,
    read_packet,
    sync_write_packet,
    write_packet,
)

__all__ = [
    "BROADCAST_ID",
    "HEADER",
    "Instruction",
    "MAX_UNICAST_ID",
    "build_packet",
    "parse_status_packet",
    "ping_packet",
    "read_packet",
    "sync_write_packet",
    "write_packet",
]
