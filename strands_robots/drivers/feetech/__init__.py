"""Feetech STS/SCS-series native driver package.

The public surface starts with the wire codec (:mod:`.protocol`); the bus and
driver skeleton land as separate PRs stacked on this one (see :issue:`360`
scope 1). Nothing imported from this module opens a serial port.

:class:`~strands_robots.drivers.feetech.protocol.ProtocolError` is exported
alongside the codec because every parser docstring names it as the class a
caller catches to separate a wire fault from a caller bug; a handler the
package will not hand out is a contract the caller cannot write.
"""

from __future__ import annotations

from strands_robots.drivers.feetech.protocol import (
    BROADCAST_ID,
    HEADER,
    MAX_UNICAST_ID,
    Instruction,
    ProtocolError,
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
    "ProtocolError",
    "build_packet",
    "parse_status_packet",
    "ping_packet",
    "read_packet",
    "sync_write_packet",
    "write_packet",
]
