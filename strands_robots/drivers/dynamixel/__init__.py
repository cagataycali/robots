"""Native Dynamixel Protocol 2.0 driver, bus and codec.

:mod:`strands_robots.drivers.dynamixel.protocol` is the Protocol 2.0 wire
format: pure functions, no I/O, verifiable byte-for-byte against
``dynamixel_sdk`` where installed. Every Dynamixel robot in the registry
(koch, aloha, vx300s, wx250s, trossen_wxai, dynamixel_2r) speaks it with a mix
of XL330 / XM430 / XM540 motors, and register 0 (``MODEL_NUMBER``) is what
discriminates them on the wire, so :func:`decode_model_number` lives here.

:class:`~strands_robots.drivers.dynamixel.driver.DynamixelDriver` drives koch
over :class:`~strands_robots.drivers.dynamixel.bus.DynamixelBus`:
``Robot("koch", mode="real", driver="strands", port=...)``. The other five arms
are refused by name until each has a verified motor map.
"""

from strands_robots.drivers.dynamixel.bus import KOCH_MOTORS, DynamixelBus
from strands_robots.drivers.dynamixel.driver import DynamixelDriver
from strands_robots.drivers.dynamixel.protocol import (
    CONTROL_TABLE,
    Instruction,
    build_packet,
    checksum,
    decode_model_number,
    parse_status_packet,
    parse_status_stream,
    sync_read_packet,
    sync_write_packet,
    write_packet,
)

__all__ = [
    "CONTROL_TABLE",
    "KOCH_MOTORS",
    "DynamixelBus",
    "DynamixelDriver",
    "Instruction",
    "build_packet",
    "checksum",
    "decode_model_number",
    "parse_status_packet",
    "parse_status_stream",
    "sync_read_packet",
    "sync_write_packet",
    "write_packet",
]
