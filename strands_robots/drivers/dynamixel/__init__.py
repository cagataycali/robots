"""Native Dynamixel Protocol 2.0 codec for Koch / ViperX / WidowX / Aloha.

:mod:`strands_robots.drivers.dynamixel.protocol` is the Protocol 2.0 wire
format: pure functions, no I/O, verifiable byte-for-byte against
``dynamixel_sdk`` where installed. Every Dynamixel robot in the registry
(koch, aloha, vx300s, wx250s, trossen_wxai, dynamixel_2r) speaks it with a mix
of XL330 / XM430 / XM540 motors, and register 0 (``MODEL_NUMBER``) is what
discriminates them on the wire, so :func:`decode_model_number` lives here.

There is no native driver yet: nothing opens the serial port, so
``Robot("koch", mode="real", driver="strands")`` is refused by name. Koch moves
through lerobot today (``driver="lerobot"`` with ``pip install
'lerobot[dynamixel]'``); the driver registers here once a bus writes goal
positions.
"""

from strands_robots.drivers.dynamixel.protocol import (
    CONTROL_TABLE,
    Instruction,
    build_packet,
    checksum,
    decode_model_number,
    parse_status_packet,
    sync_write_packet,
)

__all__ = [
    "CONTROL_TABLE",
    "Instruction",
    "build_packet",
    "checksum",
    "decode_model_number",
    "parse_status_packet",
    "sync_write_packet",
]
