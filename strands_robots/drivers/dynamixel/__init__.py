"""Native Dynamixel Protocol 2.0 driver for Koch / ViperX / WidowX / Aloha.

The scope of :issue:`359` names four things: a Protocol 2.0 codec, a bus that
opens the U2D2 and reads/writes it, a driver class satisfying
:class:`~strands_robots.drivers.base.HardwareDriver`, and a set of agent tools
(``koch_tools()``, ``aloha_tools()`` ...). Only the first and third land here.
The bus and the tools each want a hardware-verified bring-up and each want a
public API decision - see :issue:`359`'s triage note for why 3-4 unreviewed
public-API decisions in one PR is a mistake to repeat.

What this package exposes:

* :mod:`strands_robots.drivers.dynamixel.protocol` - the Protocol 2.0 wire
  format, pure functions, no I/O. Verifiable byte-for-byte against
  ``dynamixel_sdk`` as an independent oracle where installed.
* :mod:`strands_robots.drivers.dynamixel.driver` - :class:`DynamixelDriver`
  satisfying :class:`HardwareDriver`. Writes deliberately do not land yet;
  ``send_action`` returns a named ``"not wired yet (the Protocol-2.0
  serial bus)"``
  envelope, in the same shape :class:`G1Driver` uses for its own deferred
  motion path, so a caller writes the same error-checking code either way.

The driver's registered for every Dynamixel robot the package registry knows
about - koch, aloha, vx300s, wx250s, trossen_wxai, dynamixel_2r - so
``Robot("koch", mode="real", driver="strands")`` picks it up. Every one of
those uses Protocol 2.0 with a mix of XL330 / XM430 / XM540 motors, and
register 0 (``MODEL_NUMBER``) is what discriminates them on the wire. Decoding
that register is codec-level, so :func:`decode_model_number` lives here;
turning the number it returns into a model name is hardware metadata that
needs a live servo to check itself against, and lands with the bus
(:issue:`359` scope 1).
"""

from strands_robots.drivers.dynamixel.driver import DynamixelDriver
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
    "DynamixelDriver",
    "Instruction",
    "build_packet",
    "checksum",
    "decode_model_number",
    "parse_status_packet",
    "sync_write_packet",
]
