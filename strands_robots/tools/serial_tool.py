import time
from typing import Any

import serial
import serial.tools.list_ports
from strands import tool

from strands_robots.utils import bounded_count_error

# The Feetech STS/SMS control-table domains this tool can honor, read off the
# protocol it writes rather than chosen here.
#
# ``motor_id`` becomes the packet's address byte, so it must both fit a byte and
# name a motor; 1-254 is the range this tool documents. ``position`` is the
# Goal_Position register, and the 0-4095 scale is already the one this tool
# reports on - it echoes every write back as ``position / 4095 * 360`` degrees,
# so a value outside that scale has no meaningful degree reading. ``velocity``
# declares no travel limit, so it is bounded by the width of the field instead:
# both goal values are packed little-endian into two bytes.
_MOTOR_ID_RANGE = (1, 254)
_POSITION_RANGE = (0, 4095)
# Derived from the width of the field rather than restated as a magnitude: the
# two goal bytes below are ``value & 0xFF`` and ``(value >> 8) & 0xFF``, so this
# range is exactly the set of values that survives that packing.
_GOAL_REGISTER_BYTES = 2
_VELOCITY_RANGE = (0, 2 ** (8 * _GOAL_REGISTER_BYTES) - 1)

_PARAM_RANGES: dict[str, tuple[int, int]] = {
    "motor_id": _MOTOR_ID_RANGE,
    "position": _POSITION_RANGE,
    "velocity": _VELOCITY_RANGE,
}

# Which of those parameters each action writes into a packet. An action absent
# from this map writes none of them, so a caller is never refused here for a
# value the requested action does not read.
_FEETECH_PARAMS: dict[str, tuple[str, ...]] = {
    "feetech_position": ("motor_id", "position"),
    "feetech_velocity": ("motor_id", "velocity"),
    "feetech_ping": ("motor_id",),
}


def feetech_param_error(action: str, *, motor_id: Any, position: Any, velocity: Any) -> str | None:
    """Error text for the first Feetech parameter ``action`` writes but cannot carry.

    Each of these is packed into a fixed-width field of the servo packet, and
    that packing reduces an out-of-range value rather than refusing it: the two
    goal registers are written as ``value & 0xFF`` and ``(value >> 8) & 0xFF``,
    so ``65536`` puts ``0`` on the bus and ``-1`` puts ``65535`` there. A
    different command reaches the motor than the one the caller asked for, and
    than the one this tool reports back - which is why the range is checked here
    rather than left to the servo, and why it is checked before the port is
    opened rather than beside the write.

    Only the parameters ``action`` actually writes are checked, and a parameter
    that was not supplied at all is left to the branch that reads it, so its
    existing "required" message is unchanged.

    Args:
        action: The requested action; decides which parameters are effective.
        motor_id: Motor address, as supplied.
        position: Goal_Position value, as supplied.
        velocity: Goal_Velocity value, as supplied.

    Returns:
        An error message naming the action, the parameter and the accepted
        range, or ``None`` when every parameter this action writes is usable.
    """
    supplied: dict[str, Any] = {"motor_id": motor_id, "position": position, "velocity": velocity}
    for param in _FEETECH_PARAMS.get(action, ()):
        value = supplied[param]
        if value is None:
            continue
        minimum, maximum = _PARAM_RANGES[param]
        error = bounded_count_error(value, param, action, minimum=minimum, maximum=maximum)
        if error:
            return error
    return None


@tool
def serial_tool(
    action: str,
    port: str | None = None,
    baudrate: int = 9600,
    timeout: float = 1.0,
    data: str | None = None,
    hex_data: str | None = None,
    motor_id: int | None = None,
    position: int | None = None,
    velocity: int | None = None,
    read_bytes: int = 1024,
) -> dict[str, Any]:
    """Advanced serial communication tool for robot control and device communication.

    Actions:
        - "list_ports": Discover available serial ports
        - "send": Send data to serial port
        - "read": Read data from serial port
        - "send_read": Send data and read response
        - "feetech_position": Control Feetech servo position
        - "feetech_velocity": Control Feetech servo velocity
        - "feetech_ping": Ping Feetech servo motor
        - "monitor": Monitor serial port (continuous read)

    Args:
        action: Action to perform
        port: Serial port path (e.g., "/dev/ttyACM0", "COM3")
        baudrate: Communication speed (default: 9600)
        timeout: Read timeout in seconds
        data: String data to send
        hex_data: Hex string data to send (e.g., "FF FF 01 04 03 00 64 92")
        motor_id: Motor ID for Feetech commands (1-254)
        position: Target position for Feetech motors (0-4095)
        velocity: Target velocity for Feetech motors (0-65535, the width of the
            two-byte register it is written into)
        read_bytes: Number of bytes to read

    The Feetech commands pack ``motor_id`` into the packet's address byte and
    ``position`` / ``velocity`` into a two-byte register field, so each is
    refused unless it fits that field. Out-of-range values are otherwise reduced
    modulo the field width and a different command reaches the bus than the one
    reported back.

    Returns:
        Dict containing status and response content
    """

    def list_serial_ports() -> list[dict]:
        """List all available serial ports."""
        ports = []
        for port_info in serial.tools.list_ports.comports():
            ports.append(
                {
                    "device": port_info.device,
                    "name": port_info.name,
                    "description": port_info.description,
                    "manufacturer": port_info.manufacturer,
                    "vid": port_info.vid,
                    "pid": port_info.pid,
                    "serial_number": port_info.serial_number,
                }
            )
        return ports

    def build_feetech_packet(motor_id: int, instruction: int, params: list[int]) -> bytes:
        """Build Feetech servo protocol packet."""
        packet = [0xFF, 0xFF, motor_id, len(params) + 2, instruction] + params
        checksum = ~sum(packet[2:]) & 0xFF
        packet.append(checksum)
        return bytes(packet)

    try:
        if action == "list_ports":
            ports = list_serial_ports()
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"Found {len(ports)} serial ports:\n"
                        + "\n".join([f"- {p['device']} - {p['description']}" for p in ports])
                    },
                    {"json": {"ports": ports}},
                ],
            }

        if not port:
            return {"status": "error", "content": [{"text": "Port parameter required for this action"}]}

        # Refuse an unusable Feetech parameter before the bus is opened, so a
        # value that cannot be carried never energizes a motor.
        param_error = feetech_param_error(action, motor_id=motor_id, position=position, velocity=velocity)
        if param_error:
            return {"status": "error", "content": [{"text": param_error}]}

        # Open serial connection
        ser = serial.Serial(port, baudrate, timeout=timeout)

        if action == "send":
            if hex_data:
                # Parse hex string (e.g., "FF FF 01 04" -> [0xFF, 0xFF, 0x01, 0x04])
                hex_bytes = bytes.fromhex(hex_data.replace(" ", ""))
                ser.write(hex_bytes)
                response_text = f"Sent hex data: {hex_data}"
            elif data:
                ser.write(data.encode())
                response_text = f"Sent string data: {data}"
            else:
                ser.close()
                return {"status": "error", "content": [{"text": "No data or hex_data provided"}]}

            ser.close()
            return {"status": "success", "content": [{"text": response_text}]}

        elif action == "read":
            read_data = ser.read(read_bytes)
            ser.close()

            # Format response as both hex and ASCII
            hex_str = " ".join([f"{b:02X}" for b in read_data])
            ascii_str = "".join([chr(b) if 32 <= b <= 126 else f"\\x{b:02x}" for b in read_data])

            return {
                "status": "success",
                "content": [
                    {"text": f"Read {len(read_data)} bytes:\nHex: {hex_str}\nASCII: {ascii_str}"},
                    {"json": {"raw_data": read_data.hex(), "length": len(read_data)}},
                ],
            }

        elif action == "send_read":
            # Send data first
            if hex_data:
                hex_bytes = bytes.fromhex(hex_data.replace(" ", ""))
                ser.write(hex_bytes)
                sent_text = f"Sent hex: {hex_data}"
            elif data:
                ser.write(data.encode())
                sent_text = f"Sent string: {data}"
            else:
                ser.close()
                return {"status": "error", "content": [{"text": "No data to send"}]}

            # Small delay then read response
            time.sleep(0.1)
            read_data = ser.read(read_bytes)
            ser.close()

            hex_str = " ".join([f"{b:02X}" for b in read_data])
            ascii_str = "".join([chr(b) if 32 <= b <= 126 else f"\\x{b:02x}" for b in read_data])

            return {
                "status": "success",
                "content": [{"text": f"{sent_text}\nRead {len(read_data)} bytes:\nHex: {hex_str}\nASCII: {ascii_str}"}],
            }

        elif action == "feetech_position":
            if motor_id is None or position is None:
                ser.close()
                return {"status": "error", "content": [{"text": "motor_id and position required"}]}

            # Feetech position command: INST_WRITE (0x03), Goal_Position address (0x2A)
            params = [0x2A, position & 0xFF, (position >> 8) & 0xFF]
            packet = build_feetech_packet(motor_id, 0x03, params)
            ser.write(packet)
            ser.close()

            return {
                "status": "success",
                "content": [
                    {"text": f"Feetech Motor {motor_id} -> Position {position} ({position / 4095 * 360:.1f} deg)"}
                ],
            }

        elif action == "feetech_velocity":
            if motor_id is None or velocity is None:
                ser.close()
                return {"status": "error", "content": [{"text": "motor_id and velocity required"}]}

            # Feetech velocity command: Goal_Velocity address (0x2E)
            params = [0x2E, velocity & 0xFF, (velocity >> 8) & 0xFF]
            packet = build_feetech_packet(motor_id, 0x03, params)
            ser.write(packet)
            ser.close()

            return {"status": "success", "content": [{"text": f"Feetech Motor {motor_id} -> Velocity {velocity}"}]}

        elif action == "feetech_ping":
            if motor_id is None:
                ser.close()
                return {"status": "error", "content": [{"text": "motor_id required"}]}

            # Feetech ping command
            packet = build_feetech_packet(motor_id, 0x01, [])  # INST_PING
            ser.write(packet)

            time.sleep(0.1)
            response = ser.read(10)
            ser.close()

            if len(response) >= 6:
                return {
                    "status": "success",
                    "content": [{"text": f"Feetech Motor {motor_id} responded: {response.hex().upper()}"}],
                }
            else:
                return {"status": "error", "content": [{"text": f"Feetech Motor {motor_id} no response"}]}

        elif action == "monitor":
            # Continuous monitoring (limited time for safety)
            monitor_data = []
            start_time = time.time()

            while time.time() - start_time < 5.0:  # 5 second limit
                if ser.in_waiting > 0:
                    chunk = ser.read(ser.in_waiting)
                    monitor_data.append(
                        {
                            "timestamp": time.time(),
                            "data": chunk.hex(),
                            "ascii": "".join([chr(b) if 32 <= b <= 126 else f"\\x{b:02x}" for b in chunk]),
                        }
                    )
                time.sleep(0.1)

            ser.close()

            return {
                "status": "success",
                "content": [
                    {"text": f"Monitored {len(monitor_data)} data chunks in 5 seconds"},
                    {"json": {"monitor_data": monitor_data}},
                ],
            }

        else:
            ser.close()
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}\n"
                        "Available: list_ports, send, read, send_read,"
                        " feetech_position, feetech_velocity, feetech_ping, monitor"
                    }
                ],
            }

    except serial.SerialException as e:
        return {"status": "error", "content": [{"text": f"Serial error: {e}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}
