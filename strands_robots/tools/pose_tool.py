#!/usr/bin/env python3
"""
Robot Pose Management Tool

This tool provides comprehensive pose management for robotic arms, including:
- Storing and retrieving named poses
- Fine-grained motor control with small incremental movements
- Safety checks and validation
- Integration with LeRobot and serial communication
- Pose interpolation and smooth transitions
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

import serial
import serial.tools.list_ports
from strands import tool

from strands_robots.tools._path_validation import validate_save_path

logger = logging.getLogger(__name__)


@dataclass
class RobotPose:
    """Represents a robot pose with metadata."""

    name: str
    positions: dict[str, float]  # motor_name -> position
    timestamp: float
    description: str | None = None
    safety_bounds: dict[str, tuple[float, float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotPose":
        """Create from dictionary."""
        return cls(**data)


class PoseManager:
    """Manages robot poses with persistence and safety."""

    def __init__(self, robot_id: str, storage_dir: Path | None = None):
        self.robot_id = robot_id
        raw_dir = str(storage_dir) if storage_dir else str(Path.cwd() / ".strands_robots" / "poses")
        self.storage_dir = Path(validate_save_path(raw_dir, label="storage_dir"))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.pose_file = self.storage_dir / f"{robot_id}_poses.json"
        self.poses: dict[str, RobotPose] = {}
        self._load_poses()

    def _load_poses(self) -> None:
        """Load poses from storage."""
        if self.pose_file.exists():
            try:
                with open(self.pose_file) as f:
                    data = json.load(f)
                    self.poses = {name: RobotPose.from_dict(pose_data) for name, pose_data in data.items()}
                logger.info(f"Loaded {len(self.poses)} poses for robot {self.robot_id}")
            except Exception as e:
                logger.error(f"Failed to load poses: {e}")
                self.poses = {}

    def _save_poses(self) -> None:
        """Save poses to storage."""
        try:
            data = {name: pose.to_dict() for name, pose in self.poses.items()}
            with open(self.pose_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.poses)} poses for robot {self.robot_id}")
        except Exception as e:
            logger.error(f"Failed to save poses: {e}")

    def store_pose(
        self,
        name: str,
        positions: dict[str, float],
        description: str | None = None,
        safety_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> RobotPose:
        """Store a new pose."""
        pose = RobotPose(
            name=name,
            positions=positions.copy(),
            timestamp=time.time(),
            description=description,
            safety_bounds=safety_bounds,
        )
        self.poses[name] = pose
        self._save_poses()
        return pose

    def get_pose(self, name: str) -> RobotPose | None:
        """Get a stored pose."""
        return self.poses.get(name)

    def list_poses(self) -> list[str]:
        """List all pose names."""
        return list(self.poses.keys())

    def delete_pose(self, name: str) -> bool:
        """Delete a pose."""
        if name in self.poses:
            del self.poses[name]
            self._save_poses()
            return True
        return False

    def validate_pose(self, pose: RobotPose) -> tuple[bool, str]:
        """Validate pose is within safety bounds."""
        if not pose.safety_bounds:
            return True, "No safety bounds defined"

        for motor, position in pose.positions.items():
            if motor in pose.safety_bounds:
                min_pos, max_pos = pose.safety_bounds[motor]
                if not (min_pos <= position <= max_pos):
                    return False, f"Motor {motor} position {position} outside bounds [{min_pos}, {max_pos}]"

        return True, "Pose is valid"


class MotorConfig(TypedDict):
    """Configuration for a single servo motor."""

    id: int
    range: tuple[int, int]
    resolution: int


#: STS3215 tick span. ``lerobot``'s conversion divides by ``resolution - 1``
#: (``model_resolution_table[...] - 1``), so the usable span is 4095.
_TICK_SPAN = 4095


def load_lerobot_calibration(calibration_id: str) -> dict[str, dict[str, int]] | None:
    """Load a lerobot follower calibration by id, or ``None`` when absent.

    lerobot 0.5+ stores SO-family FOLLOWER calibration at
    ``$HF_LEROBOT_CALIBRATION/robots/so_follower/<id>.json`` (the directory is the
    driver CLASS name, ``so_follower``, not the robot type ``so101_follower``), one
    record per motor: ``{id, drive_mode, homing_offset, range_min, range_max}``.

    Without it, ``degrees_to_position`` had to assume every joint's centre sits at
    the midpoint of a hardcoded degree range, i.e. tick 2047 for 0 degrees. Real
    calibration on this class of arm puts the centres elsewhere: measured against
    ``so_follower/so101.json`` the hardcoded mapping is off by 152 ticks
    (13.4 degrees) on shoulder_pan, 110 (9.7 degrees) on wrist_flex, and 2029
    ticks (178 degrees) on the gripper - the gripper being the dangerous one,
    since a "fully closed" command became a hard slam past the mechanical stop.

    Args:
        calibration_id: The lerobot ``--robot.id`` the arm was calibrated under
            (e.g. ``"so101"``), NOT the robot type.

    Returns:
        Mapping of motor name to its calibration record, or ``None`` when no file
        exists or it cannot be parsed (the caller then reports the uncalibrated
        state rather than silently guessing).
    """
    root = os.environ.get("HF_LEROBOT_CALIBRATION")
    base = Path(root) if root else Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
    path = base / "robots" / "so_follower" / f"{calibration_id}.json"
    if not path.is_file():
        logger.debug("no lerobot calibration at %s", path)
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        logger.warning("could not read lerobot calibration %s: %s", path, e)
        return None
    if not isinstance(data, dict) or not data:
        logger.warning("lerobot calibration %s is not a non-empty object", path)
        return None
    return data


class MotorController:
    """Low-level motor control for fine movements."""

    def __init__(
        self,
        port: str,
        baudrate: int = 1000000,
        calibration: dict[str, dict[str, int]] | None = None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn: serial.Serial | None = None
        # lerobot calibration records keyed by motor name, when available. When
        # present, degrees_to_position / position_to_degrees use the SAME formula
        # lerobot's MotorsBus uses, so a pose stored or commanded here refers to
        # the same physical configuration the policy/teleop path does. When None,
        # the hardcoded midpoint mapping is used and the caller is told (see
        # ``pose_tool``): guessing silently is what made this tool disagree with
        # every other driver in the stack by up to 13 degrees per joint.
        self.calibration = calibration or None

        # Default motor configurations for SO-101
        self.motor_configs: dict[str, MotorConfig] = {
            "shoulder_pan": {"id": 1, "range": (-180, 180), "resolution": 4095},
            "shoulder_lift": {"id": 2, "range": (-90, 90), "resolution": 4095},
            "elbow_flex": {"id": 3, "range": (-150, 150), "resolution": 4095},
            "wrist_flex": {"id": 4, "range": (-90, 90), "resolution": 4095},
            "wrist_roll": {"id": 5, "range": (-180, 180), "resolution": 4095},
            "gripper": {"id": 6, "range": (0, 100), "resolution": 4095},
        }

    def connect(self) -> tuple[bool, str]:
        """Connect to robot.

        Returns:
            Tuple[bool, str]: (success, error_message) - error_message is empty on success
        """
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1.0)
            return True, ""
        except Exception as e:
            error_msg = f"Failed to connect to {self.port}: {e}"
            logger.error(error_msg)
            return False, error_msg

    def disconnect(self) -> None:
        """Disconnect from robot."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

    def build_feetech_packet(self, motor_id: int, instruction: int, params: list[int]) -> bytes:
        """Build Feetech servo protocol packet."""
        packet = [0xFF, 0xFF, motor_id, len(params) + 2, instruction] + params
        checksum = ~sum(packet[2:]) & 0xFF
        packet.append(checksum)
        return bytes(packet)

    def degrees_to_position(self, motor_name: str, degrees: float) -> int:
        """Convert degrees to motor position."""
        if motor_name not in self.motor_configs:
            raise ValueError(f"Unknown motor: {motor_name}")

        config = self.motor_configs[motor_name]
        min_deg, max_deg = config["range"]

        # Clamp to range
        degrees = max(min_deg, min(max_deg, degrees))

        record = (self.calibration or {}).get(motor_name)
        if record is not None:
            # Same formula as lerobot MotorsBus._unnormalize, so a tick computed
            # here means the same physical pose the policy / teleop path commands.
            range_min = int(record["range_min"])
            range_max = int(record["range_max"])
            if range_max == range_min:
                raise ValueError(
                    f"Invalid calibration for motor {motor_name!r}: range_min == range_max == {range_min}."
                )
            if motor_name == "gripper":
                # MotorNormMode.RANGE_0_100: 0..100 maps across the measured span.
                bounded = min(100.0, max(0.0, degrees))
                return int((bounded / 100.0) * (range_max - range_min) + range_min)
            # MotorNormMode.DEGREES: mid-point-centred, NOT range-normalised.
            mid = (range_min + range_max) / 2
            return int(degrees * _TICK_SPAN / 360 + mid)

        # Convert to position (0-4095 for most servos)
        if motor_name == "gripper":
            # Gripper uses 0-100 percentage
            return int((degrees / 100.0) * config["resolution"])
        else:
            # Regular joints use degree range
            normalized = (degrees - min_deg) / (max_deg - min_deg)
            return int(normalized * config["resolution"])

    def position_to_degrees(self, motor_name: str, position: int) -> float:
        """Convert motor position to degrees."""
        if motor_name not in self.motor_configs:
            raise ValueError(f"Unknown motor: {motor_name}")

        config = self.motor_configs[motor_name]
        min_deg, max_deg = config["range"]

        record = (self.calibration or {}).get(motor_name)
        if record is not None:
            # Exact inverse of the calibrated branch of degrees_to_position, so a
            # read-back value round-trips instead of drifting by the offset.
            range_min = int(record["range_min"])
            range_max = int(record["range_max"])
            if range_max == range_min:
                raise ValueError(
                    f"Invalid calibration for motor {motor_name!r}: range_min == range_max == {range_min}."
                )
            if motor_name == "gripper":
                return ((position - range_min) / (range_max - range_min)) * 100.0
            mid = (range_min + range_max) / 2
            return (position - mid) * 360 / _TICK_SPAN

        if motor_name == "gripper":
            return (position / config["resolution"]) * 100.0
        else:
            normalized = position / config["resolution"]
            return min_deg + normalized * (max_deg - min_deg)

    def move_motor(self, motor_name: str, position_degrees: float) -> bool:
        """Move a single motor to position in degrees."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return False

        try:
            motor_id = self.motor_configs[motor_name]["id"]
            position = self.degrees_to_position(motor_name, position_degrees)

            # Feetech position command: INST_WRITE (0x03), Goal_Position address (0x2A)
            params = [0x2A, position & 0xFF, (position >> 8) & 0xFF]
            packet = self.build_feetech_packet(motor_id, 0x03, params)
            self.serial_conn.write(packet)
            return True
        except Exception as e:
            logger.error(f"Failed to move motor {motor_name}: {e}")
            return False

    def disable_torque(self) -> list[str]:
        """De-energize every configured motor, returning the ones that failed.

        Writes ``Torque_Enable = 0`` to each motor. The register is address 40
        (1 byte) on the Feetech STS3215 control table - the authority is
        ``lerobote.motors.feetech.tables`` (``"Torque_Enable": (40, 1)``), the same
        table that gives ``Goal_Position`` address 42 used by :meth:`move_motor`.

        Every motor is attempted even if an earlier one fails: a partial stop
        that silently gave up on the remaining joints would be worse than no
        stop, because the caller would believe the whole arm was released.

        Returns:
            Names of motors whose write failed - empty when all succeeded. A
            non-empty list means the arm is NOT fully de-energized.
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            return list(self.motor_configs)

        failed: list[str] = []
        for motor_name, config in self.motor_configs.items():
            try:
                packet = self.build_feetech_packet(config["id"], 0x03, [0x28, 0x00])
                self.serial_conn.write(packet)
            except Exception as e:  # noqa: BLE001 - try every motor, report the rest
                logger.error(f"Failed to disable torque on {motor_name}: {e}")
                failed.append(motor_name)
        return failed

    def read_motor_position(self, motor_name: str) -> float | None:
        """Read current motor position in degrees."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return None

        try:
            motor_id = self.motor_configs[motor_name]["id"]

            # Feetech read command: INST_READ (0x02), Present_Position address (0x38), 2 bytes
            params = [0x38, 0x02]
            packet = self.build_feetech_packet(motor_id, 0x02, params)
            self.serial_conn.write(packet)

            time.sleep(0.01)  # Small delay for response
            response = self.serial_conn.read(10)

            if len(response) >= 7:
                position = response[5] | (response[6] << 8)
                return self.position_to_degrees(motor_name, position)
        except Exception as e:
            logger.error(f"Failed to read motor {motor_name}: {e}")

        return None

    def read_all_positions(self) -> dict[str, float]:
        """Read all motor positions."""
        positions = {}
        for motor_name in self.motor_configs:
            pos = self.read_motor_position(motor_name)
            if pos is not None:
                positions[motor_name] = pos
        return positions

    def move_multiple_motors(self, positions: dict[str, float], smooth: bool = True) -> bool:
        """Move multiple motors simultaneously."""
        if smooth:
            return self._smooth_move(positions)
        else:
            success = True
            for motor_name, position in positions.items():
                if not self.move_motor(motor_name, position):
                    success = False
            return success

    def _smooth_move(self, target_positions: dict[str, float], steps: int = 20, step_delay: float = 0.05) -> bool:
        """Smoothly move to target positions."""
        current_positions = self.read_all_positions()

        # Calculate step increments
        step_increments = {}
        for motor, target in target_positions.items():
            if motor in current_positions:
                current = current_positions[motor]
                step_increments[motor] = (target - current) / steps

        # Execute smooth movement
        for step in range(steps + 1):
            for motor, target in target_positions.items():
                if motor in current_positions and motor in step_increments:
                    current = current_positions[motor]
                    new_position = current + (step_increments[motor] * step)
                    self.move_motor(motor, new_position)

            time.sleep(step_delay)

        return True

    def incremental_move(self, motor_name: str, delta_degrees: float) -> bool:
        """Move motor by a small increment."""
        current_pos = self.read_motor_position(motor_name)
        if current_pos is None:
            return False

        new_pos = current_pos + delta_degrees
        return self.move_motor(motor_name, new_pos)


@tool
def pose_tool(
    action: str,
    robot_id: str = "so101_follower",
    port: str | None = "/dev/ttyACM0",
    pose_name: str | None = None,
    motor_name: str | None = None,
    position: float | None = None,
    delta: float | None = None,
    positions: dict[str, float] | None = None,
    description: str | None = None,
    smooth: bool = True,
    steps: int = 20,
    step_delay: float = 0.05,
    calibration_id: str | None = None,
) -> dict[str, Any]:
    """
    Advanced robot pose management tool with fine motor control.

    Actions:
        Pose Management:
        - "store_pose": Store current robot pose with a name
        - "load_pose": Move robot to a stored pose
        - "list_poses": List all stored poses
        - "delete_pose": Delete a stored pose
        - "show_pose": Display pose information

        Motor Control:
        - "move_motor": Move single motor to position
        - "move_multiple": Move multiple motors simultaneously
        - "incremental_move": Small incremental motor movement
        - "read_position": Read current motor position
        - "read_all": Read all motor positions
        - "calibrate_motor": Interactive motor calibration

        System:
        - "connect": Test robot connection
        - "emergency_stop": De-energize every motor (Torque_Enable=0). The arm
          goes LIMP and falls under gravity - it does not hold position, so
          anything grasped is dropped. Reports an error when any motor could
          not be released.
        - "reset_to_home": Move to safe home position

    Args:
        action: Action to perform
        robot_id: Robot identifier for pose storage
        port: Serial port for robot communication
        pose_name: Name for pose operations
        motor_name: Motor name for single motor operations
        position: Target position in degrees (or 0-100% for gripper)
        delta: Incremental movement in degrees
        positions: Dictionary of motor positions {motor_name: degrees}
        description: Description for stored poses
        smooth: Use smooth interpolated movement
        steps: Number of steps for smooth movement
        step_delay: Delay between movement steps
        calibration_id: lerobot ``--robot.id`` whose calibration to drive the arm
            through (defaults to ``robot_id``). Degrees are converted with the
            same formula lerobot's MotorsBus uses, so a pose here matches what the
            policy / teleop path commands. When no calibration file exists the
            tool warns and falls back to a nominal midpoint mapping, which on a
            real arm is off by up to 13 degrees per joint.

    Returns:
        Dict containing status and response content
    """

    # Initialize managers
    pose_manager = PoseManager(robot_id)

    try:
        if action == "list_poses":
            poses = pose_manager.list_poses()
            if not poses:
                return {
                    "status": "success",
                    "content": [
                        {"text": f"No poses stored for robot {robot_id}"},
                        {"json": {"poses": []}},
                    ],
                }

            # Get detailed pose information
            pose_details = []
            for name in poses:
                pose = pose_manager.get_pose(name)
                if pose is None:
                    continue
                pose_details.append(
                    {
                        "name": name,
                        "description": pose.description or "No description",
                        "timestamp": time.ctime(pose.timestamp),
                        "motors": len(pose.positions),
                    }
                )

            pose_list = "\n".join(
                [f"- {p['name']} - {p['description']} ({p['motors']} motors) - {p['timestamp']}" for p in pose_details]
            )

            return {
                "status": "success",
                "content": [
                    {"text": f"Stored poses for {robot_id}:\n{pose_list}"},
                    {"json": {"poses": pose_details}},
                ],
            }

        if action == "show_pose":
            if not pose_name:
                return {"status": "error", "content": [{"text": "pose_name required"}]}

            pose = pose_manager.get_pose(pose_name)
            if not pose:
                return {"status": "error", "content": [{"text": f"Pose '{pose_name}' not found"}]}

            motor_info = "\n".join([f"  - {motor}: {pos:.2f} deg" for motor, pos in pose.positions.items()])

            return {
                "status": "success",
                "content": [
                    {
                        "text": f"Pose: {pose.name}\n"
                        f"Description: {pose.description or 'None'}\n"
                        f"Created: {time.ctime(pose.timestamp)}\n"
                        f"Motor Positions:\n{motor_info}"
                    },
                    {"json": {"pose": pose.to_dict()}},
                ],
            }

        if action == "delete_pose":
            if not pose_name:
                return {"status": "error", "content": [{"text": "pose_name required"}]}

            if pose_manager.delete_pose(pose_name):
                return {"status": "success", "content": [{"text": f"Deleted pose '{pose_name}'"}]}
            else:
                return {"status": "error", "content": [{"text": f"Pose '{pose_name}' not found"}]}

        # Actions that need motor controller
        if not port:
            return {"status": "error", "content": [{"text": "port required for motor operations"}]}

        # Drive the arm through its lerobot calibration when one exists, so a
        # degree here means the same physical pose the policy / teleop path
        # commands. ``calibration_id`` is the lerobot ``--robot.id`` (default:
        # the tool's ``robot_id``, which also names the pose store).
        calibration = load_lerobot_calibration(calibration_id or robot_id)
        controller = MotorController(port, calibration=calibration)
        if calibration is None:
            logger.warning(
                "pose_tool: no lerobot calibration found for id %r; falling back to the nominal "
                "midpoint mapping (0 deg -> tick %d for every joint). Measured against a real "
                "SO-101 calibration that is off by up to 152 ticks (13 deg) per joint and 2029 "
                "ticks on the gripper, so commanded poses will NOT match what the policy or teleop "
                "path produces. Calibrate with 'lerobot-calibrate --robot.type=so101_follower "
                "--robot.id=%s', or pass calibration_id= for an arm calibrated under another id.",
                calibration_id or robot_id,
                _TICK_SPAN // 2,
                calibration_id or robot_id,
            )

        if action == "connect":
            connected, error = controller.connect()
            if connected:
                controller.disconnect()
                return {"status": "success", "content": [{"text": f"Successfully connected to robot on {port}"}]}
            else:
                return {"status": "error", "content": [{"text": f"{error}"}]}

        if action == "read_position":
            if not motor_name:
                return {"status": "error", "content": [{"text": "motor_name required"}]}

            connected, error = controller.connect()
            if not connected:
                return {"status": "error", "content": [{"text": f"{error}"}]}

            try:
                position = controller.read_motor_position(motor_name)
                if position is not None:
                    unit = "%" if motor_name == "gripper" else " deg"
                    return {
                        "status": "success",
                        "content": [
                            {"text": f"{motor_name}: {position:.2f}{unit}"},
                            {"json": {"position": position}},
                        ],
                    }
                else:
                    return {"status": "error", "content": [{"text": f"Failed to read {motor_name}"}]}
            finally:
                controller.disconnect()

        if action == "read_all":
            connected, error = controller.connect()
            if not connected:
                return {"status": "error", "content": [{"text": f"{error}"}]}

            try:
                positions = controller.read_all_positions()
                if positions:
                    pos_text = "\n".join(
                        [
                            f"  - {motor}: {pos:.2f}{'%' if motor == 'gripper' else ' deg'}"
                            for motor, pos in positions.items()
                        ]
                    )
                    return {
                        "status": "success",
                        "content": [
                            {"text": f"Current robot positions:\n{pos_text}"},
                            {"json": {"positions": positions}},
                        ],
                    }
                else:
                    return {"status": "error", "content": [{"text": "Failed to read positions"}]}
            finally:
                controller.disconnect()

        if action == "store_pose":
            if not pose_name:
                return {"status": "error", "content": [{"text": "pose_name required"}]}

            connected, error = controller.connect()
            if not connected:
                return {"status": "error", "content": [{"text": f"{error}"}]}

            try:
                current_positions = controller.read_all_positions()
                if not current_positions:
                    return {"status": "error", "content": [{"text": "Failed to read current positions"}]}

                pose = pose_manager.store_pose(pose_name, current_positions, description)

                pos_text = "\n".join(
                    [
                        f"  - {motor}: {pos:.2f}{'%' if motor == 'gripper' else ' deg'}"
                        for motor, pos in current_positions.items()
                    ]
                )

                return {
                    "status": "success",
                    "content": [
                        {"text": f"Stored pose '{pose_name}':\n{pos_text}"},
                        {"json": {"pose": pose.to_dict()}},
                    ],
                }
            finally:
                controller.disconnect()

        if action == "load_pose":
            if not pose_name:
                return {"status": "error", "content": [{"text": "pose_name required"}]}

            pose = pose_manager.get_pose(pose_name)
            if not pose:
                return {"status": "error", "content": [{"text": f"Pose '{pose_name}' not found"}]}

            # Validate pose
            is_valid, msg = pose_manager.validate_pose(pose)
            if not is_valid:
                return {"status": "error", "content": [{"text": f"Pose validation failed: {msg}"}]}

            connected, error = controller.connect()
            if not connected:
                return {"status": "error", "content": [{"text": f"{error}"}]}

            try:
                success = controller.move_multiple_motors(pose.positions, smooth)
                if success:
                    return {
                        "status": "success",
                        "content": [
                            {"text": f"Moved to pose '{pose_name}'"},
                            {"json": {"target_positions": pose.positions}},
                        ],
                    }
                else:
                    return {"status": "error", "content": [{"text": f"Failed to move to pose '{pose_name}'"}]}
            finally:
                controller.disconnect()

        if action == "move_motor":
            if not motor_name or position is None:
                return {"status": "error", "content": [{"text": "motor_name and position required"}]}

            connected, error = controller.connect()
            if not connected:
                return {"status": "error", "content": [{"text": f"{error}"}]}

            try:
                success = controller.move_motor(motor_name, position)
                if success:
                    unit = "%" if motor_name == "gripper" else " deg"
                    return {"status": "success", "content": [{"text": f"Moved {motor_name} to {position}{unit}"}]}
                else:
                    return {"status": "error", "content": [{"text": f"Failed to move {motor_name}"}]}
            finally:
                controller.disconnect()

        if action == "move_multiple":
            if not positions:
                return {"status": "error", "content": [{"text": "positions dict required"}]}

            connected, error = controller.connect()
            if not connected:
                return {"status": "error", "content": [{"text": f"{error}"}]}

            try:
                success = controller.move_multiple_motors(positions, smooth)
                if success:
                    pos_text = "\n".join(
                        [
                            f"  - {motor}: {pos:.2f}{'%' if motor == 'gripper' else ' deg'}"
                            for motor, pos in positions.items()
                        ]
                    )
                    return {"status": "success", "content": [{"text": f"Moved multiple motors:\n{pos_text}"}]}
                else:
                    return {"status": "error", "content": [{"text": "Failed to move motors"}]}
            finally:
                controller.disconnect()

        if action == "incremental_move":
            if not motor_name or delta is None:
                return {"status": "error", "content": [{"text": "motor_name and delta required"}]}

            connected, error = controller.connect()
            if not connected:
                return {"status": "error", "content": [{"text": f"{error}"}]}

            try:
                success = controller.incremental_move(motor_name, delta)
                if success:
                    unit = "%" if motor_name == "gripper" else " deg"
                    sign = "+" if delta >= 0 else ""
                    return {"status": "success", "content": [{"text": f"Moved {motor_name} by {sign}{delta}{unit}"}]}
                else:
                    return {"status": "error", "content": [{"text": f"Failed to move {motor_name}"}]}
            finally:
                controller.disconnect()

        if action == "reset_to_home":
            # Define safe home position
            home_positions = {
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": 0.0,
            }

            connected, error = controller.connect()
            if not connected:
                return {"status": "error", "content": [{"text": f"{error}"}]}

            try:
                success = controller.move_multiple_motors(home_positions, smooth=True)
                if success:
                    return {
                        "status": "success",
                        "content": [
                            {"text": "Robot moved to home position"},
                            {"json": {"home_positions": home_positions}},
                        ],
                    }
                else:
                    return {"status": "error", "content": [{"text": "Failed to move to home position"}]}
            finally:
                controller.disconnect()

        if action == "emergency_stop":
            # This handler used to return "Emergency stop executed (torque
            # disabled)" while executing NO code at all - a fabricated safety
            # confirmation, the worst possible failure on a safety path: an
            # operator (or an agent) reading success would believe a moving arm
            # had been released. It now actually writes Torque_Enable=0 to every
            # motor, and reports failure when it cannot.
            connected, error = controller.connect()
            if not connected:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"EMERGENCY STOP FAILED - the arm was NOT de-energized: {error}. "
                                "Use the hardware power cutoff."
                            )
                        }
                    ],
                }
            try:
                failed = controller.disable_torque()
            finally:
                controller.disconnect()

            if failed:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                "EMERGENCY STOP INCOMPLETE - torque is still enabled on "
                                f"{sorted(failed)}; those joints are NOT released. Use the "
                                "hardware power cutoff."
                            )
                        },
                        {"json": {"torque_disabled": False, "failed_motors": sorted(failed)}},
                    ],
                }
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            "Emergency stop executed: Torque_Enable=0 written to all "
                            f"{len(controller.motor_configs)} motors. The arm is de-energized and "
                            "will fall limp under gravity - it is NOT holding position."
                        )
                    },
                    {"json": {"torque_disabled": True, "motors": sorted(controller.motor_configs)}},
                ],
            }

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}\n"
                        "Available actions: store_pose, load_pose, list_poses, delete_pose, show_pose, "
                        "move_motor, move_multiple, incremental_move, read_position, read_all, "
                        "connect, reset_to_home, emergency_stop"
                    }
                ],
            }

    except Exception as e:
        logger.error(f"Pose tool error: {e}")
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}
