"""mkdocs hook: the robot catalog and one page per robot, generated from the registry.

Two jobs, both filesystem only (no ``strands_robots`` import):

1. ``on_pre_build`` writes ``docs/robots/<name>.md`` for every robot in
   ``strands_robots/registry/robots.json`` and ``docs/robots/<family>/index.md`` for
   every category. A file is rewritten only when its content changes, so
   ``mkdocs serve`` does not loop on its own output and ``git status`` is quiet
   when the registry is unchanged. The output is committed: the nav names the
   files, graders can grep them, and a reviewer sees the diff a registry change
   makes.
2. ``on_page_markdown`` substitutes the tokens: ``{{robot_cards}}`` (every
   robot as a card), ``{{robot_cards:<family>}}`` (one family),
   ``{{robot_family_table:<family>}}`` (the family's rows as a table),
   ``{{robot_chips:<name>}}`` (one robot page's family, joints, sim, real and
   driver chips) and ``{{driver_facts}}`` (every native driver's facts, rendered
   once on the drivers page that each robot page links to).

The "Policies verified on this robot" section is read from
``docs/hooks/data/checkpoints.json``: one row per checkpoint that was run on the
robot first-hand, each with the script or public artifact that produced its
numbers in ``source``. A robot without rows states that no checkpoint has been
verified on it and links the recording page, so no page implies a policy ran.

Hardware facts come from :data:`DRIVERS`, one entry per native driver class,
each line re-read from the driver module it names, and from the registry's
``hardware`` block. The join of "which driver builds this robot" is
:mod:`coverage` (``docs/hooks/coverage.py``), loaded from the same directory.
"""

from __future__ import annotations

import html
import importlib.util
import json
import logging
import re
import sys
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.robot_pages")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_DOCS = _REPO / "docs"
_OUT = _DOCS / "robots"
_MANIFEST = _DOCS / "assets" / "viewer" / "robots.json"
_CHECKPOINTS = _HERE / "data" / "checkpoints.json"
_CDN = "https://cdn.jsdelivr.net/gh/"

_TOKEN_CARDS = re.compile(r"^\{\{\s*robot_cards(?::([a-z_]+))?\s*\}\}\s*$", re.M)
_TOKEN_TABLE = re.compile(r"^\{\{\s*robot_family_table:([a-z_]+)\s*\}\}\s*$", re.M)
_TOKEN_FACTS = re.compile(r"^\{\{\s*driver_facts\s*\}\}\s*$", re.M)
_TOKEN_CHIPS = re.compile(r"^\{\{\s*robot_chips:([A-Za-z0-9_.-]+)\s*\}\}\s*$", re.M)

_GENERATED = "<!-- generated: docs/hooks/robot_pages.py -->"

#: Category id -> (nav label, one sentence for the family page).
FAMILIES: dict[str, tuple[str, str]] = {
    "arm": ("Arms", "Fixed-base manipulators, from desk servos to industrial cells."),
    "bimanual": ("Bimanual", "Two arms on one frame, one action dict."),
    "hand": ("Hands and grippers", "End effectors: dexterous hands and parallel grippers."),
    "humanoid": ("Humanoids", "Bipeds with arms, from 14-servo ducks to 29-joint adults."),
    "mobile": ("Mobile bases", "Wheeled and legged platforms that move through a room."),
    "mobile_manip": ("Mobile manipulators", "A base that carries an arm."),
    "aerial": ("Aerial", "Quadrotors, commanded as a setpoint stream."),
    "expressive": ("Expressive", "Desk robots whose output is posture and attention, not a grasp."),
}

#: Family -> the sentence a simulated robot page ends its fence with: what to do
#: next with this kind of body, pointing at the guide that shows it.
NEXT_STEPS: dict[str, str] = {
    "arm": "Add a cube and camera ([worlds and objects](../learn/simulation/worlds-and-objects.md)), then run a checkpoint on it ([same checkpoint](../start/first-policy.md)).",
    "bimanual": "One action dict drives both arms; [composition](../learn/policies/index.md) runs a policy per arm.",
    "hand": "`robot_joint_names` lists the finger joints a policy drives; set them by name with `set_joint_positions` or from a [policy](../learn/policies/index.md).",
    "humanoid": "Walk it with a whole-body controller ([wbc](../learn/policies/wbc.md)) or a trained gait ([rl](../learn/policies/rl.md)).",
    "mobile": "Command it as a velocity setpoint stream, or train a gait in batch ([rl](../learn/policies/rl.md)).",
    "mobile_manip": "The base and the arm share one action dict; [worlds and objects](../learn/simulation/worlds-and-objects.md) gives it a room.",
    "aerial": "Commanded as a setpoint stream; [simulation](../learn/simulation/index.md) covers gravity, timestep and cameras.",
    "expressive": "Its output is posture and attention; [agents](../learn/agents.md) shows a model driving it as a tool.",
}

#: Native driver class -> the facts a hardware section states. Every line is
#: read from the driver module named in ``module`` at this commit.
DRIVERS: dict[str, dict[str, object]] = {
    "FeetechDriver": {
        "module": "strands_robots/drivers/feetech/driver.py",
        "link": "Feetech STS/SMS serial bus",
        "port": 'serial device of the SCS bus, for example `"/dev/ttyACM0"` or `"/dev/tty.usbserial-*"`',
        "example": '"/dev/ttyACM0"',
        "sdk": "`pyserial` (`pip install pyserial`); calibration file from `lerobot-calibrate`",
        "kwargs": '`baud_rate=1_000_000`, `calibration=<path or records>`, `motor_ids=()`, `timeout=1.0`, `transport="serial"` or `"twin"`',
        "units": "degrees per joint, `gripper` in percent open; keys `shoulder_pan` or `shoulder_pan.pos`",
        "checks": (
            "the bus opens the port and discovers the servo ids on connect",
            "without `calibration=` the driver reads and commands the servo's full travel, not the arm's measured travel; `get_status` reports `calibration_source`",
            "`stop` releases torque on every motor and names any that stayed driven",
            '`transport="twin"` answers the same verbs from the arm\'s MuJoCo model',
        ),
    },
    "DynamixelDriver": {
        "module": "strands_robots/drivers/dynamixel/driver.py",
        "link": "Dynamixel Protocol 2.0 serial bus",
        "port": 'serial device of the U2D2 or bus adapter, for example `"/dev/ttyUSB0"`',
        "example": '"/dev/ttyUSB0"',
        "sdk": "`pyserial` (`pip install pyserial`); calibration file from `lerobot-calibrate` (`koch_follower`)",
        "kwargs": "`baud_rate=1_000_000`, `calibration=<path or records>`, `motor_ids=()`, `timeout=1.0`",
        "units": "degrees per joint, `gripper` in percent open; keys `shoulder_pan` or `shoulder_pan.pos`",
        "checks": (
            "the arm keeps the operating modes `lerobot-calibrate` wrote; the driver does not rewrite EEPROM",
            "a reply whose error byte carries an error number is dropped; the hardware-alert bit alone is not",
            "`stop` releases torque on every motor and names any that stayed driven",
        ),
    },
    "FrankaDriver": {
        "module": "strands_robots/drivers/franka/driver.py",
        "link": "Franka Control Interface (FCI) through `panda-py`",
        "port": "IP address of the arm's control box",
        "example": '"172.16.0.2"',
        "sdk": "`panda-py` (`pip install panda-python`); resolved on connect, never at import",
        "kwargs": "`speed_factor=0.2`, `stream_rate_hz=30.0`",
        "units": "radians; Panda joints `joint1..joint7`, FR3 `fr3_joint1..`, FR3 v2 `fr3v2_joint1..`, the names the arm's own MuJoCo asset uses",
        "checks": (
            "a motion command is refused unless the driver is connected, every joint named belongs to this arm, every value is finite, and all seven joints are given",
            "no 1 kHz torque loop: joint motion goes through `panda-py`'s guarded motion generator, which owns the realtime context",
            "state is sourced at 1000 Hz and downsampled to `stream_rate_hz`; the stride is reported",
        ),
    },
    "G1Driver": {
        "module": "strands_robots/drivers/g1.py",
        "link": "CycloneDDS through `unitree_sdk2py`",
        "port": "the robot's IP, recorded for logging; DDS binds to `network_interface`",
        "example": '"192.168.123.164", network_interface="eth0"',
        "sdk": "`pip install 'strands-robots[ros2]'` then `git clone https://github.com/unitreerobotics/unitree_sdk2_python` and `pip install --no-deps -e ./unitree_sdk2_python`",
        "kwargs": '`network_interface="eth0"`, `battery_floor_pct=15.0`',
        "units": "radians, keyed by the 29 joint names of the `unitree_g1` model",
        "checks": (
            "`send_action` refuses unless the high-level FSM id is one of `500`, `501`, `801` (`HANDSHAKE_FSMS`)",
            "`send_action` refuses under the battery floor, as a separate refusal",
            "`run_policy` rolls a built policy on a 500 Hz thread with a per-step re-gate and a zero-torque frame on exit; `start_task` refuses by name",
        ),
    },
    "Go2Driver": {
        "module": "strands_robots/drivers/go2.py",
        "link": "CycloneDDS through `unitree_sdk2py`",
        "port": "the robot's IP, recorded for logging; DDS binds to `network_interface`",
        "example": '"192.168.123.161", network_interface="eth0"',
        "sdk": "`pip install 'strands-robots[ros2]'` then `git clone https://github.com/unitreerobotics/unitree_sdk2_python` and `pip install --no-deps -e ./unitree_sdk2_python`",
        "kwargs": '`network_interface="eth0"`, `battery_floor_pct=15.0`',
        "units": "radians, keyed by joint name (`GO2_JOINT_INDEX`); an index is never accepted, because the SDK's leg order differs from the model's",
        "checks": (
            "`send_action` refuses until `release_sport_mode()` has confirmed the onboard sport service is released",
            "`send_action` refuses under the battery floor",
            "`rt/lowcmd` frames are `unitree_go` structs; a `unitree_hg` frame fails CRC and is dropped by the robot",
        ),
    },
    "ReachyDriver": {
        "module": "strands_robots/drivers/reachy.py",
        "link": "Reachy daemon REST API plus its real-time link",
        "port": 'daemon host, optionally with a port: `"reachy-a.local"` or `"reachy-a.local:8000"`',
        "example": '"reachy-mini.local:8000"',
        "sdk": "none; `REACHY_HOST`/`REACHY_PORT` are read when `port` is omitted, then `localhost` and `reachy-mini.local` are probed",
        "kwargs": "`api_port=8000`, `media_port=8443`, `tts_url=None`",
        "units": "head pose and antennas inside the shared envelope; a write outside it is refused naming the limit",
        "checks": (
            "`connect_eagerly` probes `GET /api/daemon/status`, which also reports Lite or Wireless hardware",
            "`_imu`, `_pose` and `_battery` are cached from the daemon link and published by the mesh when present",
        ),
    },
    "MicroduckDriver": {
        "module": "strands_robots/drivers/microduck.py",
        "link": "`robotd` JSON-RPC over a unix socket",
        "port": 'a unix socket path, or `"ssh://[user@]host"` to have the driver forward the duck\'s socket',
        "example": '"ssh://radxa@microduck.local"',
        "sdk": "none; `MICRODUCK_SOCKET`, then `MICRODUCK_HOST`, then `/run/robotd.sock` are tried when `port` is omitted",
        "kwargs": "`api_version=31`, `timeout=5.0`, `subscribe_hz=None`",
        "units": "intents: `robot.move` (twist), `robot.head`, `robot.pose`, `robot.do` (skills), `robot.enable`/`robot.relax`",
        "checks": (
            "`robotd` exposes no per-joint write, so `run_policy` and `start_task` refuse and name the intent path; the on-robot policy is the same `alpha_walking.onnx` the sim runs",
            "continuous intents go as JSON-RPC notifications; discrete ones wait for a reply",
        ),
    },
    "RobotiqDriver": {
        "module": "strands_robots/drivers/robotiq/driver.py",
        "link": "Modbus TCP",
        "port": "the gripper's IP address or hostname",
        "example": '"192.168.1.11"',
        "sdk": "none; the codec is `strands_robots.drivers.robotiq.protocol`",
        "kwargs": "`tcp_port=502`, `unit_id=9` (often `0` behind a UR controller), `stroke_mm=85.0`, `speed=1.0`, `force=1.0`",
        "units": "`gripper` or `gripper.pos` as a closed fraction 0.0 to 1.0, or `position`/`aperture_mm` in millimetres",
        "checks": (
            "`connect_eagerly` activates the gripper and waits for `gSTA == ACTIVE`; a 2F-85 ignores every position command until then",
            "`send_action` refuses while the gripper is not activated",
            "`start_task` and `run_policy` refuse: a 1-DOF end effector is commanded as one dimension of the arm's action",
        ),
    },
    "BoosterDriver": {
        "module": "strands_robots/drivers/booster.py",
        "link": "Booster SDK (`booster_robotics_sdk_python`, DDS)",
        "port": "the robot's IP address; an empty string discovers on the default interface",
        "example": '"192.168.10.102"',
        "sdk": "`pip install booster_robotics_sdk_python` (vendor wheel, imported on connect)",
        "kwargs": '`domain_id=0`, `robot_name=None`, `cmd_type="parallel"`',
        "units": "the eight upper-body joints only; head and legs through `rotate_head` and `move`",
        "checks": (
            "`send_action` refuses until `enable_upper_body()` has handed the upper body to the host",
            "every non-upper-body slot is sent `q=0, kp=0, kd=0` so the onboard controller keeps the legs",
            "a write is refused while the fall state is anything but `IS_READY`",
        ),
    },
    "RBY1Driver": {
        "module": "strands_robots/drivers/rby1.py",
        "link": "gRPC through `rby1-sdk`",
        "port": "the robot's `ip:port`",
        "example": '"192.168.30.1:50051"',
        "sdk": "`pip install 'strands-robots[rby1]'`",
        "kwargs": "`control_frequency=50.0`, `priority=1`",
        "units": "radians, torso, arms and head; no wheels or grippers",
        "checks": (
            "a pressed e-stop or faulted control manager is refused before power-on",
            "range and per-step speed are gated on the robot's own dynamics model",
            "`stop` cancels control; the next write opens a new stream",
        ),
    },
    "URDriver": {
        "module": "strands_robots/drivers/ur.py",
        "link": "RTDE through `ur_rtde`",
        "port": 'controller IP or hostname, optionally with `":30004"`',
        "example": '"192.168.1.10"',
        "sdk": "`pip install 'strands-robots[ur]'` (`rtde_control`, `rtde_receive`)",
        "kwargs": "`model=None`, `control_frequency=125.0`, `rtde_frequency=None`",
        "units": "radians, `shoulder_pan_joint .. wrist_3_joint`, the order the MuJoCo assets and the RTDE wire share",
        "checks": (
            "the receive interface opens first: a controller in `PROTECTIVE_STOP` accepts a connection and performs no motion",
            "`send_action` maps onto `servoJ`, gated on the controller mode and on the size of the step",
            "off hardware every read returns its cache and every write refuses `not connected`",
        ),
    },
    "CrazyflieDriver": {
        "module": "strands_robots/drivers/crazyflie.py",
        "link": "CRTP over a Crazyradio through `cflib`",
        "port": '`"radio://<dongle>/<channel>/<rate>/<address>"` or `"usb://0"`',
        "example": '"radio://0/80/2M/E7E7E7E7E7"',
        "sdk": "`pip install 'strands-robots[crazyflie]'`",
        "kwargs": "`setpoint_hz=20`",
        "units": "twists in SI (`wz` in rad/s, converted to the wire's degrees per second in one place)",
        "checks": (
            "a setpoint is a subscription: the driver re-sends the last accepted setpoint at `setpoint_hz` because the firmware cuts thrust when the stream goes quiet",
            "`send_action` returns once the setpoint is latched, not when motion ends",
            "stopping and landing are different verbs",
        ),
    },
    "EarthRoverDriver": {
        "module": "strands_robots/drivers/earthrover.py",
        "link": "HTTP to the vendor `earth-rovers-sdk`",
        "port": '`"http://host:8000"` or a bare `"host:port"`',
        "example": '"http://localhost:8000"',
        "sdk": "`pip install 'strands-robots[earthrover]'` plus the SDK process on the host",
        "kwargs": "`timeout_s=10.0`, `turn_sign=1.0`",
        "units": "`linear`, `angular`, `lamp`, each normalised to `[-1, 1]`",
        "checks": (
            "`POST /control` carries one twist frame; `GET /data` is the telemetry snapshot; `GET /v2/front` and `/v2/rear` are the cameras",
            "`turn_sign=-1.0` corrects a rover observed turning the wrong way, at the call site",
        ),
    },
    "YahboomM3ProDriver": {
        "module": "strands_robots/drivers/yahboom_m3pro.py",
        "link": "the robot's ROS 2 graph, over rosbridge or in-process `rclpy`",
        "port": '`"host[:port]"` of `rosbridge_server`, default `"localhost:9090"`; ignored by `transport="ros2"`',
        "example": '"192.168.1.50:9090"',
        "sdk": "`pip install 'strands-robots[rosbridge]'` from any host, or `rclpy` on the robot (`ROS_DOMAIN_ID=30` on the shipped image)",
        "kwargs": '`transport="rosbridge"`, `"ros2"` or `"twin"`; `timeout_s=5.0`, `joint_signs=(1.0,)*5`, `move_time_ms=1500`',
        "units": "`arm1.pos .. arm5.pos` and `gripper.pos` in radians (the model's vocabulary), converted to servo degrees at the wire; base twists in SI",
        "checks": (
            "every `/cmd_vel` write passes the operator gate: approved by the agent's operator or pre-approved with `STRANDS_ROS2_COMMAND_ALLOW=/cmd_vel`",
            "the firmware zeroes the base after 200 to 500 ms without a message, so a held move is a 10 Hz stream and an explicit zero",
            "`get_observation` returns `{}` on the robot: the board publishes no arm joint-state topic",
        ),
    },
}


def _load_coverage():  # noqa: ANN202 - a sibling hook module
    """Load ``docs/hooks/coverage.py`` under a name that cannot shadow a package."""
    name = "docs_hooks_coverage"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _HERE / "coverage.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def registry() -> dict[str, dict]:
    """The robot registry, read once: ``robots.json`` plus the URDF tail (``registry_view.py``)."""
    return dict(_registry_view().merged())


@lru_cache(maxsize=1)
def manifest() -> dict[str, dict]:
    """The viewer manifest; regenerated by the coordinator's hook when missing."""
    if not _MANIFEST.exists():
        spec = importlib.util.spec_from_file_location("docs_hooks_manifest", _HERE / "manifest.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_manifest()["robots"]
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))["robots"]


@lru_cache(maxsize=1)
def checkpoints() -> dict[str, list[dict[str, str]]]:
    """Checkpoints run on each robot, from ``docs/hooks/data/checkpoints.json``."""
    rows: dict[str, list[dict[str, str]]] = json.loads(_CHECKPOINTS.read_text(encoding="utf-8"))["robots"]
    return rows


def _policies_section(name: str, cov) -> str:  # noqa: ANN001
    """The section that says which checkpoints ran on this robot, or that none did."""
    out: list[str] = ["## Policies verified on this robot", ""]
    rows = checkpoints().get(name, ())
    if rows:
        out += ["| Checkpoint | Provider | Where | What happened |", "|---|---|---|---|"]
        for row in rows:
            out.append(
                f"| {row['checkpoint']} ({row['kind']}) | `{row['provider']}` | {row['where']} | {row['result']}. "
                f"Source: {row['source']} |"
            )
        out.append("")
    else:
        none_yet = (
            "No checkpoint verified on this robot yet. Record one: [Record](../learn/data/record.md), then "
            "[train](../learn/training/lerobot.md) and run it with `run_policy`."
        )
        out += [none_yet, ""]
    if cov.policies:
        providers = ", ".join(f"`{p}`" for p in cov.policies)
        out += [
            f"Providers written for this body: {providers}; the rest are in the [policy matrix](../learn/policies/index.md).",
            "",
        ]
    return "\n".join(out)


def _families_in_order() -> list[str]:
    seen: list[str] = []
    for spec in registry().values():
        if spec["category"] not in seen:
            seen.append(spec["category"])
    return sorted(seen, key=lambda c: list(FAMILIES).index(c) if c in FAMILIES else 99)


def _label(category: str) -> str:
    return FAMILIES.get(category, (category, ""))[0]


def _model_link(base_url: str | None) -> str | None:
    """``https://cdn.jsdelivr.net/gh/<owner>/<repo>@<ref>/<dir>/`` -> markdown link to GitHub."""
    if not base_url or not base_url.startswith(_CDN):
        return None
    rest = base_url[len(_CDN) :].rstrip("/")
    owner, _, rest = rest.partition("/")
    name_ref, _, subdir = rest.partition("/")
    name, _, ref = name_ref.partition("@")
    repo = f"{owner}/{name}"
    label = f"{repo}/{subdir}" if subdir else repo
    href = f"https://github.com/{repo}/tree/{ref}/{subdir}" if subdir else f"https://github.com/{repo}/tree/{ref}"
    return f"[{label}]({href})"


#: What the joints chip counts, as its tooltip. The registry figure is the count
#: ``get_robot_state`` reports (issue #4147): every joint of the loaded model except
#: a floating base, which the state carries as ``base``. Graded per robot by
#: ``scripts/audit_registry_joints.py``.
_JOINTS_TITLE = "joints get_robot_state reports: every joint of the loaded model except a floating base"


def _joints_chip(joints: object) -> str:
    """The ``N joints`` chip, or nothing for a robot that reports no joint (a quadrotor)."""
    if not isinstance(joints, int) or joints <= 0:
        return ""
    return f'<span class="sr-chip" title="{_JOINTS_TITLE}">{joints} joints</span>'


def _chips(name: str, spec: dict, cov, entry: dict) -> str:  # noqa: ANN001
    parts = [
        f'<span class="sr-chip sr-chip-family" data-family="{spec["category"]}">{html.escape(_label(spec["category"]))}</span>'
    ]
    if chip := _joints_chip(spec.get("joints")):
        parts.append(chip)
    if entry.get("sim"):
        parts.append(f'<span class="sr-chip sr-chip-sim">{"sim" if cov.real else "sim only"}</span>')
    if cov.real:
        parts.append('<span class="sr-chip sr-chip-real">real</span>')
    if cov.drivers:
        parts.append(f'<span class="sr-chip sr-chip-driver">driver: {", ".join(cov.drivers)}</span>')
    return '<p class="sr-chips">' + "".join(parts) + "</p>"


def chips(name: str) -> str:
    """One robot page's chips as HTML, what ``{{robot_chips:<name>}}`` renders to."""
    return _chips(name, registry()[name], _load_coverage().row(name), manifest().get(name, {}))


# lerobot's bimanual configs declare no ``port``: each takes ``left_arm_config`` and
# ``right_arm_config``, one single-arm config per side. (module, class) per lerobot type.
BIMANUAL_ARM_CONFIG: dict[str, tuple[str, str]] = {
    "bi_so_follower": ("lerobot.robots.so_follower.config_so_follower", "SOFollowerConfig"),
    "bi_openarm_follower": ("lerobot.robots.openarm_follower.config_openarm_follower", "OpenArmFollowerConfig"),
    "bi_rebot_b601_follower": (
        "lerobot.robots.rebot_b601_follower.config_rebot_b601_follower",
        "RebotB601FollowerConfig",
    ),
}


# lerobot types that connect over the network rather than a serial device: the keyword the
# config declares, an example value for the fence, and the prose the hardware section prints.
LEROBOT_NETWORK_WIRING: dict[str, tuple[str, str, str]] = {
    "unitree_g1": ("robot_ip", '"192.168.123.164"', "`robot_ip=` is the robot's address (DDS over Ethernet)"),
    "lekiwi_client": ("remote_ip", '"192.168.1.50"', "`remote_ip=` is the address of the Pi that runs `lekiwi`"),
    "reachy2": ("ip_address", '"192.168.1.42"', "`ip_address=` is the robot's address (gRPC, `port=` 50065)"),
    "earthrover_mini_plus": ("sdk_url", '"http://localhost:8000"', "`sdk_url=` is where the EarthRover SDK listens"),
}


def _lerobot_wiring(lerobot_type: str) -> tuple[str, str]:
    """(fence keyword and value, prose clause) for how a lerobot robot is addressed."""
    if lerobot_type in LEROBOT_NETWORK_WIRING:
        keyword, example, prose = LEROBOT_NETWORK_WIRING[lerobot_type]
        return f"{keyword}={example}", prose
    return 'port="/dev/ttyACM0"', "`port=` is the serial device"


def _real_fences(name: str, spec: dict, cov) -> list[str]:  # noqa: ANN001
    """The ``mode="real"`` lines, one per driver that builds this robot."""
    lines: list[str] = []
    hardware = spec.get("hardware") or {}
    if cov.lerobot_type in BIMANUAL_ARM_CONFIG:
        module, cls = BIMANUAL_ARM_CONFIG[cov.lerobot_type]
        pin = "" if cov.default_driver == "lerobot" else ', driver="lerobot"'
        lines += [
            f"from {module} import {cls}",
            "",
            f'robot = Robot("{name}", mode="real"{pin},  # lerobot {cov.lerobot_type}: one config per arm, no single port',
            f'              left_arm_config={cls}(port="/dev/ttyACM0"),',
            f'              right_arm_config={cls}(port="/dev/ttyACM1"))',
        ]
    elif cov.lerobot_type:
        pin = "" if cov.default_driver == "lerobot" else ', driver="lerobot"'
        wiring, _ = _lerobot_wiring(cov.lerobot_type)
        lines.append(f'robot = Robot("{name}", mode="real"{pin}, {wiring})  # lerobot {cov.lerobot_type}')
    if cov.native_driver:
        facts = DRIVERS[cov.native_driver]
        pin = "" if cov.default_driver == "strands" else ', driver="strands"'
        lines.append(f'robot = Robot("{name}", mode="real"{pin}, port={facts["example"]})  # {cov.native_driver}')
    if hardware.get("requires_lerobot_from_source"):
        lines.append("# this lerobot type is on lerobot main, not on PyPI: install lerobot from source")
    return lines


def _hardware_section(name: str, spec: dict, cov) -> str:  # noqa: ANN001
    out: list[str] = ["## Hardware", ""]
    hardware = spec.get("hardware") or {}
    if cov.lerobot_type:
        source = (
            " Install lerobot from source: the type is not in the PyPI release."
            if hardware.get("requires_lerobot_from_source")
            else ""
        )
        default = " The default when `driver=` is not given." if cov.default_driver == "lerobot" else ""
        pin = "" if cov.default_driver == "lerobot" else ', driver="lerobot"'
        if cov.lerobot_type in BIMANUAL_ARM_CONFIG:
            cls = BIMANUAL_ARM_CONFIG[cov.lerobot_type][1]
            wiring = (
                f"there is no single `port=`; pass `left_arm_config=` and `right_arm_config=`, one `{cls}` per arm "
                f"with its own `port` and `cameras`."
            )
        else:
            wiring = f"{_lerobot_wiring(cov.lerobot_type)[1]}, `cameras=` the lerobot camera dict."
        out.append(
            f'**lerobot.** `Robot("{name}", mode="real"{pin})` builds lerobot\'s `{cov.lerobot_type}` '
            f"with `pip install 'strands-robots[lerobot]'`; {wiring}{default}{source}"
        )
        out.append("")
    if cov.native_driver:
        facts = DRIVERS[cov.native_driver]
        default = (
            "the default for this robot" if cov.default_driver == "strands" else 'selected with `driver="strands"`'
        )
        out.append(
            f"**`{cov.native_driver}`** ({default}) speaks {facts['link']}: "
            f"[port, SDK, kwargs and checks](../learn/hardware/drivers.md#{cov.native_driver.lower()})."
        )
        out.append("")
    if not cov.drivers:
        out.append(
            f"No driver builds `{name}` for real at this commit; the registry lists no `lerobot_type` and no native driver registers for it."
        )
        out.append("")
    return "\n".join(out)


def driver_facts() -> str:
    """Every native driver's facts, once: the section ``{{driver_facts}}`` expands to.

    One heading per driver (the anchor each robot page links to) with the
    table and checks folded into a collapsed block, so the page scrolls by
    one line per driver.
    """
    out: list[str] = []
    for cls, facts in DRIVERS.items():
        body = [
            "| | |",
            "|---|---|",
            f"| `port=` | {facts['port']} |",
            f"| SDK | {facts['sdk']} |",
            f"| Other kwargs | {facts['kwargs']} |",
            f"| Action keys | {facts['units']} |",
            "",
            "Checks before it writes:",
            "",
            *(f"- {check}" for check in facts["checks"]),  # type: ignore[attr-defined]
        ]
        out += [f"### {cls}", "", f"Speaks {facts['link']}. Source: `{facts['module']}`.", ""]
        out += ['??? info "Port, SDK, kwargs and checks"', ""]
        out += [f"    {line}" if line else "" for line in body]
        out.append("")
    return "\n".join(out)


def _urdf_provenance(name: str, spec: dict) -> str:
    """One line on where a URDF robot's model comes from and what the loader adds."""
    repo, commit = spec.get("repository"), spec.get("commit")
    module = (spec.get("asset") or {}).get("robot_descriptions_module") or f"{name}_description"
    src = f"[{repo}@{commit[:7]}](https://github.com/{repo}/tree/{commit})" if repo and commit else f"`{module}`"
    base = "floating base" if spec.get("floating") else "fixed base"
    joints = spec.get("joints")
    drive = f"{joints} position actuators" if joints else "position actuators"
    return f"URDF from {src}, [compiled for MuJoCo](../learn/simulation/urdf.md) on first use: {drive}, {base}."


def robot_page(name: str) -> str:
    """Markdown for one robot."""
    spec = registry()[name]
    entry = manifest().get(name, {})
    cov = _load_coverage().row(name)
    description = spec.get("description", name)
    sim = bool(entry.get("sim"))
    lines: list[str] = [
        "---",
        f"title: {name}",
        f"description: {json.dumps(description)}",
        "---",
        "",
        _GENERATED,
        "",
        f"# {description}",
        "",
        f"{{{{robot_chips:{name}}}}}",
        "",
    ]
    if sim:
        intro = ""
    elif spec.get("source") == "urdf":
        intro = (
            f"The loader could not compile this description at this commit: {spec.get('refusal', '')}. "
            f'`Robot("{name}")` refuses with that sentence.'
        )
    elif cov.real:
        intro = f'The registry ships no simulation asset for it, so `Robot("{name}")` in the default sim mode refuses by name.'
    else:
        intro = f"`{name}` is registered by name and alias, with no simulation asset and no driver at this commit."
    if intro:
        lines += [intro, ""]
    if spec.get("source") == "urdf":
        lines += [_urdf_provenance(name, spec), ""]
    if sim:
        if entry.get("viewer"):
            lines += [f'<robot-viewer name="{name}"></robot-viewer>', ""]
        elif spec.get("source") == "urdf":
            thumb = entry.get("thumbnail")
            lines += (
                [
                    f'<img class="sr-thumb" src="../{thumb}" alt="{html.escape(name)}, a local MuJoCo render" '
                    'loading="lazy" width="640" height="480">',
                    "",
                ]
                if thumb
                else []
            )
        else:
            lines += [
                "The model has no public source to stream, so this page has no 3D view; the thumbnail is a local render.",
                "",
            ]
        asset = spec.get("asset") or {}
        if asset.get("auto_download", True) is False:
            placement = f"{asset['dir']}/{asset['model_xml']}"
            lines += [
                f"The model is not fetched for you (`auto_download: false`): place `{placement}` under "
                "`~/.strands_robots/assets/` (or `$STRANDS_ASSETS_DIR`) first, or the call refuses with "
                '"model file is not on disk".',
                "",
                '```python title="sketch"',
                "from strands_robots import Robot",
                "",
                f'robot = Robot("{name}")  # needs ~/.strands_robots/assets/{placement} on disk',
                "```",
                "",
            ]
        elif spec.get("source") == "urdf":
            lines += [
                '```python title="sketch"',
                "from strands_robots import Robot",
                "",
                f'robot = Robot("{name}")  # clones the description and compiles the URDF on first use',
                f'print(robot.robot_joint_names("{name}"))',
                "robot.cleanup()",
                "```",
                "",
            ]
        else:
            lines += [
                "```python",
                "from strands_robots import Robot",
                "",
                f'robot = Robot("{name}")',
                f'print(robot.robot_joint_names("{name}"))',
                "robot.cleanup()",
                "```",
                "",
            ]
        lines += [NEXT_STEPS.get(spec["category"], ""), ""]
    if cov.real:
        lines += [
            '```python title="sketch"',
            *_real_fences(name, spec, cov),
            "```",
            "",
        ]
    aliases = spec.get("aliases") or []
    if aliases:
        lines += ["Aliases: " + ", ".join(f"`{a}`" for a in aliases) + ".", ""]
    labels = spec.get("joint_labels")
    if labels:
        lines += ["| Model joint | Action key |", "|---|---|"]
        lines += [f"| `{k}` | `{v}` |" for k, v in labels.items()]
        lines.append("")
    gripper = spec.get("gripper")
    if gripper:
        acts = ", ".join(f"`{a}`" for a in gripper.get("actuators", ()))
        lines += [
            f"Gripper actuator {acts}: closed at the {gripper.get('closed')} end of travel, open at the {gripper.get('open')} end.",
            "",
        ]
    if cov.real or spec.get("hardware"):
        lines += [_hardware_section(name, spec, cov)]
    lines += [_policies_section(name, cov)]
    model = _model_link(entry.get("base_url"))
    if model:
        lines += [f"Model: {model}, scene `{entry.get('scene')}`.", ""]
    return "\n".join(lines)


def family_page(category: str) -> str:
    """Markdown for one family index page."""
    label, sentence = FAMILIES.get(category, (category, ""))
    names = [n for n, s in registry().items() if s["category"] == category]
    lines = [
        "---",
        f"title: {label}",
        "---",
        "",
        _GENERATED,
        "",
        f"# {label}",
        "",
        sentence,
        "",
        f"{{{{robot_cards:{category}}}}}",
        "",
        f"{{{{robot_family_table:{category}}}}}",
        "",
    ]
    if not names:
        lines.insert(8, f"No robot is registered under `{category}` at this commit.")
    return "\n".join(lines)


def _write_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def generate() -> tuple[list[Path], int]:
    """Write every robot and family page; return the paths and how many changed."""
    _OUT.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    changed = 0
    for category in _families_in_order():
        path = _OUT / category / "index.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        changed += _write_if_changed(path, family_page(category))
        written.append(path)
    for name in registry():
        path = _OUT / f"{name}.md"
        changed += _write_if_changed(path, robot_page(name))
        written.append(path)
    return written, changed


def card(name: str, prefix: str) -> str:
    """One robot card as HTML."""
    spec = registry()[name]
    entry = manifest().get(name, {})
    cov = _load_coverage().row(name)
    thumb = entry.get("thumbnail")
    figure = (
        f'<img src="{prefix}{thumb}" alt="{html.escape(name)} rendered from its MuJoCo model" loading="lazy" width="640" height="480">'
        if thumb
        else '<span class="sr-robot-nothumb">no simulation asset</span>'
    )
    badges = ""
    if entry.get("sim"):
        badges += '<span class="sr-chip sr-chip-sim">sim</span>'
    if cov.real:
        badges += '<span class="sr-chip sr-chip-real">real</span>'
    badges += _joints_chip(spec.get("joints"))
    aliases = spec.get("aliases") or []
    alias_html = (
        '<p class="sr-robot-aliases">' + " ".join(f"<code>{html.escape(a)}</code>" for a in aliases[:4]) + "</p>"
        if aliases
        else ""
    )
    return (
        f'<article class="sr-robot" data-family="{spec["category"]}" data-sim="{str(bool(entry.get("sim"))).lower()}" data-real="{str(cov.real).lower()}">'
        f'<a class="sr-robot-thumb" href="{prefix}robots/{name}/">{figure}</a>'
        f'<h3><a href="{prefix}robots/{name}/"><code>{html.escape(name)}</code></a></h3>'
        f'<p class="sr-robot-desc">{html.escape(spec.get("description", ""))}</p>'
        f'<p class="sr-chips"><span class="sr-chip sr-chip-family">{html.escape(_label(spec["category"]))}</span>{badges}</p>'
        f"{alias_html}"
        "</article>"
    )


def cards(category: str | None, prefix: str) -> str:
    """The card grid for a family (or every robot when ``category`` is None)."""
    names = [n for n, s in registry().items() if category is None or s["category"] == category]
    if category is not None and not names:
        log.warning("robot_cards: unknown family %r", category)
    return '<div class="sr-robots" markdown="0">\n' + "\n".join(card(n, prefix) for n in names) + "\n</div>\n"


def family_table(category: str, link_prefix: str = "") -> str:
    """The family table as markdown, one row per robot."""
    cov = _load_coverage()
    lines = ["| Robot | Description | Joints | Sim | Real | Drivers |", "|---|---|---:|:---:|:---:|---|"]
    for name, spec in registry().items():
        if spec["category"] != category:
            continue
        r = cov.row(name)
        entry = manifest().get(name, {})
        joints = spec.get("joints")
        lines.append(
            f"| [`{name}`]({link_prefix}{name}.md) | {spec.get('description', '')} | {joints if isinstance(joints, int) else '-'} | "
            f"{'yes' if entry.get('sim') else '-'} | {'yes' if r.real else '-'} | {', '.join(f'`{d}`' for d in r.drivers) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def substitute(markdown: str, prefix: str, link_prefix: str = "") -> str:
    """Expand the robot cards, chips, driver facts and family table tokens in ``markdown``."""
    markdown = _TOKEN_CARDS.sub(lambda m: cards(m.group(1), prefix), markdown)
    markdown = _TOKEN_FACTS.sub(lambda _m: driver_facts(), markdown)
    markdown = _TOKEN_CHIPS.sub(lambda m: chips(m.group(1)), markdown)
    return _TOKEN_TABLE.sub(lambda m: family_table(m.group(1), link_prefix), markdown)


def _site_prefix(page, config) -> str:  # noqa: ANN001
    """Relative path from this page's built URL to the site root (raw HTML is not rewritten by mkdocs)."""
    url = page.url.strip("/")
    depth = url.count("/") + 1 if url else 0
    if not config.get("use_directory_urls", True):
        depth = page.file.src_path.count("/")
    return "../" * depth


def on_pre_build(config) -> None:  # noqa: ANN001 - mkdocs signature
    """mkdocs hook entry point: regenerate the robot pages before the build."""
    written, changed = generate()
    log.info("robot pages: %d generated under docs/robots/, %d changed", len(written), changed)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001 - mkdocs signature
    """mkdocs hook entry point: expand the robot tokens on a page."""
    if "{{" not in markdown:
        return markdown
    src = page.file.src_path.replace("\\", "/")
    link_prefix = "../" if re.fullmatch(r"robots/[a-z_]+/index\.md", src) else ""
    return substitute(markdown, _site_prefix(page, config), link_prefix)


def _registry_view():  # noqa: ANN202 - a sibling hook module, loaded by path like the others
    """``docs/hooks/registry_view.py``: robots.json merged with the URDF long tail."""
    name = "docs_hooks_registry_view"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / "registry_view.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    paths, n = generate()
    print(f"{len(paths)} pages, {n} changed")
    for p in paths:
        print(p.relative_to(_REPO))
