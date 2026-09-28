"""mkdocs hook: the shipped native driver table, read from the source.

``{{drivers_table}}`` in a page becomes a Markdown table with one row per
robot a native driver registers for. Robot names and driver classes come from
``strands_robots/drivers/__init__.py`` (``_SHIPPED_DRIVERS``) and from each
driver module's ``SUPPORTED_ROBOTS`` tuple, so the table cannot drift from
what ``Robot(name, mode="real", driver="strands")`` builds.

Transport and the install extra are the two facts a module does not declare
as data; they are kept in :data:`_FAMILY` here, keyed by module path, and a
module missing from that map fails the build rather than shipping a blank
cell. Reads only the filesystem: no ``strands_robots`` import.
"""

from __future__ import annotations

import ast
import logging
import re
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.drivers")

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "strands_robots"
_TOKEN = re.compile(r"\{\{\s*drivers_table\s*\}\}")

#: module path -> (transport, install line, page). The page is relative to
#: ``docs/learn/hardware/``.
_FAMILY: dict[str, tuple[str, str, str]] = {
    "strands_robots.drivers.feetech.driver": ("serial (Feetech SCS bus)", "`pip install pyserial`", "feetech-arms.md"),
    "strands_robots.drivers.dynamixel.driver": (
        "serial (Dynamixel Protocol 2.0)",
        "`pip install pyserial`",
        "feetech-arms.md",
    ),
    "strands_robots.drivers.franka.driver": ("ethernet (FCI, libfranka)", "`panda-py`, vendor wheel", "franka.md"),
    "strands_robots.drivers.g1": ("DDS (CycloneDDS)", "`[ros2]` + `unitree_sdk2_python`", "unitree.md"),
    "strands_robots.drivers.go2": ("DDS (CycloneDDS)", "`[ros2]` + `unitree_sdk2_python`", "unitree.md"),
    "strands_robots.drivers.reachy": ("http + WebSocket (reachy daemon)", "`pip install websockets`", "reachy-mini.md"),
    "strands_robots.drivers.microduck": (
        "unix socket, JSON-RPC (robotd)",
        "base install, `ssh` for a remote duck",
        "microduck.md",
    ),
    "strands_robots.drivers.robotiq.driver": ("ethernet (Modbus TCP)", "base install", "drivers.md"),
    "strands_robots.drivers.booster": (
        "DDS (vendor SDK)",
        "`booster_robotics_sdk_python`, vendor wheel",
        "booster-t1.md",
    ),
    "strands_robots.drivers.ur": ("ethernet (RTDE, port 30004)", "`[ur]`", "ur.md"),
    "strands_robots.drivers.crazyflie": ("radio (CRTP over Crazyradio)", "`[crazyflie]`", "drivers.md"),
    "strands_robots.drivers.earthrover": ("http (earth-rovers-sdk)", "`[earthrover]`", "drivers.md"),
    "strands_robots.drivers.yahboom_m3pro": (
        "rosbridge WebSocket / rclpy / twin",
        "`[rosbridge]` or `[ros2]`",
        "drivers.md",
    ),
}


def _module_file(module_path: str) -> Path:
    return _REPO / (module_path.replace(".", "/") + ".py")


def _shipped() -> list[tuple[str, str, tuple[str, ...] | str]]:
    """Parse ``_SHIPPED_DRIVERS`` out of ``drivers/__init__.py`` with ``ast``."""
    tree = ast.parse((_PKG / "drivers" / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        targets = getattr(node, "targets", None) or ([node.target] if hasattr(node, "target") else [])
        if any(isinstance(t, ast.Name) and t.id == "_SHIPPED_DRIVERS" for t in targets):
            return list(ast.literal_eval(node.value))
    raise RuntimeError("_SHIPPED_DRIVERS not found in strands_robots/drivers/__init__.py")


def _supported(module_path: str, attr: str) -> tuple[str, ...]:
    tree = ast.parse(_module_file(module_path).read_text(encoding="utf-8"))
    for node in tree.body:
        targets = getattr(node, "targets", None) or ([node.target] if hasattr(node, "target") else [])
        if any(isinstance(t, ast.Name) and t.id == attr for t in targets):
            return tuple(ast.literal_eval(node.value))
    raise RuntimeError(f"{module_path}.{attr} not found")


@lru_cache(maxsize=1)
def rows() -> list[tuple[str, str, str, str, str]]:
    """One row per (robot, driver): robot, class, transport, install, page."""
    out: list[tuple[str, str, str, str, str]] = []
    for module_path, class_name, names in _shipped():
        if module_path not in _FAMILY:
            raise RuntimeError(f"drivers hook: no transport/extra entry for {module_path}")
        transport, extra, page = _FAMILY[module_path]
        robots = _supported(module_path, names) if isinstance(names, str) else tuple(names)
        for robot in robots:
            out.append((robot, class_name, transport, extra, page))
    return out


def table() -> str:
    """The native-driver table as markdown."""
    lines = ["| robot | driver class | transport | install | setup |", "|---|---|---|---|---|"]
    for robot, cls, transport, extra, page in rows():
        lines.append(f"| `{robot}` | `{cls}` | {transport} | {extra} | [{page[:-3]}]({page}) |")
    return "\n".join(lines)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001 - mkdocs signature
    """mkdocs hook entry point: expand the drivers token."""
    if not _TOKEN.search(markdown):
        return markdown
    return _TOKEN.sub(lambda _m: table(), markdown)


if __name__ == "__main__":
    print(table())
