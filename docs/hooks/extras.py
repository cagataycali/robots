"""mkdocs hook: the install extras table, read from pyproject.toml.

``{{extras:table}}`` in a page becomes a Markdown table of every
``[project.optional-dependencies]`` group with the packages it pulls in, and
``{{extras:python}}`` becomes the ``requires-python`` line. The hook reads the
file with :mod:`tomllib` and never imports ``strands_robots``, so the docs
build works without the package installed. An unknown token warns, which
``mkdocs build --strict`` turns into a failed build.
"""

from __future__ import annotations

import logging
import re
import tomllib
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.extras")

_REPO = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO / "pyproject.toml"
_TOKEN = re.compile(r"\{\{\s*extras:([a-z_]+)\s*\}\}")
_SELF = "strands-robots["
_PIN = re.compile(r"^([A-Za-z0-9_.\-]+(?:\[[^\]]*\])?)")

# What each extra is for, in one clause. Keys not in pyproject are ignored;
# an extra without a row here gets an empty purpose column.
_PURPOSE: dict[str, str] = {
    "sim": "asset download through robot_descriptions",
    "sim-mujoco": "MuJoCo simulation, offscreen rendering, IK (the default sim)",
    "sim-newton": "Newton GPU simulation on warp",
    "sim-isaac": "Isaac Sim backend (USD assets)",
    "sim-gs": "Gaussian splat rendering (gsplat)",
    "lerobot": "lerobot drivers, teleoperation, recording, Feetech buses",
    "smolvla": "SmolVLA policies through lerobot",
    "molmoact2": "MolmoAct 2 policies through lerobot",
    "rl": "reinforcement learning on the MuJoCo backend",
    "mesh": "Zenoh fleet mesh",
    "mesh-iot": "AWS IoT Core bridge for the mesh",
    "dashboard": "the web dashboard",
    "device-connect": "device-connect edge agent",
    "ros2": "pure DDS (cyclonedds) ROS 2 bridge",
    "rosbridge": "rosbridge WebSocket client",
    "groot": "GR00T N1.7 through lerobot (policy_type groot)",
    "cosmos3-service": "Cosmos 3 service client",
    "cosmos3-diffusers": "Cosmos 3 through diffusers",
    "cosmos3-sim": "IK solvers for the Cosmos 3 sim path",
    "moveit2": "MoveIt 2 planning client",
    "curobo": "cuRobo planning (no pins, install cuRobo yourself)",
    "wbc": "whole body controller (ONNX checkpoints from the Hub)",
    "kimodo": "Kimodo diffusion policies",
    "protomotions": "ProtoMotions humanoid checkpoints",
    "microduck": "Microduck walking policy",
    "crazyflie": "Crazyflie radio driver",
    "ur": "Universal Robots RTDE driver",
    "earthrover": "Earth Rover HTTP driver",
    "ollama": "Ollama model provider for the agent",
    "inference": "WebSocket inference server",
    "sagemaker": "SageMaker endpoints",
    "all": "a curated bundle; GPU backends, cosmos3, ros2 and the hardware drivers stay opt-in",
    "dev": "tests and linters",
}


@lru_cache(maxsize=1)
def _project() -> dict:
    with _PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]


def _package_names(specs: list[str]) -> str:
    """Render one extra's pins as package names, folding self-references into ``[x]``."""
    names: list[str] = []
    for spec in specs:
        if spec.startswith(_SELF):
            names.append(f"`[{spec[len(_SELF) : -1]}]`")
            continue
        match = _PIN.match(spec)
        names.append(f"`{match.group(1) if match else spec}`")
    return ", ".join(names)


def extras_table() -> str:
    """The install-extras table as markdown."""
    extras = _project()["optional-dependencies"]
    rows = ["| extra | installs | purpose |", "|---|---|---|"]
    for name, specs in extras.items():
        rows.append(f"| `{name}` | {_package_names(specs)} | {_PURPOSE.get(name, '')} |")
    return "\n".join(rows)


def substitute(markdown: str, page_path: str = "<string>") -> str:
    """Expand the extras token in ``markdown``; ``page_path`` names the page in warnings."""

    def _one(match: re.Match[str]) -> str:
        key = match.group(1)
        if key == "table":
            return extras_table()
        if key == "python":
            return str(_project()["requires-python"])
        log.warning("%s: unknown extras token {{extras:%s}} (known: table, python)", page_path, key)
        return match.group(0)

    return _TOKEN.sub(_one, markdown)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ANN001 - mkdocs signature
    """mkdocs hook entry point: expand the extras token."""
    return substitute(markdown, page.file.src_path)
