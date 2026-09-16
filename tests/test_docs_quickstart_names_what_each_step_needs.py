"""The quickstart's "what you need" line matches what its steps actually do.

The line read "Steps 1 and 3-real need hardware; step 2 needs a GPU. Everything
else runs in sim." It accounted for two of the five steps of the whole-loop
block and was wrong about two more, in the same way each time: a step that needs
something the page never installs was covered by "everything else".

* Step 5 is ``Simulation(ros2_bridge=True)``. ``rclpy`` is not published on
  PyPI - it ships with a system ROS 2 install - so on a machine with no sourced
  distro the constructor raises ``ImportError`` naming
  ``source /opt/ros/<distro>/setup.bash``.
* Step 4 is ``follower.mesh.tell(peer, ...)`` with ``peer`` read off
  ``follower.mesh.peers[0]["peer_id"]`` (two lines since the call grew its
  policy kwargs; the cell pins both, not one spelling of them). The
  mesh transport comes from the ``[mesh]`` extra (``eclipse-zenoh``), which the
  install command at the top of the page does not pull in, and the mesh also
  declines to start until a posture is chosen (``STRANDS_MESH_LOCAL_DEV`` or an
  ACL file). Missing either, the mesh stays off, so ``mesh.peers`` is empty and
  ``peers[0]`` raises ``IndexError``.

These cells pin the sentence to all three of the things it claims: the steps it
describes are still the ones in the code block, the install line it points at
really does leave the mesh out, and the refusal it quotes is the one the library
raises.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

from tests._blocked_module import blocked

REPO_ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = REPO_ROOT / "docs" / "getting-started" / "quickstart.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Reaching the ``rclpy`` probe means constructing a simulation, which needs the
#: sim backend. Absent it there is nothing to grade, rather than something that
#: passes for the wrong reason.
_HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _quickstart() -> str:
    return QUICKSTART.read_text(encoding="utf-8")


def _needs_line() -> str:
    """The closing "what you need" paragraph, below the whole-loop block."""
    match = re.search(r"Steps 1 and 3-real need hardware;.*?\n\n", _quickstart(), re.DOTALL)
    assert match, "the quickstart lost its 'what you need' paragraph"
    return match.group(0)


def _numbered_steps() -> dict[int, str]:
    """Map each ``# <n>.`` step of the whole-loop block to its own source lines.

    Scoped to that one fenced block, and split on the numbered comments inside
    it, so a cell below is about *that step*: a snippet that moved to another
    step - or into the prose - would still satisfy a search over the whole page.
    """
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for raw in _quickstart().splitlines():
        if raw.lstrip().startswith("```"):
            if in_fence:
                blocks.append("\n".join(current))
                current = []
            in_fence = not in_fence
            continue
        if in_fence:
            current.append(raw)
    loop = [b for b in blocks if "# 1. TELEOPERATE" in b]
    assert len(loop) == 1, "the quickstart lost its single whole-loop code block"

    steps: dict[int, list[str]] = {}
    number: int | None = None
    for line in loop[0].splitlines():
        heading = re.match(r"# (\d+)\. ", line)
        if heading:
            number = int(heading.group(1))
            steps[number] = []
        if number is not None:
            steps[number].append(line)
    return {n: "\n".join(lines) for n, lines in steps.items()}


def _extras_the_page_installs() -> list[str]:
    """The extras named by the page's own ``pip install`` command."""
    installs = re.findall(r'strands-robots\[([a-z0-9,-]+)\]"', _quickstart())
    assert installs, "the quickstart lost its install command"
    return installs[0].split(",")


def _requirements(extra: str, seen: frozenset[str] = frozenset()) -> set[str]:
    """Every third-party requirement ``[extra]`` pulls in, following self-refs."""
    extras = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    out: set[str] = set()
    for dep in extras.get(extra, []):
        nested = re.fullmatch(r"strands-robots\[([a-z0-9,-]+)\]", dep)
        if not nested:
            out.add(dep)
            continue
        for name in nested.group(1).split(","):
            if name not in seen:
                out |= _requirements(name, seen | {extra, name})
    return out


@pytest.mark.parametrize(
    ("step", "snippet"),
    [
        (4, 'follower.mesh.peers[0]["peer_id"]'),
        (4, "follower.mesh.tell(peer,"),
        (5, "Simulation(ros2_bridge=True)"),
    ],
)
def test_the_needs_line_describes_the_steps_that_are_there(step: int, snippet: str) -> None:
    """Steps 4 and 5 are still the mesh call and the ROS 2 bridge.

    Step 4 is pinned to its two halves - the ``peers[0]`` read that raises
    ``IndexError`` with the mesh off, and the ``tell`` it feeds - rather than
    to one line joining them: the call carries policy kwargs now, so the page
    splits it, and a cell that pinned the joined spelling went red the moment
    a sibling change split it, while the sentence it guards stayed true.
    """
    assert snippet in _numbered_steps()[step]


@pytest.mark.parametrize(
    "requirement",
    [
        # Step 4: the extra that carries the transport, and the step it belongs to.
        "step 4",
        "[mesh]",
        # Step 5: the distro, and the package that is not on PyPI.
        "step 5",
        "ros 2",
        "rclpy",
    ],
)
def test_the_needs_line_names_what_steps_4_and_5_need(requirement: str) -> None:
    """Each step that needs more than a CPU is named, with what it needs.

    One requirement per case so a line that drops just one of them reports which.
    """
    assert requirement in _needs_line().lower()


def test_the_needs_line_no_longer_covers_those_steps_with_everything_else() -> None:
    """The claim that hid them is gone, not merely qualified."""
    assert "everything else runs in sim" not in _needs_line().lower()


def test_the_install_the_page_gives_does_not_provide_the_mesh() -> None:
    """The page's own install command really does leave ``eclipse-zenoh`` out.

    This is what makes step 4 a requirement to state rather than a detail: a
    reader who ran the command at the top of the page has no mesh transport.
    """
    installed = {dep for extra in _extras_the_page_installs() for dep in _requirements(extra)}

    assert not any(dep.startswith("eclipse-zenoh") for dep in installed), installed
    assert any(dep.startswith("eclipse-zenoh") for dep in _requirements("mesh"))


@pytest.mark.skipif(not _HAS_MUJOCO, reason="the sim backend is needed to reach the rclpy probe")
def test_the_bridge_raises_the_import_error_the_line_describes() -> None:
    """Without ``rclpy`` the bridge refuses, naming the shell command to run.

    The absence is established rather than assumed, so this holds on a host with
    a ROS 2 distro sourced too - where the alternative is a cell that skips, or
    one that builds a live node on the domain it meant to grade.
    """
    from strands_robots.simulation import Simulation

    with blocked("rclpy"), pytest.raises(ImportError, match="rclpy") as refusal:
        Simulation(ros2_bridge=True)

    assert "setup.bash" in str(refusal.value)
