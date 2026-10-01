"""The Start pages' "what you need" claims match what their fences actually do.

The old quickstart closed a five-step whole-loop block with "Steps 1 and 3-real
need hardware; step 2 needs a GPU. Everything else runs in sim." It accounted
for two of the five steps and was wrong about two more in the same way each
time: a step that needed something the page never installed (the ``[mesh]``
extra for step 4, a sourced ROS 2 distro for step 5) was covered by
"everything else".

The Start section is five pages now, and the claims moved with it:

* ``start/index.md``: "Every ``python`` fence on these pages ran against this
  commit on a laptop with no GPU. Fences needing an arm on USB are noted in
  the text, not run." (the Markdown marker is ``title="sketch"``;
  ``docs/hooks/visuals.py`` strips it, so the reader sees a plain fence)
* ``start/first-robot.md``: "No hardware, no GPU."
* ``start/first-agent.md``: "The sim fences run without a model. The two fences
  that call ``agent("...")`` need a model provider."
* ``start/install.md`` gives the install line, ``strands-robots[sim-mujoco]``.

These cells pin each sentence to the fences it describes: a fence that opens a
serial port is a ``sketch`` and a ``sketch`` opens one; the install line the
pages give really does leave the mesh out, so no un-flagged fence on them may
start one or a ROS 2 bridge; the count of model-calling fences is the count the
sentence states; and the ``rclpy`` refusal ``learn/ros2.md`` describes is the one
the library raises.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

from tests._blocked_module import blocked

REPO_ROOT = Path(__file__).resolve().parents[1]
START = REPO_ROOT / "docs" / "start"
ROS2_PAGE = REPO_ROOT / "docs" / "learn" / "ros2.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Reaching the ``rclpy`` probe means constructing a simulation, which needs the
#: sim backend. Absent it there is nothing to grade, rather than something that
#: passes for the wrong reason.
_HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None

_FENCE = re.compile(r"```python([^\n]*)\n(.*?)```", re.DOTALL)
_SERIAL_PORT = re.compile(r"/dev/tty|COM\d")


def _start_pages() -> list[Path]:
    pages = sorted(START.glob("*.md"))
    assert len(pages) >= 5, f"the Start section holds {len(pages)} pages, expected the five the index lists"
    return pages


def _fences(page: Path) -> list[tuple[bool, str]]:
    """Every python fence on ``page`` as ``(is_sketch, source)``."""
    return [("sketch" in attrs, body) for attrs, body in _FENCE.findall(page.read_text(encoding="utf-8"))]


def _needs_hardware(body: str) -> bool:
    """A fence needs an arm on USB when it opens a serial port for a real robot."""
    return 'mode="real"' in body and bool(_SERIAL_PORT.search(body)) and "mock=True" not in body


def test_the_index_makes_the_sketch_promise() -> None:
    text = (START / "index.md").read_text(encoding="utf-8")
    assert "no GPU" in text, "start/index.md lost the 'no GPU' claim about its fences"
    assert "arm on USB are noted in the text, not run" in text, (
        "start/index.md lost the promise that hardware fences are noted in the text and were not run"
    )


def test_every_fence_that_opens_a_serial_port_is_a_sketch() -> None:
    """The index's promise, one direction: hardware fences carry the mark."""
    unmarked = [
        f"{page.name}: {body.strip().splitlines()[0]}"
        for page in _start_pages()
        for is_sketch, body in _fences(page)
        if _needs_hardware(body) and not is_sketch
    ]
    assert not unmarked, "fences that need an arm on USB but are not marked sketch:\n" + "\n".join(unmarked)


def test_every_sketch_opens_a_serial_port() -> None:
    """The other direction: the mark is not spent on fences a laptop can run."""
    sketches = [(page.name, body) for page in _start_pages() for is_sketch, body in _fences(page) if is_sketch]
    assert sketches, "no Start page carries a sketch fence, so the promise describes nothing"
    idle = [f"{name}: {body.strip().splitlines()[0]}" for name, body in sketches if not _needs_hardware(body)]
    assert not idle, "sketch fences that do not open a serial port:\n" + "\n".join(idle)


def test_first_robot_needs_no_hardware_and_no_gpu() -> None:
    """The page says "No hardware, no GPU"; its fences must agree."""
    page = START / "first-robot.md"
    assert "No hardware, no GPU" in page.read_text(encoding="utf-8")
    fences = _fences(page)
    assert fences, "first-robot.md carries no python fence"
    for is_sketch, body in fences:
        assert not is_sketch and 'mode="real"' not in body, "first-robot.md drives a real robot"
        assert not _SERIAL_PORT.search(body), "first-robot.md opens a serial port"
        assert "cuda" not in body.lower() and "device=" not in body, "first-robot.md reaches for a GPU"


def test_first_agent_counts_the_fences_that_need_a_model() -> None:
    """ "The two fences that call ``agent("...")`` need a model provider" is a count, so count."""
    text = (START / "first-agent.md").read_text(encoding="utf-8")
    sentence = re.search(r"The (\w+) fences that call `agent\(\"\.\.\.\"\)` need a model provider", text)
    assert sentence, "first-agent.md lost the sentence naming which fences need a model"
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    stated = words[sentence.group(1).lower()]
    calling = [body for _, body in _fences(START / "first-agent.md") if re.search(r"\bagent\(\s*\"", body)]
    assert len(calling) == stated, f"the page says {stated} fences call agent(...) but {len(calling)} do"
    silent = [body for _, body in _fences(START / "first-agent.md") if body not in calling]
    for body in silent:
        assert "agent(" not in body.replace("Agent(", ""), "a fence the sentence calls model-free calls the agent"


def _extras_the_pages_install() -> set[str]:
    """The extras named by the Start pages' own ``pip install`` commands."""
    installs: set[str] = set()
    for page in _start_pages():
        for match in re.findall(r'pip install "strands-robots\[([a-z0-9,-]+)\]"', page.read_text(encoding="utf-8")):
            installs |= set(match.split(","))
    assert installs, "the Start pages lost their install command"
    return installs


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


def test_every_extra_the_pages_install_exists() -> None:
    extras = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    unknown = _extras_the_pages_install() - set(extras)
    assert not unknown, f"the Start pages install extras pyproject.toml does not define: {sorted(unknown)}"


def test_the_install_the_pages_give_does_not_provide_the_mesh() -> None:
    """The install commands really do leave ``eclipse-zenoh`` out.

    This is what makes a mesh step a requirement to state rather than a detail:
    a reader who ran the command on the install page has no mesh transport.
    """
    installed = {dep for extra in _extras_the_pages_install() for dep in _requirements(extra)}
    assert not any(dep.startswith("eclipse-zenoh") for dep in installed), installed
    assert any(dep.startswith("eclipse-zenoh") for dep in _requirements("mesh"))


def test_no_start_fence_needs_what_the_install_line_leaves_out() -> None:
    """No un-flagged fence starts the mesh or a ROS 2 bridge the install line cannot supply."""
    offenders = [
        f"{page.name}: {body.strip().splitlines()[0]}"
        for page in _start_pages()
        for is_sketch, body in _fences(page)
        if not is_sketch and re.search(r"mesh=True|\.mesh\.|ros2_bridge=True", body)
    ]
    assert not offenders, "Start fences that need [mesh] or a ROS 2 distro without saying so:\n" + "\n".join(offenders)


def test_the_ros2_page_names_the_distro_and_the_missing_wheel() -> None:
    """``learn/ros2.md`` now carries the claim the quickstart used to: rclpy is a distro, not a wheel."""
    text = ROS2_PAGE.read_text(encoding="utf-8")
    assert "setup.bash" in text, "learn/ros2.md no longer tells the reader to source a distro"
    assert "not on PyPI" in text, "learn/ros2.md no longer says rclpy is not on PyPI"


@pytest.mark.skipif(not _HAS_MUJOCO, reason="the sim backend is needed to reach the rclpy probe")
def test_the_bridge_raises_the_import_error_the_page_describes() -> None:
    """Without ``rclpy`` the bridge refuses, naming the shell command to run.

    The absence is established rather than assumed, so this holds on a host with
    a ROS 2 distro sourced too, where the alternative is a cell that skips, or
    one that builds a live node on the domain it meant to grade.
    """
    from strands_robots.simulation import Simulation

    with blocked("rclpy"), pytest.raises(ImportError, match="rclpy") as refusal:
        Simulation(ros2_bridge=True)

    assert "setup.bash" in str(refusal.value)
    assert "PyPI" in str(refusal.value)


#: The arguments each call in first-robot.md's "What each call did" table needs
#: to run; a name missing here is a new row the pin cannot drive yet.
_CALL_ARGS: dict[str, tuple] = {"send_action": ({"1": 0.0},), "step": (1,)}


def _is_envelope(result: object) -> bool:
    return isinstance(result, dict) and "status" in result and isinstance(result.get("content"), list)


@pytest.mark.skipif(not _HAS_MUJOCO, reason="the calls run against a MuJoCo sim robot")
def test_first_robot_names_every_call_that_does_not_return_the_envelope() -> None:
    """The sentence under the table names, as exceptions, exactly the calls that return no envelope."""
    from strands_robots import Robot

    text = (START / "first-robot.md").read_text(encoding="utf-8")
    calls = re.findall(r"^\| `(\w+)\(", text, re.MULTILINE)
    assert "cleanup" in calls and calls[-1] == "cleanup", f"the table rows changed: {calls}"
    sentence = re.search(r"^Every call but (.*?) returns the same envelope", text, re.MULTILINE)
    assert sentence, "first-robot.md lost the sentence saying which calls return the envelope"

    robot = Robot("so101")
    plain = [name for name in calls if not _is_envelope(getattr(robot, name)(*_CALL_ARGS.get(name, ())))]
    assert set(re.findall(r"`(\w+)\(\)`", sentence.group(1))) == set(plain)
