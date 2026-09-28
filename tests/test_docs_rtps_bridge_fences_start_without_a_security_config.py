"""A documented pure-RTPS bridge construction must start on a fresh install.

``Robot(ros2_bridge=True, ros2_transport="rtps")`` defaults ``ros2_commands`` to
``True``, and on that transport an enabled inbound ``joint_command`` surface is
refused by ``HardwareRtpsBridge`` unless a ``dds_security_config`` is supplied or
the operator sets ``STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE``. The gate is
deliberate; a docs fence that reaches it without either is not: copied as
written into a fresh shell it raises ``ValueError: Refusing to start an inbound
joint_command surface on an unsecured DDS graph`` on the page's first bridge
line, with the remedy several sections below.

So every ``Robot(...)`` fence on ``docs/**`` and the README that selects the
RTPS transport is driven through the documented route (``Robot._init_ros_bridge``)
with the fence's own keywords, the opt-out unset and a cyclonedds stand-in, and
must construct. A fence with no security config is the fresh-install path and
must not hit the gate; a fence that supplies one (``docs/learn/ros2.md`` does,
with ``file:`` URIs) is the remedy, and its dict must be one the validator
accepts: every key the bridge requires, spelled the way the code spells it.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

import strands_robots
from strands_robots.hardware_robot import Robot
from strands_robots.ros_telemetry import ROS2_INSECURE_ENV
from tests.test_ros2_command_surface_flag_domain import _init_bridge, _robot, fake_dds  # noqa: F401,F811

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_PYTHON_FENCE = re.compile(r"```python[^\n]*\n(.*?)```", re.DOTALL)


def _rtps_constructions() -> list[tuple[str, dict[str, Any]]]:
    """Every ``Robot(...)`` docs call selecting ``ros2_transport="rtps"``, with the bridge keywords it passes."""
    found: list[tuple[str, dict[str, Any]]] = []
    pages = [
        p for p in sorted(_REPO_ROOT.glob("docs/**/*.md")) if "robots" not in p.relative_to(_REPO_ROOT / "docs").parts
    ]
    pages.append(_REPO_ROOT / "README.md")
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for match in _PYTHON_FENCE.finditer(text):
            fence_line = text.count("\n", 0, match.start(1))
            try:
                tree = ast.parse(match.group(1))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Robot"):
                    continue
                keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                transport = keywords.get("ros2_transport")
                if not (isinstance(transport, ast.Constant) and transport.value == "rtps"):
                    continue
                bridge_params = set(inspect.signature(Robot._init_ros_bridge).parameters) - {"self"}
                kwargs = {name: ast.literal_eval(value) for name, value in keywords.items() if name in bridge_params}
                found.append((f"{page.relative_to(_REPO_ROOT)}:{fence_line + node.lineno}", kwargs))
    return found


_CASES = _rtps_constructions()
_FRESH = [(where, kw) for where, kw in _CASES if "dds_security_config" not in kw]
_SECURED = [(where, kw) for where, kw in _CASES if "dds_security_config" in kw]


def test_the_docs_still_show_an_rtps_bridge_construction() -> None:
    assert _CASES, "no docs fence constructs Robot(..., ros2_transport='rtps'); the graders below read nothing"


def _construct(where: str, kwargs: dict[str, Any]) -> Any:
    kwargs = {key: value for key, value in kwargs.items() if key not in {"ros2_transport", "ros2_domain"}}
    try:
        bridge = _init_bridge(_robot(), **kwargs)
    except ValueError as exc:
        pytest.fail(f"{where}: Robot({kwargs}) copied from the docs is refused: {exc}")
    assert bridge is not None, f"{where}: the fence asks for a bridge and none was built"
    return bridge


@pytest.mark.parametrize(("where", "kwargs"), _FRESH, ids=[where for where, _ in _FRESH])
def test_a_fence_without_a_security_config_constructs_on_a_fresh_install(
    where: str,
    kwargs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    fake_dds: dict[str, list[str]],  # noqa: F811
) -> None:
    monkeypatch.delenv(ROS2_INSECURE_ENV, raising=False)
    _construct(where, kwargs)


@pytest.mark.parametrize(("where", "kwargs"), _SECURED, ids=[where for where, _ in _SECURED])
def test_a_fence_with_a_security_config_supplies_every_key_the_bridge_requires(
    where: str, kwargs: dict[str, Any]
) -> None:
    """The remedy the page shows is the one the validator accepts, key for key.

    Graded through the bridge's own validator rather than by constructing: the
    secured participant needs the real cyclonedds QoS module, which the stand-in
    does not provide, and the validator is exactly the gate the fence has to pass.
    """
    from strands_robots.hardware_rtps_bridge import HardwareRtpsBridge
    from strands_robots.ros_telemetry import _DDS_SECURITY_OPTIONAL_KEY, _DDS_SECURITY_REQUIRED_KEYS

    config = kwargs["dds_security_config"]
    unknown = set(config) - set(_DDS_SECURITY_REQUIRED_KEYS) - {_DDS_SECURITY_OPTIONAL_KEY}
    assert not unknown, f"{where}: the fence spells {sorted(unknown)}, which the bridge never reads"
    try:
        accepted = HardwareRtpsBridge._validate_dds_security_config(config)
    except ValueError as exc:
        pytest.fail(f"{where}: the dds_security_config copied from the docs is refused: {exc}")
    assert accepted == config
    # The fence turns the command surface on; with this config it must not be the insecure path.
    assert kwargs.get("ros2_commands", True) is True
