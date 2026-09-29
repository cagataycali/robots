"""Both agent tool schemas name ``embodiment`` in the ``lerobot_local`` key list.

``policy_config`` is the only per-provider key list an agent driving the schema
ever sees, and ``embodiment`` is the one ``lerobot_local`` keyword every
SO-arm checkpoint needs: without it the state vector is ordered by the
observation's own keys, degree-trained actions reach a radian actuator, and the
camera renames the model was trained with never fire. The docs page uses it in
every example, the ``run_policy`` registry entry accepts it, and the provider's
preflight refusal names it as the remedy - yet neither schema advertised it, so
an agent that read the schema alone had no way to learn the keyword before the
first refusal. The sibling grader
``test_policy_config_keys_are_constructor_params.py`` guarantees every
advertised key exists on the constructor; this one pins the converse for the
one key whose absence costs a rollout.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from strands_robots.hardware_robot import Robot as HwRobot

_SPEC_PATH = Path(__file__).resolve().parents[3] / "strands_robots/simulation/mujoco/tool_spec.json"
_GROUP_RE = re.compile(r"For '?lerobot_local'?: ([^.]*)\.")


def _sim_lerobot_keys() -> list[str]:
    spec = json.loads(_SPEC_PATH.read_text())
    description = str(spec["properties"]["policy_config"]["description"])
    match = _GROUP_RE.search(description)
    assert match, "the sim tool_spec no longer enumerates lerobot_local keys"
    return [k.strip() for k in match.group(1).split(",")]


def _hardware_lerobot_blob() -> str:
    hw = HwRobot.__new__(HwRobot)  # the schema is a plain attribute read; no arm is opened
    hw.tool_name_str = "so101_arm"
    hw.robot = object()
    schema = hw.tool_spec
    description = str(schema["inputSchema"]["json"]["properties"]["policy_config"]["description"])
    match = _GROUP_RE.search(description)
    assert match, "the hardware tool schema no longer enumerates lerobot_local keys"
    return match.group(1)


class TestEmbodimentIsAdvertised:
    def test_the_sim_tool_names_embodiment_for_lerobot_local(self) -> None:
        assert "embodiment" in _sim_lerobot_keys()

    def test_the_hardware_tool_names_embodiment_for_lerobot_local(self) -> None:
        assert re.search(r"\bembodiment\b", _hardware_lerobot_blob())
