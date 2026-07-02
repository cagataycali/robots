"""Enforce the Strands tool-result contract across ``strands_robots``.

Tool results must expose exactly ``{"status", "content"}`` at the top level
(the runtime may add ``toolUseId``). Structured telemetry belongs inside
``content`` as a ``{"json": {...}}`` block, never as extra top-level keys --
extras are dropped by the agent runtime and never reach ``agent.messages``.

The repo-wide ``test_no_extra_top_level_keys`` test is the standing guard: it
statically scans every tool-result-shaped dict literal in the package and
fails if any carries extra top-level keys. The remaining tests apply
``assert_strands_tool_result`` to live results from representative call sites
(teleop stats, simulation policy rollout, action dispatch).
"""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest

import strands_robots
from strands_robots.teleop_mixin import TeleopMixin
from tests.tool_result_contract import (
    VALID_TOP_LEVEL_KEYS,
    assert_strands_tool_result,
)

_PKG_ROOT = Path(strands_robots.__file__).resolve().parent


def _tool_result_violations() -> list[str]:
    """Statically find tool-result dict literals with extra top-level keys.

    A dict literal counts as tool-result-shaped when it has constant string
    keys including both ``status`` and ``content``. Any key outside
    ``{"status", "content", "toolUseId"}`` is a contract violation.
    """
    violations: list[str] = []
    for path in _PKG_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if len(keys) != len(node.keys):  # non-constant / **spread keys present
                continue
            ks = set(keys)
            if {"status", "content"} <= ks:
                extra = ks - set(VALID_TOP_LEVEL_KEYS)
                if extra:
                    rel = path.relative_to(_PKG_ROOT.parent)
                    violations.append(f"{rel}:{node.lineno} extra_keys={sorted(extra)}")
    return violations


def test_no_extra_top_level_keys():
    """No tool-result-shaped return in the package may carry extra top-level keys."""
    violations = _tool_result_violations()
    assert not violations, "Tool-result contract violations found:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# Representative call site 1: TeleopMixin._teleop_stats (success/degraded/error)
# ---------------------------------------------------------------------------


class _FakeHost(TeleopMixin):
    def __init__(self):
        self.tool_name_str = "ct_host"
        self.mesh = None
        self.peer_id = None
        self._send_lock = threading.Lock()

    def send_action(self, action, robot_name=None, n_substeps=1):  # noqa: ARG002
        return {"status": "success", "content": [{"text": "ok"}]}


@pytest.mark.parametrize(
    ("frames", "errors", "expected"),
    [
        (100, 0, "success"),  # clean run
        (100, 5, "degraded"),  # some per-frame errors, mostly fine
        (0, 0, "success"),  # no frames, no errors (idle stop)
        (10, 10, "error"),  # every frame errored
        (0, 3, "error"),  # no good frames at all
    ],
)
def test_teleop_stats_contract_and_status(frames, errors, expected):
    host = _FakeHost()
    host._ensure_teleop_state()
    host._teleop_frames = frames
    host._teleop_errors = errors
    host._teleop_start_time = 1.0  # non-zero so elapsed/hz compute

    result = host._teleop_stats(blocking=True)
    assert_strands_tool_result(result)
    assert result["status"] == expected

    # Telemetry must live in the json content block, not at the top level.
    json_blocks = [b["json"] for b in result["content"] if "json" in b]
    assert len(json_blocks) == 1
    telemetry = json_blocks[0]
    assert telemetry["frames"] == frames
    assert telemetry["errors"] == errors
    assert "hz_actual" in telemetry
    assert telemetry["status"] == expected


def test_teleop_status_and_list_contract():
    host = _FakeHost()
    host._ensure_teleop_state()
    assert_strands_tool_result(host.get_teleoperate_status())
    assert_strands_tool_result(host.list_teleops())


# ---------------------------------------------------------------------------
# Representative call sites 2 & 3: Simulation.run_policy + action dispatch
# ---------------------------------------------------------------------------


@pytest.fixture
def sim():
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    s = Simulation(tool_name="contract_sim", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


def test_run_policy_result_contract(sim):
    result = sim.run_policy(policy_provider="mock", n_steps=2)
    assert_strands_tool_result(result)


def test_dispatch_action_results_contract(sim):
    for action, params in [
        ("list_robots", {}),
        ("reset", {}),
        ("step", {"n_steps": 1}),
        ("get_gravity", {}),
    ]:
        assert_strands_tool_result(sim._dispatch_action(action, params))
