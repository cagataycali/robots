"""Measure what an out-of-domain max_onframe_failures does to the GH #117 watchdog."""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import strands_robots.simulation.policy_runner as pr_mod

print("TREE:", Path(pr_mod.__file__).parents[2], flush=True)

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.policy_runner import PolicyRunner


class _Capture(logging.Handler):
    """Collect emitted records; record a format failure the way logging does."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []
        self.format_errors = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception as e:  # a %d against a non-integer
            self.format_errors += 1
            self.messages.append(f"<LOGGING FORMAT ERROR: {type(e).__name__}: {e}>")


VALUES: list[tuple[str, object]] = [
    ("3 (usable)", 3),
    ("None (default 5)", None),
    ("0", 0),
    ("-5", -5),
    ("True", True),
    ("2.7", 2.7),
    ("nan", math.nan),
    ("inf", math.inf),
    ("'5'", "5"),
    ("[5]", [5]),
]

rows = []
for label, value in VALUES:
    sim = Simulation(tool_name="probe", mesh=False)
    sim.create_world()
    sim.add_robot(name="alice", data_config="so100")
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("alice"))

    calls = {"n": 0}

    def always_fails(step: int, obs: dict, action: dict) -> None:
        calls["n"] += 1
        raise ValueError(f"boom-{step}")

    cap = _Capture()
    pr_mod.logger.addHandler(cap)
    old_level = pr_mod.logger.level
    pr_mod.logger.setLevel(logging.WARNING)
    try:
        runner = PolicyRunner(sim)
        # duration 2.0 s at 50 Hz = 100 steps if nothing aborts.
        result = runner.run(
            "alice",
            policy,
            duration=2.0,
            control_frequency=50,
            fast_mode=True,
            on_frame=always_fails,
            max_onframe_failures=value,  # type: ignore[arg-type]
        )
        status = result.get("status")
        text = result["content"][0]["text"] if result.get("content") else ""
    except BaseException as e:  # noqa: BLE001 - an escape is the finding
        status = f"RAISED {type(e).__name__}"
        text = str(e)[:200]
    finally:
        pr_mod.logger.removeHandler(cap)
        pr_mod.logger.setLevel(old_level)
        sim.cleanup()

    aborted = "times in a row" in text
    rows.append(
        {
            "label": label,
            "status": status,
            "hook_calls": calls["n"],
            "aborted": aborted,
            "warnings": len([m for m in cap.messages if "on_frame hook failed" in m]),
            "log_format_errors": cap.format_errors,
            "first_warning": next(iter(cap.messages), "")[:110],
            "text": text[:150],
        }
    )
    print(
        f"{label:18s} status={status:<26s} hook_calls={calls['n']:<4d} "
        f"aborted={aborted!s:<6s} logfmt_err={cap.format_errors:<4d} | {text[:70]}",
        flush=True,
    )

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/onframe.json")
out.write_text(json.dumps({"tree": str(Path(pr_mod.__file__).parents[2]), "rows": rows}, indent=2))
print("wrote", out)
