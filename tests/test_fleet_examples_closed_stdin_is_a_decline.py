"""A fleet example's HITL gate treats a closed stdin as a decline, not a crash.

Every fleet example asks the operator ``[y/N]`` through ``input()`` before a
command reaches a robot, and documents a decline as a first-class outcome:
nothing is sent, the run continues, the summary and audit still print.

Measured at 6a8a8ea23 with stdin closed (``</dev/null``, a pipe that ran dry,
a detached run): ``input()`` raised ``EOFError`` and the example died in a
traceback. With ``echo y |`` the first dispatch was approved and executed, then
the second prompt killed the process before the summary - an approved action
with no record of the run. The gate now answers a closed stdin the way it
answers ``N``.
"""

from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FLEET_DIR = _REPO_ROOT / "examples" / "fleet"


def _load(filename: str) -> Any:
    name = f"_fleet_example_stdin_{filename.split('_')[0]}"
    spec = importlib.util.spec_from_file_location(name, _FLEET_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def closed_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRANDS_MESH_HITL_ACTIONS", raising=False)

    def _eof(prompt: str = "") -> str:
        raise EOFError("EOF when reading a line")

    monkeypatch.setattr(builtins, "input", _eof)


@pytest.mark.parametrize(
    "filename",
    [
        "01_skill_dispatch_multi_vendor.py",
        "02_cross_zone_transport.py",
        "05_work_order_dispatch.py",
    ],
)
def test_dispatch_gate_declines_when_stdin_is_closed(
    filename: str, closed_stdin: None, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = _load(filename).make_hitl_gate()

    approved = gate("dispatch", "so101", "task T-01")

    assert approved is False
    assert "declined, no operator on stdin" in capsys.readouterr().out


def test_evacuation_resume_gate_declines_when_stdin_is_closed(
    closed_stdin: None, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load("04_emergency_evacuation.py")

    assert mod._operator_approves("resume the muster lockout") is False
    assert "declined, no operator on stdin" in capsys.readouterr().out
