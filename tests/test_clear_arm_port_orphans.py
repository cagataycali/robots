"""The orphan-clearing script decides safely, and refuses by default (Q80/Q81).

The judgement under test is the one a human should never have to re-derive at 1am: the lsof pid list
for an arm port contains the LIVE arm children as well as the orphans, and killing a live one
mid-motion is the outcome nobody wants.
"""
import importlib.util
import os
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "clear_arm_port_orphans",
    Path(__file__).resolve().parents[1] / "scripts" / "clear_arm_port_orphans.py",
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

BOOTSTRAP = 'python3.12 -c import json,sys {"peer_id": "so101-leader", "port": "/dev/cu.usbmodem5AB0181806"}'


def test_an_orphaned_robot_child_is_a_candidate():
    ok, why = mod.classify_holder(pid=101, ppid=1, cmdline=BOOTSTRAP)
    assert ok is True and "orphaned robot child" in why


def test_a_live_dashboard_child_is_never_killed_even_though_it_holds_the_port():
    # The whole point of the script: the pid list from lsof contains these too.
    ok, why = mod.classify_holder(pid=202, ppid=1, cmdline=BOOTSTRAP, live_pids=frozenset({202}))
    assert ok is False
    assert "LIVE child" in why and "someone's robot" in why


def test_a_process_with_a_living_parent_is_not_an_orphan():
    ok, why = mod.classify_holder(pid=303, ppid=2519, cmdline=BOOTSTRAP)
    assert ok is False and "not orphaned" in why and "2519" in why


def test_an_unrelated_port_holder_is_left_alone():
    ok, why = mod.classify_holder(pid=404, ppid=1, cmdline="/Applications/Arduino.app/MacOS/arduino")
    assert ok is False and "refusing to kill an unrelated process" in why


def test_it_never_kills_itself():
    ok, why = mod.classify_holder(pid=os.getpid(), ppid=1, cmdline=BOOTSTRAP)
    assert ok is False and why == "this script itself"


def test_the_default_is_a_report_and_the_torque_warning_is_in_the_file():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "clear_arm_port_orphans.py").read_text()
    assert "--confirm" in src
    assert "DRY RUN" in src, "a tool that kills by default is the wrong tool for a rig"
    assert "go limp and FALL" in src, "the physical risk must be stated where the decision is made"
    # And it must tell the operator the follow-up, because a freed port does not restart a robot.
    assert "has to be started again" in src
