"""The Q81 tripwire: a green run that orphaned a real robot child must say so."""
from tests.hardware_leak import hardware_leak_report, robot_children

BOOTSTRAP = (
    "<repo>/.venv/bin/python -c import os, sys, time, json "
    "cfg = json.loads(sys.argv[1]) {\"peer_id\": \"so101-arm-2\", \"port\": "
    "\"/dev/cu.usbmodem5AB01584281\", \"mode\": \"real\"}"
)


def test_a_clean_run_says_nothing():
    assert hardware_leak_report([]) == []
    assert hardware_leak_report([{"pid": 5, "cmdline": "/bin/sleep 1"}]) == []
    assert hardware_leak_report(None) == []


def test_a_leaked_robot_child_is_named_with_its_port_and_peer():
    lines = hardware_leak_report([{"pid": 4242, "cmdline": BOOTSTRAP}])
    text = "\n".join(lines)
    assert "ORPHAN 1 REAL ROBOT PROCESS" in text
    assert "4242" in text and "so101-arm-2" in text
    assert "/dev/cu.usbmodem5AB01584281" in text
    assert "Port is in use" in text, "the consequence, not just the fact"


def test_it_says_the_fix_is_to_find_the_new_door():
    # autospawn_veto already closed the known one, so a hit here means a DIFFERENT path reached
    # hardware. A report that only offered cleanup would train the reader to cope with the leak.
    text = "\n".join(hardware_leak_report([{"pid": 1, "cmdline": BOOTSTRAP}]))
    assert "NEW door" in text and "autospawn_veto" in text


def test_it_refuses_to_kill_and_explains_why():
    text = "\n".join(hardware_leak_report([{"pid": 77, "cmdline": BOOTSTRAP}]))
    assert "NOT killed" in text and "torque" in text and "Park the arm first" in text
    assert "kill 77" in text, "the recipe is offered to a human, not run"


def test_a_child_with_no_port_in_its_command_line_is_still_reported():
    lines = hardware_leak_report([{"pid": 9, "cmdline": "python -m strands_robots.something"}])
    assert lines and "no serial port in its command line" in "\n".join(lines)


def test_unrelated_children_are_left_alone():
    assert robot_children([{"pid": 2, "cmdline": "node /usr/local/bin/vite"}]) == []
