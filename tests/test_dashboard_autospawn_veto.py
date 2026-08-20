"""USB auto-spawn must not bring real boards up inside a test process (Q81).

MEASURED 2026-08-20: `tests/test_dashboard_datasets_route_recording.py` builds the app with
`with TestClient(app)`, the startup hook starts the auto-spawn watcher, and the watcher spawned the
saved SO-101 profiles for real. 185 orphaned children (ppid=1) from ~30 runs of that ONE file were
holding cagatay's two arm ports, so the live arm could not read its motors at all
(`[TxRxResult] Port is in use!`) and the dashboard showed a connected arm with zero joints.

This is the same class as Q30 (a pytest sweep e-stopping the real fleet) and Q32 (a process joining
the mesh with STRANDS_MESH=false): the suite reached hardware because nothing in the product asked
whether it was allowed to.
"""
import os

from strands_robots.dashboard.device_manager import AutoSpawnWatcher, autospawn_veto


def test_a_pytest_process_may_not_take_a_serial_port():
    why = autospawn_veto({"PYTEST_CURRENT_TEST": "tests/test_x.py::test_y (call)"})
    assert why is not None
    assert "pytest" in why and "tests/test_x.py" in why, "name the run, so the fix is findable"
    assert "STRANDS_DASHBOARD_AUTOSPAWN=1" in why, "a refusal must name its way past"


def test_pytest_version_alone_is_enough():
    # A child that inherited the env but not the current-test marker is still a test's child.
    assert autospawn_veto({"PYTEST_VERSION": "9.1.1"}) is not None


def test_the_mesh_kill_switch_also_forbids_starting_a_robot_child():
    why = autospawn_veto({"STRANDS_MESH": "false"})
    assert why is not None and "kill switch" in why


def test_an_ordinary_dashboard_process_still_auto_spawns():
    assert autospawn_veto({}) is None
    assert autospawn_veto({"STRANDS_MESH": "true", "HOME": "/Users/cagatay"}) is None


def test_the_operators_own_switch_still_wins_both_ways():
    assert autospawn_veto({"STRANDS_DASHBOARD_AUTOSPAWN": "0"}) == "STRANDS_DASHBOARD_AUTOSPAWN is off"
    assert autospawn_veto({"STRANDS_DASHBOARD_AUTOSPAWN": "off"}) is not None
    # An explicit yes is an OVERRIDE: a refusal with no way past it gets patched out downstream.
    assert autospawn_veto({"STRANDS_DASHBOARD_AUTOSPAWN": "1", "PYTEST_CURRENT_TEST": "t"}) is None
    assert autospawn_veto({"STRANDS_DASHBOARD_AUTOSPAWN": "force", "STRANDS_MESH": "false"}) is None


def test_the_veto_guards_the_door_not_the_logic():
    # AutoSpawnWatcher.enabled() stays the operator's env switch: a test driving a watcher over a
    # FAKE manager exercises logic and takes no port, and tests/test_dashboard_usb_autospawn.py
    # must keep doing exactly that. The refusal lives on the path that scans the REAL bus.
    assert AutoSpawnWatcher.enabled() is True
    assert os.environ.get("PYTEST_CURRENT_TEST"), "PYTEST_CURRENT_TEST is the signal being relied on"
    assert autospawn_veto(dict(os.environ)) is not None, "THIS run may not spawn hardware"


def test_the_real_door_refuses_in_this_very_process():
    from strands_robots.dashboard.device_manager import DeviceManager

    dm = DeviceManager.__new__(DeviceManager)
    dm.robots = {}
    assert dm.start_autospawn() is None, "a pytest process must get no watcher over the real bus"


def test_the_dashboard_app_does_not_start_a_watcher_under_pytest():
    # The whole point, at the level that actually spawned hardware: build the app the way the
    # offending test file does and assert no watcher is created.
    from fastapi.testclient import TestClient

    from strands_robots.dashboard.server import create_app

    app = create_app()
    with TestClient(app):
        assert getattr(app.state, "autospawn_task", None) is None
        assert getattr(app.state.devices, "autospawn", None) is None
