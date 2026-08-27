"""Q8 / Q22 / Q23 - three small honesty defects on the devices + checkpoints APIs.

* **Q8**: `GET /api/devices → managed` never carried the child's OS pid, so nothing
  in the UI could show or match the process it had just started. The loop variable
  in that comprehension was itself named ``pid`` while holding a *peer id* - the
  name was taken, which is very likely why the real one never appeared.
* **Q22**: ``rows[: max(limit, len(local))]`` kept every local cache row regardless
  of the requested limit, so a type-ahead asking for 1 row got 16; 0 and -5 also
  got 16.
* **Q23**: `GET /api/devices/logs/{unknown}` answered **200** with an ``error``
  body, so ``res.ok`` was true for a peer that does not exist.
"""

from __future__ import annotations

import types

import pytest

from strands_robots.dashboard.checkpoints import MAX_LIMIT, clamp_limit

# --------------------------------------------------------------------------
# Q22 - a limit is a promise to the caller
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "asked,expected",
    [(1, 1), (5, 5), (15, 15), (40, 40), (41, 40), (1000, MAX_LIMIT), (0, 1), (-5, 1)],
)
def test_the_limit_is_clamped_at_both_ends(asked, expected):
    assert clamp_limit(asked) == expected


@pytest.mark.parametrize("junk", ["junk", None, "", [], {}, 3.7])
def test_junk_falls_back_to_the_default_instead_of_exploding(junk):
    got = clamp_limit(junk)
    assert got == 15 or (junk == 3.7 and got == 3)


def test_a_type_ahead_asking_for_one_row_gets_one_row(monkeypatch):
    """The bug as the UI met it: 16 locally cached checkpoints, limit=1 -> 16."""
    from strands_robots.dashboard import checkpoints

    local = [{"repo_id": f"local/ckpt-{i}", "source": "local"} for i in range(16)]
    monkeypatch.setattr(checkpoints, "local_checkpoints", lambda q: local)
    monkeypatch.setattr(checkpoints, "hub_search", lambda q, limit: ([], None))
    monkeypatch.setattr(checkpoints, "hf_auth_state", lambda: {})

    assert len(checkpoints.search("ckpt", limit=1)["results"]) == 1
    assert len(checkpoints.search("ckpt", limit=5)["results"]) == 5
    assert len(checkpoints.search("ckpt", limit=0)["results"]) == 1


def test_local_rows_still_win_the_space_they_are_given(monkeypatch):
    """The old expression existed to protect local rows; ordering does that job."""
    from strands_robots.dashboard import checkpoints

    local = [{"repo_id": "local/mine", "source": "local"}]
    remote = [{"repo_id": f"hub/other-{i}", "source": "hub"} for i in range(30)]
    monkeypatch.setattr(checkpoints, "local_checkpoints", lambda q: local)
    monkeypatch.setattr(checkpoints, "hub_search", lambda q, limit: (remote, None))
    monkeypatch.setattr(checkpoints, "hf_auth_state", lambda: {})

    out = checkpoints.search("x", limit=3)
    assert [r["repo_id"] for r in out["results"]][0] == "local/mine"
    assert len(out["results"]) == 3
    # ...and the caller can still tell there was more behind the limit.
    assert out["total_matched"] == 31


# --------------------------------------------------------------------------
# Q8 - the managed payload carries the pid it spawned
# --------------------------------------------------------------------------

def _dm_with(process):
    from strands_robots.dashboard.device_manager import DeviceManager, ManagedRobot

    dm = DeviceManager.__new__(DeviceManager)
    dm.robots = {"so101-arm-1": ManagedRobot(
        peer_id="so101-arm-1", robot_name="so101", mode="real",
        port="/dev/cu.usbmodem1", process=process, started_at=1.0,
    )}
    return dm


def _payload(dm, monkeypatch):
    """Call the real devices() with only the HARDWARE scans stubbed."""
    from strands_robots.dashboard import device_manager as dmod

    monkeypatch.setattr(dmod, "scan_serial_ports", lambda: [])
    # _cameras also takes the mesh's frame evidence now (U14: configured != streaming).
    monkeypatch.setattr(dmod.DeviceManager, "_cameras", lambda self, refresh=False, live_cameras=None: [])
    monkeypatch.setattr(dmod.DeviceManager, "_camera_names", lambda self, refresh=False: [])
    return dmod.DeviceManager.devices(dm)


def test_a_live_managed_robot_reports_its_pid(monkeypatch):
    dm = _dm_with(types.SimpleNamespace(pid=4242, poll=lambda: None))
    entry = _payload(dm, monkeypatch)["managed"]["so101-arm-1"]
    assert entry["pid"] == 4242
    assert entry["alive"] is True
    assert entry["returncode"] is None


def test_the_entry_is_still_keyed_by_peer_id_not_by_pid(monkeypatch):
    """The loop variable used to be named `pid` while holding a peer id -- the
    name was taken, which is why the real pid never made it into the payload."""
    dm = _dm_with(types.SimpleNamespace(pid=4242, poll=lambda: None))
    managed = _payload(dm, monkeypatch)["managed"]
    assert list(managed) == ["so101-arm-1"]
    assert managed["so101-arm-1"]["peer_id"] == "so101-arm-1"


def test_a_dead_child_reports_its_pid_and_exit_code_together(monkeypatch):
    """A pid for a dead process is what the logs refer to, so it is reported
    ALONGSIDE alive=False rather than blanked."""
    dm = _dm_with(types.SimpleNamespace(pid=4242, poll=lambda: 1))
    entry = _payload(dm, monkeypatch)["managed"]["so101-arm-1"]
    assert entry["pid"] == 4242
    assert entry["alive"] is False
    assert entry["returncode"] == 1


def test_a_never_started_child_reports_none_for_both(monkeypatch):
    dm = _dm_with(None)
    entry = _payload(dm, monkeypatch)["managed"]["so101-arm-1"]
    assert entry["pid"] is None and entry["returncode"] is None and entry["alive"] is False
