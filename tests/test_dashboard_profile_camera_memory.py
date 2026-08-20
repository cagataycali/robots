"""A spawn that does not mention cameras must not forget the ones the operator configured.

U19 asks whether a respawn honours the camera fps/resolution someone chose. The plumbing does:
the profile carries the full lerobot camera mapping and AutoSpawnWatcher respawns from it. But
`ProfileStore.save()` protected only the MEASURED role fields, while every spawn writes
`"cameras": cameras` -- so a camera-less spawn (the watcher on a replug, a joints-only spawn from
the run form, a CLI spawn) stored None over the mapping. The next automatic respawn then brought
the arm up BLIND, and the U19 reconfigure editor opened blank, with nothing saying a choice had
been dropped. Same trap that once ate a measured role, different victim.
"""
import pytest

from strands_robots.dashboard.device_manager import ProfileStore

TUNED = {
    "top": {"index_or_path": 0, "fps": 15, "width": 1280, "height": 720},
    "wrist": {"index_or_path": 1, "fps": 30, "width": 640, "height": 480},
}


@pytest.fixture
def store(tmp_path):
    return ProfileStore(path=str(tmp_path / "profiles.json"))


def _spawn_payload(**over):
    # Shape device_manager builds for every spawn: the key is ALWAYS present.
    payload = {"robot_name": "so101", "mode": "real", "peer_id": "so101-arm-1",
               "port": "/dev/cu.usbmodem5AB0181806", "cameras": None, "robot_id": "arm1"}
    payload.update(over)
    return payload


def test_a_camera_less_spawn_does_not_forget_the_tuned_cameras(store):
    store.save("5AB0181806", _spawn_payload(cameras=TUNED))
    after = store.save("5AB0181806", _spawn_payload())          # watcher respawn, no cameras stated

    assert after["cameras"] == TUNED, "the operator's fps/resolution must survive a plain respawn"
    # Verbatim, not just the names: U19 is about the numbers.
    assert after["cameras"]["top"]["fps"] == 15
    assert after["cameras"]["wrist"]["width"] == 640


def test_reload_from_disk_keeps_them_too(store, tmp_path):
    store.save("5AB0181806", _spawn_payload(cameras=TUNED))
    store.save("5AB0181806", _spawn_payload())

    reopened = ProfileStore(path=str(tmp_path / "profiles.json"))
    assert reopened.get("5AB0181806")["cameras"] == TUNED


def test_new_cameras_replace_the_old_ones(store):
    store.save("5AB0181806", _spawn_payload(cameras=TUNED))
    changed = {"top": {"index_or_path": 0, "fps": 30, "width": 1920, "height": 1080}}
    after = store.save("5AB0181806", _spawn_payload(cameras=changed))

    assert after["cameras"] == changed, "a reconfigure must win over the memory"


def test_an_explicit_empty_mapping_forgets_them(store):
    # Going back to joints-only has to remain expressible, or the memory becomes a trap.
    store.save("5AB0181806", _spawn_payload(cameras=TUNED))
    after = store.save("5AB0181806", _spawn_payload(cameras={}))

    assert after["cameras"] == {}


def test_the_measured_role_still_survives_alongside(store):
    store.save("5AB0181806", _spawn_payload(cameras=TUNED))
    store.record_role("5AB0181806", {"role": "follower", "volts": 12.6})
    after = store.save("5AB0181806", _spawn_payload())

    assert after["role"] == "follower", "the older carry-over must not regress"
    assert after["cameras"] == TUNED
    assert after["role_volts"] == pytest.approx(12.6)


def test_nothing_is_invented_for_a_board_that_never_had_cameras(store):
    after = store.save("5AB0181806", _spawn_payload())
    assert after.get("cameras") is None, "silence must stay silence"
