"""The suite must not share a profiles file with the operator's live dashboard (Q84 fallout).

Found in the wild: ~/.strands_dashboard/profiles.json carried an entry named "q1-bad" whose camera
config is the invalid {"main": 3} from a regression fixture — a test had written a robot definition
into the production file. Those entries are exactly what autospawn spawns from on real hardware.
"""
from __future__ import annotations

import os
from pathlib import Path

from strands_robots.dashboard import device_manager as dm
from strands_robots.dashboard.device_manager import DeviceManager


def _real_path() -> str:
    return os.path.join(Path.home(), ".strands_dashboard", "profiles.json")


def test_the_default_construction_does_not_land_on_the_operators_file():
    # DeviceManager() with no argument is the shape used by create_app() and by two existing tests.
    assert DeviceManager().profiles.path != _real_path()


def test_the_app_state_manager_is_isolated_too():
    from strands_robots.dashboard.server import create_app

    app = create_app()
    assert app.state.devices.profiles.path != _real_path()


def test_the_constant_itself_is_untouched():
    # The redirect must come from the environment, not from patching the default — the running
    # dashboard has to keep resolving to the operator's real file.
    assert dm.DEFAULT_PROFILES_PATH == _real_path()


def test_a_save_cannot_reach_the_real_file(tmp_path):
    before = Path(_real_path()).read_bytes() if Path(_real_path()).exists() else None
    mgr = DeviceManager()
    mgr.profiles.save("TESTONLY", {"robot_name": "so101", "mode": "sim", "peer_id": "iso-test"})
    after = Path(_real_path()).read_bytes() if Path(_real_path()).exists() else None
    assert before == after, "a test just wrote into the operator's live profiles.json"
    # ...and it landed somewhere, so the test proves isolation rather than a no-op save.
    assert "TESTONLY" in Path(mgr.profiles.path).read_text()
