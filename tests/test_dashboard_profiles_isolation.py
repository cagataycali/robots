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


# --- the other three production stores under his home -------------------------------------------------

def test_the_auth_store_is_not_his_real_passkey_file():
    """His passkey store is the only door into robots.cagatay.my while he travels.

    26 test files redirect this themselves, which is exactly why it needs a floor: the 27th file to touch
    auth is the one that writes his real credentials, and it will not look like an auth test.
    """
    from strands_robots.dashboard import auth

    real = Path.home() / ".strands_dashboard" / "auth.json"
    assert auth._store_path() != real.resolve()


def test_registering_a_credential_leaves_his_real_store_byte_identical():
    from strands_robots.dashboard import auth

    real = Path.home() / ".strands_dashboard" / "auth.json"
    before = real.read_bytes() if real.exists() else None
    auth._save({"credentials": [{"id": "isolation-test", "public_key": "x"}]})
    after = real.read_bytes() if real.exists() else None
    assert before == after, "a test just wrote into his real passkey store"
    assert "isolation-test" in auth._store_path().read_text()   # ...and it did happen somewhere


def test_the_record_crumb_and_settings_file_are_redirected_too():
    from strands_robots.dashboard import record_crash, settings

    assert record_crash.crumb_path() != Path.home() / ".strands_dashboard" / "record_session.json"
    # settings.SETTINGS_FILE is a module constant resolved at import, so the env var alone cannot move it;
    # this asserts the attribute patch, i.e. the mechanism, not just the intent.
    assert Path.home() not in settings.SETTINGS_FILE.parents


def test_saving_config_cannot_drop_a_dotenv_into_the_repo(tmp_path, monkeypatch):
    """The fifth store, and the odd one out: .env resolves against the CURRENT DIRECTORY.

    A test that saves config would write .env into whatever tree pytest ran from — the repo root — and
    that file is not inert: the dashboard reads .env at startup, so test values for the trust and
    allowlist flags (the ones gating remote code execution) could become his live configuration.
    """
    from strands_robots.dashboard import config_api

    cwd_before = sorted(Path.cwd().glob(".env"))
    config_api.upsert_env_file({"STRANDS_TRUST_REMOTE_CODE": "0"})
    assert sorted(Path.cwd().glob(".env")) == cwd_before, "a test just wrote .env into the working tree"
    assert Path(config_api.ENV_FILE).exists() and "STRANDS_TRUST_REMOTE_CODE" in Path(config_api.ENV_FILE).read_text()
    assert Path.home() not in Path(config_api.ENV_FILE).parents
