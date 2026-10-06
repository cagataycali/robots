"""A page write reaches neither a gate-bearing variable nor a file path of its choosing.

The env fence in :mod:`strands_robots.dashboard.config_api` refuses a ``POST /api/config``
env write to ``STRANDS_TRUST_REMOTE_CODE`` and every ``STRANDS_MESH_*`` name. Three
neighbouring doors stayed open: :func:`~strands_robots.dashboard.settings.apply_mesh_env`
exported ``mesh.policy_type_allow`` and ``runtime.trust_remote_code`` into ``os.environ``
from a page-writable settings file; ``STRANDS_DASH_RECORD_CRUMB`` was page-writable and
:func:`~strands_robots.dashboard.record_crash.crumb_path` honoured it verbatim, so the
crumb was created and unlinked wherever the page said; and ``ALLOWED_ENV_KEYS`` was built
from the display list, so showing a key made it writable.
"""

from __future__ import annotations

import json
import os

import pytest

from strands_robots.dashboard import config_api, record_crash, settings

_SCHEMA_ENV = tuple(env for keys in settings._SCHEMA.values() for env, _default in keys.values() if env)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A scratch settings file and no schema variable in the environment."""
    path = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "SETTINGS_FILE", path)
    for env_name in _SCHEMA_ENV:
        monkeypatch.setenv(env_name, "")  # recorded first, so teardown restores
        monkeypatch.delenv(env_name)
    yield path
    settings.clear_overrides()
    settings.load(refresh=True)


def test_a_stored_gate_is_not_exported_and_the_transport_knobs_are(store):
    store.write_text(
        json.dumps(
            {
                "mesh": {"policy_type_allow": ["planted"], "connect": ["tcp/peer:7447"], "port": 7448},
                "runtime": {"trust_remote_code": True},
            }
        )
    )
    settings.clear_overrides()
    settings.load(refresh=True)

    applied = settings.apply_mesh_env()

    for gate in ("STRANDS_TRUST_REMOTE_CODE", "STRANDS_MESH_POLICY_TYPE_ALLOW"):
        assert gate not in applied and gate not in os.environ, gate
    assert applied == {"ZENOH_CONNECT": "tcp/peer:7447", "STRANDS_MESH_PORT": "7448"}


def test_every_name_a_setting_may_export_is_reviewed():
    """The export set is a closed subset of the schema, and the gate-bearing names stay out."""
    assert settings.EXPORTED_ENV <= set(_SCHEMA_ENV)
    assert not settings.EXPORTED_ENV & config_api.GATE_BEARING_ENV_KEYS
    for env_name in set(settings.MESH_ENV.values()) - settings.EXPORTED_ENV:
        assert config_api.env_key_gate_bearing(env_name), env_name


@pytest.mark.parametrize("key", ["STRANDS_DASH_RECORD_CRUMB", "STRANDS_ROBOTS_VIDEO_ROOT"])
def test_a_key_that_names_a_write_path_is_shown_but_not_page_writable(key):
    assert config_api.env_entry_error(key, "/tmp/chosen") is not None
    assert key in config_api.SHOWN_ENV_KEYS


def test_a_display_only_key_is_not_writable_by_being_shown(monkeypatch):
    monkeypatch.setattr(config_api, "INTERESTING_ENV", [*config_api.INTERESTING_ENV, "STRANDS_NEW_DISPLAY_KEY"])
    assert config_api.env_entry_error("STRANDS_NEW_DISPLAY_KEY", "x") is not None
    assert "STRANDS_NEW_DISPLAY_KEY" not in config_api.ALLOWED_ENV_KEYS


@pytest.mark.parametrize(
    ("override", "inside"),
    [
        (None, True),
        ("~/.strands_dashboard/sessions/crumb.json", True),
        ("{outside}/chosen/path.json", False),
        ("~/.strands_dashboard/../escaped.json", False),
        ("~/.strands_dashboard", False),
    ],
)
def test_the_crumb_stays_in_the_service_directory(tmp_path, monkeypatch, override, inside):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    service = (home / ".strands_dashboard").resolve()
    if override is None:
        monkeypatch.delenv("STRANDS_DASH_RECORD_CRUMB", raising=False)
    else:
        monkeypatch.setenv("STRANDS_DASH_RECORD_CRUMB", override.format(outside=tmp_path / "outside"))

    path = record_crash.crumb_path()
    record_crash.write_crumb({"dataset": "d"})

    assert path.is_relative_to(service) and path != service
    assert path.exists()
    if not inside:
        assert path == service / "record_session.json"
        assert not (tmp_path / "outside").exists()
        assert not (home / "escaped.json").exists()
    record_crash.clear_crumb()
    assert not path.exists()
