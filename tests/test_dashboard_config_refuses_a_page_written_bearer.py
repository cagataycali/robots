"""A settings write from the page cannot plant the dashboard's standing bearer.

``security.auth_token`` is the bearer every ``/api`` and ``/ws`` request must present, and
:func:`~strands_robots.dashboard.access.caller` honours it independently of passkey
enrolment: a value written to ``settings.json`` keeps admitting its holder after the
operator has deleted every passkey and enrolled again. Its env spelling,
``DASHBOARD_AUTH_TOKEN``, is in ``GATE_BEARING_ENV_KEYS`` and so is refused on the ``env``
half of a ``POST /api/config`` body, but the ``security`` half of the same body copied the
section wholesale into the patch and ``POST /api/settings`` handed its body to the store
directly. Either door let any admitted session, the pre-enrolment loopback posture
included, write a durable credential the Settings drawer then reported as "auth enabled".

Both doors now share one fence, ``REFUSED_SETTINGS_KEYS``: a credential-bearing settings
key is refused with a reason naming the out-of-band way to set it. Clearing one stays
page-writable, because that is the operator's remedy for a bearer they did not set.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import config_api, settings  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402
from tests._dashboard_bootstrap import bootstrap_headers, configure_bootstrap  # noqa: E402

PLANTED = "planted-by-a-visitor"


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """A fresh settings file and env file, no env override, auth store empty."""
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config_api, "ENV_FILE", tmp_path / ".env")
    configure_bootstrap(monkeypatch)  # the fresh-install open posture admits the page only with the bootstrap proof
    settings.clear_overrides()
    settings.load(refresh=True)
    yield tmp_path
    settings.clear_overrides()
    settings.load(refresh=True)


@pytest.fixture()
def client(isolated):
    return TestClient(create_app(), headers=bootstrap_headers())


def _stored_token(isolated) -> object:
    path = isolated / "settings.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("security", {}).get("auth_token")


def test_the_roster_names_every_credential_bearing_settings_key():
    """Derived from the schema: a key whose env spelling names a secret must be refused."""
    for section, keys in settings._SCHEMA.items():
        for key, (env_name, _default) in keys.items():
            if env_name and config_api.is_secret(env_name):
                assert (section, key) in config_api.REFUSED_SETTINGS_KEYS, f"{section}.{key} ({env_name})"
    for section, key in config_api.REFUSED_SETTINGS_KEYS:
        assert key in settings._SCHEMA[section], f"{section}.{key} is not a settings key"
    assert ("security", "auth_token") in config_api.REFUSED_SETTINGS_KEYS


def test_the_two_fences_agree_on_every_refused_key():
    """The env spelling of a refused settings key is refused on the env half as well."""
    for section, key in config_api.REFUSED_SETTINGS_KEYS:
        env_name = settings._SCHEMA[section][key][0]
        assert env_name and config_api.env_key_gate_bearing(env_name), f"{section}.{key} -> {env_name}"


def test_apply_refuses_a_bearer_and_stores_nothing(isolated):
    result = config_api.apply({"security": {"auth_token": PLANTED}})
    assert result["errors"] and "security.auth_token" in result["errors"][0]
    assert "DASHBOARD_AUTH_TOKEN" in result["errors"][0], "the reason names the out-of-band way"
    assert PLANTED not in result["errors"][0], "the refused value is not echoed"
    assert result["applied"] == []
    assert settings.get("security", "auth_token") is None
    assert _stored_token(isolated) is None


def test_apply_refuses_the_bearer_but_keeps_the_rest_of_the_section(isolated):
    result = config_api.apply({"security": {"auth_token": PLANTED, "cors_origins": ["https://a.example"]}})
    assert any("security.auth_token" in e for e in result["errors"])
    assert settings.get("security", "auth_token") is None
    assert settings.get("security", "cors_origins") == ["https://a.example"]


def test_apply_still_clears_a_configured_bearer(isolated):
    settings.update({"security": {"auth_token": "set-on-the-host"}})
    result = config_api.apply({"security": {"auth_token": None}})
    assert result["errors"] == []
    assert settings.get("security", "auth_token") is None
    assert _stored_token(isolated) is None


def test_apply_treats_an_empty_bearer_as_no_change_not_a_refusal(isolated):
    """The drawer's save button sends ``auth_token: null`` next to the CORS field."""
    result = config_api.apply({"security": {"auth_token": None, "cors_origins": ["https://b.example"]}})
    assert result["errors"] == []
    assert settings.get("security", "cors_origins") == ["https://b.example"]


def test_post_config_from_the_open_posture_cannot_plant_a_bearer(client, isolated):
    assert client.get("/api/whoami").json()["via"] == "loopback"
    r = client.post("/api/config", json={"security": {"auth_token": PLANTED}})
    assert r.status_code == 422, r.text
    assert "security.auth_token" in r.json()["error"]
    assert _stored_token(isolated) is None
    refused = client.get("/api/whoami", headers={"authorization": f"Bearer {PLANTED}"})
    # On a fresh install a bearer that is not the bootstrap proof is refused outright (f002).
    assert refused.status_code == 401 and "via" not in refused.json(), "the planted value admits nobody"


def test_post_settings_from_the_open_posture_cannot_plant_a_bearer(client, isolated):
    r = client.post("/api/settings", json={"security": {"auth_token": PLANTED}})
    assert r.status_code == 422, r.text
    assert any("security.auth_token" in e for e in r.json()["errors"])
    assert r.json()["changed"] == []
    assert _stored_token(isolated) is None


def test_post_settings_still_writes_what_is_not_a_credential(client, isolated):
    r = client.post("/api/settings", json={"security": {"cors_origins": ["https://c.example"]}})
    assert r.status_code == 200, r.text
    assert r.json()["changed"] == ["security.cors_origins"]


def test_no_settings_key_reaches_the_store_with_a_bearer_in_the_same_request(client, isolated):
    """A refused key is dropped before the store sees the patch; the rest of the patch lands."""
    r = client.post("/api/settings", json={"security": {"auth_token": PLANTED}, "agent": {"temperature": 0.3}})
    assert r.status_code == 422, r.text
    assert r.json()["changed"] == ["agent.temperature"]
    assert _stored_token(isolated) is None
