"""/api/training/trainers publishes the field vocabulary this server accepts (Q78).

Why: a dashboard process is long-lived (the one on this Mac had been up for days) and the PWA bundle is
rebuilt often, so the two halves of this app are routinely different ages. Measured against the live
dashboard on 2026-08-20: the freshly built form offered the new `val_episodes` holdout and the running
server answered

    unknown field(s): val_episodes. Valid fields: provider, dataset_root, ...

a true sentence no operator can act on -- the actual remedy is "restart the dashboard", and nothing in
that refusal says so. So the form asks the server what it takes rather than assuming its own age.

`fields` is a NEW key next to `unsupported`: `trainers` keeps its shape forever, because a cached older
bundle renders that list directly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from strands_robots.dashboard import server as srv
from strands_robots.dashboard import training


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """This machine's real settings.json carries a live auth token, so an un-isolated TestClient
    gets a 401 before it reaches the route (the same trap tests/test_dashboard_api_404.py documents),
    and Q62's lesson applies: a settings override outlives the test that made it."""
    from strands_robots.dashboard import auth
    from strands_robots.dashboard import settings as dsettings

    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None
    auth._cache_key = None
    auth._cache = {}


class _StubBridge:
    peers: dict = {}

    def snapshot(self):
        return {"peers": {}}

    def start(self):
        pass

    def stop(self):
        pass


def _trainers_payload() -> dict:
    r = TestClient(srv.create_app(bridge=_StubBridge())).get("/api/training/trainers")
    assert r.status_code == 200, r.text
    return r.json()


def test_fields_is_exactly_the_accepted_vocabulary() -> None:
    body = _trainers_payload()
    assert body["fields"] == list(training.SPEC_KEYS)
    # The point of the key: the newest field is visible to a form that wants to offer it, so a
    # form talking to an OLDER server can tell that this one is not there.
    assert "val_episodes" in body["fields"]


def test_every_published_field_is_actually_accepted() -> None:
    """A published name the validator refuses would be a worse lie than silence."""
    _spec, err = training._spec_kwargs({k: 1 for k in training.SPEC_KEYS})
    assert err is None, err


def test_trainers_and_unsupported_keep_their_shape_for_cached_bundles() -> None:
    body = _trainers_payload()
    assert isinstance(body["trainers"], list) and all(isinstance(t, str) for t in body["trainers"])
    assert isinstance(body["unsupported"], dict)
