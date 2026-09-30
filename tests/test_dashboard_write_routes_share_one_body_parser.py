"""Every dashboard write route reads its body through one parser, so a bad body is a 4xx, never a 500.

``routes_auth._json_body`` is the dashboard's body parser: a body that is not
``application/json`` is a 415 (a ``text/plain`` write is a no-preflight simple
request, the reason that rule exists), a body that does not parse is a 400, and
an empty body is ``{}``. The sim, config and consent routes used to call
``request.json()`` themselves, so a truncated or non-JSON body raised
``JSONDecodeError`` out of the handler as a 500 with a traceback in the log, and
a ``text/plain`` write was accepted. The route table is walked from the source,
so a new write route that parses its own body fails here too.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import settings, sim_session  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402
from tests.test_dashboard_sim_routes import OWN_PAGE, FakeEngine  # noqa: E402

DASHBOARD = Path(__file__).resolve().parents[1] / "strands_robots" / "dashboard"

#: Write routes that take a JSON object body. ``{sid}`` is a live fake session.
ROUTES = [
    "/api/sim",
    "/api/sim/{sid}/joints",
    "/api/config",
    "/api/consent",
    "/api/consent/revoke",
]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(sim_session, "_default_factory", FakeEngine)
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings.clear_overrides()
    settings.load(refresh=True)
    app = create_app()
    with TestClient(app, headers=OWN_PAGE, raise_server_exceptions=False) as c:
        r = c.post("/api/sim", json={"robot": "so101"})
        assert r.status_code == 201, r.text
        c.sid = r.json()["id"]
        yield c
    app.state.safety.store.shutdown()


@pytest.mark.parametrize("route", ROUTES)
def test_a_body_that_is_not_json_is_a_400_not_a_500(client, route):
    r = client.post(
        route.format(sid=client.sid), content=b'{"robot": "so1', headers={"content-type": "application/json"}
    )
    assert r.status_code == 400, (r.status_code, r.text)
    assert "not JSON" in r.json()["error"]


@pytest.mark.parametrize("route", ROUTES)
def test_a_text_plain_body_is_refused_like_every_other_write(client, route):
    r = client.post(route.format(sid=client.sid), content=b'{"robot": "so101"}', headers={"content-type": "text/plain"})
    assert r.status_code == 415, (r.status_code, r.text)


@pytest.mark.parametrize("route", ROUTES)
def test_a_json_array_is_named_as_the_wrong_shape(client, route):
    r = client.post(route.format(sid=client.sid), json=[1, 2])
    assert r.status_code == 400, (r.status_code, r.text)


def test_the_well_formed_calls_still_work(client):
    assert client.post(f"/api/sim/{client.sid}/joints", json={"positions": [0.1]}).status_code == 200
    assert client.post("/api/sim", json={"robot": "so100"}).status_code == 201


def test_no_route_module_parses_a_body_itself():
    """Only the parser calls ``request.json()``; every route goes through it."""
    offenders = []
    for path in sorted(DASHBOARD.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.AsyncFunctionDef) and node.name != "_json_body":
                for call in ast.walk(node):
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "json"
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "request"
                    ):
                        offenders.append(f"{path.name}:{call.lineno} {node.name}")
    assert offenders == [], offenders
