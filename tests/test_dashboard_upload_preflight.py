"""Q72: the upload verdict must be reachable BEFORE the recording, not only after it."""

from __future__ import annotations

from strands_robots.dashboard.upload_preflight import destination, upload_preflight

AUTHED = {"authenticated": True, "user": "cagataycali", "detail": None}
NO_TOKEN = {"authenticated": False, "user": None, "detail": "no HF token on this machine"}
REVOKED = {"authenticated": False, "user": None, "detail": "HF token present but rejected (HTTPError)"}


def test_a_logged_in_user_gets_the_full_destination_not_just_the_name():
    v = upload_preflight(dataset="so101-pick", auth=AUTHED)
    assert v["ok"] is True
    # The namespace is invisible in the UI's old wording ("publishes as so101-pick"); it is where
    # the data actually lands.
    assert v["destination"] == "cagataycali/so101-pick"
    assert "cagataycali/so101-pick" in v["detail"]


def test_no_credential_refuses_the_tick_rather_than_warning_about_it():
    v = upload_preflight(dataset="so101-pick", auth=NO_TOKEN)
    assert v["ok"] is False
    assert v["state"] == "no_credential"
    assert v["needs_force"] is False, "there is nothing to insist on: the push cannot work"
    # The sentence must carry the cost AND the fix.
    assert "huggingface-cli login" in v["detail"]
    assert "no retry" in v["detail"]
    assert "stay on this machine" in v["detail"]


def test_a_revoked_token_is_not_described_as_anonymity():
    v = upload_preflight(dataset="so101-pick", auth=REVOKED)
    assert v["state"] == "credential_rejected"
    assert "REJECTED" in v["detail"]
    assert v["state"] != upload_preflight(dataset="so101-pick", auth=NO_TOKEN)["state"]


def test_someone_elses_namespace_is_refused_but_continuable():
    v = upload_preflight(dataset="HashtagRobotics/tic-tac-toe", auth=AUTHED)
    assert v["ok"] is False
    assert v["state"] == "foreign_namespace"
    # An org membership cannot be established from here, so this refusal admits its uncertainty
    # instead of calling the operator wrong.
    assert v["needs_force"] is True
    assert "organisation" in v["detail"]
    assert "cagataycali" in v["detail"] and "HashtagRobotics" in v["detail"]


def test_your_own_namespace_written_out_is_fine():
    v = upload_preflight(dataset="cagataycali/so101-pick", auth=AUTHED)
    assert v["ok"] is True
    assert v["destination"] == "cagataycali/so101-pick"


def test_an_unnamed_session_says_so_instead_of_guessing():
    for empty in (None, "", "   ", "/"):
        v = upload_preflight(dataset=empty, auth=AUTHED)
        assert v["ok"] is False, empty
        assert v["state"] == "no_dataset"
        assert v["destination"] is None


def test_destination_is_pure_and_admits_what_it_cannot_know():
    assert destination("thing", "me") == "me/thing"
    assert destination("them/thing", None) == "them/thing"
    assert destination("thing", None) is None, "a bare name without a user is not a repo id"
    assert destination("  spaced  ", "me") == "me/spaced"


def test_credentials_missing_outranks_a_foreign_namespace():
    """Two problems at once must report the one that is CERTAIN."""
    v = upload_preflight(dataset="HashtagRobotics/x", auth=NO_TOKEN)
    assert v["state"] == "no_credential"


def test_the_route_answers_from_the_live_session_and_never_writes(monkeypatch):
    """GET /api/record/upload-preflight is read-only and reads THIS session's dataset name."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from strands_robots.dashboard import checkpoints, record_api

    class FakeController:
        closed = False

        def session(self) -> dict[str, object]:
            return {"open": True, "dataset": "so101-pick", "episodes": []}

        def close(self, body=None):  # pragma: no cover - must never be reached
            FakeController.closed = True
            raise AssertionError("a preflight must not touch the session")

    monkeypatch.setattr(checkpoints, "hf_auth_state", lambda: dict(AUTHED))
    app = FastAPI()
    app.include_router(record_api.build_router(FakeController()))  # type: ignore[arg-type]
    with TestClient(app) as client:
        body = client.get("/api/record/upload-preflight").json()
    assert body["destination"] == "cagataycali/so101-pick"
    assert body["ok"] is True
    assert FakeController.closed is False


def test_the_route_survives_a_session_with_no_dataset(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from strands_robots.dashboard import checkpoints, record_api

    class Empty:
        def session(self) -> dict[str, object]:
            return {"open": False}

    monkeypatch.setattr(checkpoints, "hf_auth_state", lambda: dict(NO_TOKEN))
    app = FastAPI()
    app.include_router(record_api.build_router(Empty()))  # type: ignore[arg-type]
    with TestClient(app) as client:
        body = client.get("/api/record/upload-preflight").json()
    # No dataset AND no token: report the certain one first (see the pure test above).
    assert body["ok"] is False
    assert body["state"] == "no_dataset"


# --- Q78: the destination already exists on the Hub -------------------------------------------
#
# push_to_hub does not create a second repo and does not refuse — it uploads INTO the existing one.
# Recording refuses to reuse a local dataset dir (Q39), so what is being finished is always a NEW,
# shorter dataset: publishing rewrites meta over a longer published history while the old episode
# files stay behind. Nothing warned about that anywhere.

Q78_AUTHED = {"authenticated": True, "user": "cagatay"}


def test_existing_destination_refuses_but_stays_the_operators_call():
    r = upload_preflight(
        dataset="so101-cubes", auth=Q78_AUTHED, existing={"exists": True, "episodes": 40}
    )
    assert r["ok"] is False
    assert r["state"] == "destination_exists"
    # Replacing their own earlier take deliberately is legitimate — only they know if that is this.
    assert r["needs_force"] is True
    assert r["destination"] == "cagatay/so101-cubes"
    assert "40 episode" in r["detail"]
    # The consequence, not just the fact: it is an upload INTO that repo.
    assert "INTO that repo" in r["detail"]


def test_existing_destination_without_a_count_still_refuses():
    r = upload_preflight(dataset="so101-cubes", auth=Q78_AUTHED, existing={"exists": True})
    assert r["state"] == "destination_exists"
    assert "episode(s)" not in r["detail"]


def test_a_free_destination_is_unaffected():
    for existing in ({"exists": False}, {}, None):
        r = upload_preflight(dataset="so101-cubes", auth=Q78_AUTHED, existing=existing)
        assert r["ok"] is True, existing
        assert r["state"] == "ready"


def test_no_evidence_is_treated_as_nothing_there():
    # A Hub lookup that failed (no network, 5xx) must never block a publish: silence keeps the old
    # behaviour exactly.
    assert upload_preflight(dataset="x", auth=Q78_AUTHED, existing={})["ok"] is True


def test_auth_refusals_still_outrank_an_existing_destination():
    # The credential failure is certain and is fixed differently; it must be the sentence shown.
    r = upload_preflight(
        dataset="so101-cubes", auth={"authenticated": False}, existing={"exists": True}
    )
    assert r["state"] in ("no_credential", "credential_rejected")
    r2 = upload_preflight(
        dataset="someorg/cubes", auth=Q78_AUTHED, existing={"exists": True, "episodes": 3}
    )
    assert r2["state"] == "foreign_namespace"


def test_an_empty_dataset_name_outranks_everything():
    r = upload_preflight(dataset="  ", auth=Q78_AUTHED, existing={"exists": True})
    assert r["state"] == "no_dataset"


def test_hub_facts_never_raises_and_says_nothing_without_a_namespace():
    from strands_robots.dashboard import record_api

    # No namespace yet = nothing to ask the Hub about.
    assert record_api._hub_facts("bare-name") == {}
    assert record_api._hub_facts("") == {}
    assert record_api._hub_facts(None) == {}


def test_hub_facts_reports_existence_and_swallows_a_broken_hub(monkeypatch):
    import sys
    import types

    from strands_robots.dashboard import record_api

    class FakeApi:
        def repo_exists(self, repo_id, repo_type):  # noqa: ARG002
            return True

        def dataset_info(self, repo_id):  # noqa: ARG002
            raise RuntimeError("hub is having a day")

    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = FakeApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    # Existence is the fact that matters; the count is a nicety and its failure is swallowed.
    assert record_api._hub_facts("cagatay/cubes") == {"exists": True, "episodes": None}

    class ExplodingApi:
        def repo_exists(self, repo_id, repo_type):  # noqa: ARG002
            raise RuntimeError("no network")

    mod.HfApi = ExplodingApi  # type: ignore[attr-defined]
    assert record_api._hub_facts("cagatay/cubes") == {}
