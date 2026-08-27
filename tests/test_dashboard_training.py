"""Training pipeline of the operator dashboard (consolidated).

Consolidated verbatim from: test_dashboard_training_export_base_model.py, test_dashboard_training_fields.py, test_dashboard_training_spec_validation.py.
Each section keeps its original tests unchanged.
"""

from __future__ import annotations

from unittest import mock

from fastapi.testclient import TestClient

from strands_robots.dashboard import server as srv
from strands_robots.dashboard import training

# ============================================================================
# from tests/test_dashboard_training_export_base_model.py
# Export must forward base_model - the spec revalidates on export, so a
# ============================================================================


def test_export_forwards_base_model():
    with mock.patch("strands_robots.tools.train_policy.train_policy") as tp:
        tp.return_value = {"content": [{"text": "ok"}], "status": "success"}
        training.export("mock", "/out", "/data", None, "lerobot/smolvla_base")
    kwargs = tp.call_args.kwargs
    assert kwargs["base_model"] == "lerobot/smolvla_base"
    assert kwargs["action"] == "export"


def test_export_without_base_model_sends_empty_string():
    # ACT-from-scratch trains with base_model="" - export must mirror that
    with mock.patch("strands_robots.tools.train_policy.train_policy") as tp:
        tp.return_value = {"content": [{"text": "ok"}], "status": "success"}
        training.export("mock", "/out", "/data")
    assert tp.call_args.kwargs["base_model"] == ""


# ============================================================================
# from tests/test_dashboard_training_fields.py
# /api/training/trainers publishes the field vocabulary this server accepts (Q78).
# ============================================================================


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
    r = TestClient(srv.create_app(bridge=_StubBridge())).get("/api/training/trainers")  # type: ignore[arg-type]
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


# ============================================================================
# from tests/test_dashboard_training_spec_validation.py
# Q6: /api/training/validate + submit must refuse unknown fields as JSON, never 500.
# ============================================================================


class TestSpecKwargs:
    def test_clean_body_passes_through(self):
        kwargs, err = training._spec_kwargs({"provider": "lerobot_local", "steps": 500, "dataset_root": "/tmp/d"})
        assert err is None
        assert kwargs == {"provider": "lerobot_local", "steps": 500, "dataset_root": "/tmp/d"}

    def test_unknown_keys_named_and_sorted(self):
        kwargs, err = training._spec_kwargs({"policy": "x", "dataset": "y"})
        assert kwargs is None
        assert err["status"] == "error"
        # both offenders named, alphabetical, so the message is deterministic
        assert "unknown field(s): dataset, policy" in err["text"]

    def test_error_teaches_the_valid_vocabulary(self):
        _, err = training._spec_kwargs({"nope": 1})
        for field in ("dataset_root", "base_model", "steps", "method"):
            assert field in err["text"]

    def test_none_values_dropped_not_forwarded(self):
        kwargs, err = training._spec_kwargs({"provider": "lerobot_local", "seed": None})
        assert err is None
        assert "seed" not in kwargs

    def test_action_key_tolerated(self):
        # callers historically included action; it is routing, not spec
        kwargs, err = training._spec_kwargs({"action": "validate", "steps": 10})
        assert err is None
        assert kwargs == {"steps": 10}


class TestValidateAndSubmitRefuseBadBodies:
    def test_validate_bad_body_is_structured_error(self):
        # the exact body from the BUGS.md Q6 repro
        res = training.validate({"dataset": "nobody/nothing-zz", "policy": "nope"})
        assert res["status"] == "error"
        assert "unknown field(s)" in res["text"]

    def test_submit_bad_body_is_structured_error_and_records_no_job(self, tmp_path, monkeypatch):
        monkeypatch.setattr(training, "JOBS_FILE", tmp_path / "jobs.json")
        res = training.submit({"stepz": 100})
        assert res["status"] == "error"
        assert "stepz" in res["text"]
        assert training.jobs() == []

    def test_validate_and_submit_share_one_vocabulary(self):
        # SPEC_KEYS drifting between the two would let a field validate but
        # silently vanish on submit - the worst kind of "it worked in check"
        v = training.validate({"definitely_not_a_field": 1})
        s = training.submit({"definitely_not_a_field": 1})
        assert v["text"] == s["text"]
