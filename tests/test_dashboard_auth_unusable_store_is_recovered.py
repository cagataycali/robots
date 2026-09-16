"""A credential store that parses and still cannot serve as one must not answer 500 forever.

`_load()` recovered a store it could not READ - an OSError or unparseable JSON went to
`_preserve_corrupt`, which keeps the bytes and comes up on a fresh default. A store that
parsed was handed back as-is, whatever it held, and every reader in the module then indexed
it without a second look. So a file that is valid JSON and not a store put the fault inside
whichever route asked:

* no `jwt_secret` -> `KeyError` from `_jwt_secret`, i.e. a 500 on every route that verifies a
  session, for as long as the file sits there, while the same request WITHOUT a token was a
  clean 401. The remedy appeared in no response and no log.
* a non-string or empty `jwt_secret` -> quieter and worse: every token failed to verify (401
  forever) and signing in raised from PyJWT, so an operator saw "invalid session" and could
  never get a valid one.
* a top-level JSON array/string/number/null, or a `credentials` value that is not a list of
  records with an id -> `TypeError`/`AttributeError` out of `auth_enabled()` or
  `list_credentials()`, which is the login screen rather than one guarded route.

All of it is the same condition as an unparseable file, one step later, so it now takes the
same recovery: keep the bytes aside, record why, come up on a working default - and stay
sealed against enrollment from anywhere but the machine while it does.
"""

import json
import time
from types import SimpleNamespace

import jwt
import pytest
from fastapi import HTTPException

from strands_robots.dashboard import auth


class FakeRequest:
    def __init__(self, headers=None, client_host="127.0.0.1", scheme="http"):
        self.headers = headers or {"host": "localhost:8090"}
        self.client = type("C", (), {"host": client_host})()
        self.url = SimpleNamespace(scheme=scheme)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    for k in ("STRANDS_DASH_AUTH_ENABLED", "STRANDS_DASH_AUTH_RP_ID", "STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    auth._cache = {}
    auth._corrupt = None
    yield
    auth._corrupt = None


# Every way a store can parse and still be unusable, with the fault each one used to
# produce. The reason fragment is what an operator reads out of `store_corruption()` and
# the log, so it is pinned per row: a single "unusable" for all of them would leave the
# person holding the file no better off than the traceback did.
UNUSABLE = {
    "no jwt_secret": ('{"credentials": [{"id": "isolation-test", "public_key": "x"}]}', "no usable jwt_secret"),
    "null jwt_secret": ('{"jwt_secret": null, "credentials": []}', "no usable jwt_secret"),
    "numeric jwt_secret": ('{"jwt_secret": 42, "credentials": []}', "no usable jwt_secret"),
    "empty jwt_secret": ('{"jwt_secret": "", "credentials": []}', "no usable jwt_secret"),
    "a JSON array": ("[]", "not an object"),
    "a JSON string": ('"hello"', "not an object"),
    "a JSON null": ("null", "not an object"),
    "a JSON number": ("42", "not an object"),
    "credentials not a list": ('{"jwt_secret": "kept-secret", "credentials": 42}', "not a list of records"),
    "credentials a mapping": ('{"jwt_secret": "kept-secret", "credentials": {"a": 1}}', "not a list of records"),
    "a credential with no id": (
        '{"jwt_secret": "kept-secret", "credentials": [{"public_key": "x"}]}',
        "carrying an id",
    ),
    "a credential that is not a record": ('{"jwt_secret": "kept-secret", "credentials": ["x"]}', "carrying an id"),
}
unusable = pytest.mark.parametrize("body,reason", UNUSABLE.values(), ids=list(UNUSABLE))

#: A bearer of the right shape, minted against a secret this store never held - exactly what a
#: browser still holding a session from the previous store presents.
STALE_TOKEN = jwt.encode({"sub": "root", "exp": int(time.time()) + 3600}, "a-previous-secret", algorithm="HS256")


class TestTheRequestGetsAnAnswerRatherThanATraceback:
    @unusable
    def test_a_presented_token_is_refused_not_a_server_error(self, body, reason, tmp_path):
        (tmp_path / "auth.json").write_text(body)
        with pytest.raises(HTTPException) as e:
            auth.verify_token(STALE_TOKEN)
        assert e.value.status_code == 401, "a store nobody can use is not the caller's fault to pay 500 for"

    @unusable
    def test_the_login_screen_still_renders(self, body, reason, tmp_path):
        """`auth_enabled()` and `list_credentials()` feed the screen; either raising takes it down."""
        (tmp_path / "auth.json").write_text(body)
        assert auth.auth_enabled() is False, "the recovered default holds no passkey yet"
        assert auth.list_credentials() == []
        assert auth.status(FakeRequest())["setup_required"] is True

    @unusable
    def test_signing_in_works_again(self, body, reason, tmp_path):
        """The point of the recovery: a session can be minted and verified after it."""
        (tmp_path / "auth.json").write_text(body)
        claims = auth.verify_token(auth.issue_token("root", "root"))
        assert claims["sub"] == "root"


class TestTheOperatorsFileAndTheDiagnosis:
    @unusable
    def test_the_bytes_are_kept_and_a_working_store_takes_over(self, body, reason, tmp_path):
        path = tmp_path / "auth.json"
        path.write_text(body)

        auth._load()

        backups = list(tmp_path.glob("auth.json.corrupt-*"))
        assert len(backups) == 1 and backups[0].read_text() == body, "the file may hold the only credential id"
        assert isinstance(json.loads(path.read_text())["jwt_secret"], str)

    @unusable
    def test_the_reason_names_the_fault_rather_than_just_reporting_one(self, body, reason, tmp_path):
        (tmp_path / "auth.json").write_text(body)

        auth._load()

        damage = auth.store_corruption()
        assert damage, "a silent repair of the file that decides whether this dashboard is sealed"
        assert reason in damage["reason"], damage["reason"]

    @unusable
    def test_a_stranger_cannot_seize_the_dashboard_through_it(self, body, reason, tmp_path):
        """Quarantining empties `credentials`, so enrollment must be re-sealed - as for a corrupt file."""
        (tmp_path / "auth.json").write_text(body)
        auth._load()

        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest(client_host="203.0.113.9"), label="attacker")
        assert e.value.status_code == 403
        assert "corrupt-" in e.value.detail and "BOOTSTRAP_TOKEN" in e.value.detail


# A store is usable when this module's readers can serve it, not when it matches
# `_default_store()` field for field: absent optional keys and unknown extra ones are how a
# store written by another build arrives. Quarantining those would destroy working passkeys.
USABLE = {
    "the default store": json.dumps(auth._default_store()),
    "no credentials key at all": '{"jwt_secret": "kept-secret"}',
    "an empty credential list": '{"jwt_secret": "kept-secret", "credentials": []}',
    "a full credential record": (
        '{"jwt_secret": "kept-secret", "credentials": [{"id": "AAAA", "public_key": "x", '
        '"sign_count": 7, "name": "phone", "created": 1.0}]}'
    ),
    "keys this build does not know": '{"jwt_secret": "kept-secret", "credentials": [], "future_field": {"a": 1}}',
}


@pytest.mark.parametrize("body", USABLE.values(), ids=list(USABLE))
def test_a_usable_store_is_served_untouched(body, tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(body)

    store = auth._load()

    assert store == json.loads(body), "served as written - a repair here would destroy live passkeys"
    assert not list(tmp_path.glob("auth.json.corrupt-*"))
    assert auth.store_corruption() is None
    assert path.read_text() == body
